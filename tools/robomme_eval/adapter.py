from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Sequence, TYPE_CHECKING, cast

import numpy as np
import torch

from vla_scratch.helpers.data import create_dataset
from vla_scratch.transforms.data_keys import (
    FRONT_IMAGE_KEY,
    GENERATION_ANSWER_KEY,
    GENERATION_PROMPT_KEY,
    PROCESSED_ACTION_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
    WRIST_IMAGE_KEY,
)

if TYPE_CHECKING:
    from vla_scratch.datasets.config import DataConfig
    from vla_scratch.policies.base import BasePolicy
    from vla_scratch.policies.config import PolicyConfig
    from vla_scratch.transforms.base import TransformFn
    from vla_scratch.transforms.data_types import DataSample


def image_chw(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = np.array(image, copy=True)
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    return tensor.contiguous().to(torch.uint8)


def chw_to_hwc(image: np.ndarray | torch.Tensor) -> np.ndarray:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW/HWC image with 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    return tensor.detach().cpu().numpy().astype(np.uint8)


def as_hwc_frame_list(images: np.ndarray | torch.Tensor) -> list[np.ndarray]:
    tensor = torch.as_tensor(images)
    if tensor.ndim == 3:
        return [chw_to_hwc(tensor)]
    if tensor.ndim == 4:
        return [chw_to_hwc(tensor[i]) for i in range(tensor.shape[0])]
    raise ValueError(f"Expected image tensor with 3 or 4 dims, got {tuple(tensor.shape)}")


def pack_state(joint_state: Any, gripper_state: Any) -> np.ndarray:
    joint = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    gripper = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
    if gripper.size == 0:
        raise ValueError("gripper_state is empty")
    return np.concatenate([joint, gripper[:1]], axis=0, dtype=np.float32)


def select_even(items: Sequence[Any], count: int) -> list[Any]:
    if count <= 1:
        return [items[-1]]
    if len(items) <= count:
        return [items[0]] * (count - len(items)) + list(items)
    positions = np.linspace(0, len(items) - 1, count, dtype=np.int32)
    return [items[int(pos)] for pos in positions]


def select_recent(items: Sequence[Any], count: int, stride: int) -> list[Any]:
    if count <= 1:
        return [items[-1]]
    selected = []
    for offset in range((count - 1) * stride, -1, -stride):
        idx = max(0, len(items) - 1 - offset)
        selected.append(items[idx])
    return selected


def initialize_policy_dims(data_cfg: "DataConfig", policy_cfg: "PolicyConfig") -> Any:
    data_cfg.action_horizon = policy_cfg.action_horizon
    data_cfg.state_history = policy_cfg.state_history

    dataset = create_dataset(data_cfg, policy_cfg)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; unable to infer policy dimensions.")

    data_sample: "DataSample" = dataset[0][0]
    if data_sample.action_chunk is None:
        raise ValueError("Dataset sample has no actions; unable to infer action_dim.")
    if data_sample.observation.state is None:
        raise ValueError("Dataset sample has no state; unable to infer state_dim.")

    if policy_cfg.action_dim is None:
        policy_cfg.action_dim = int(data_sample.action_chunk.actions.shape[-1])
    if policy_cfg.state_dim is None:
        policy_cfg.state_dim = int(data_sample.observation.state.shape[-1])
    return dataset


def resolve_action_bounds(dataset: Any) -> tuple[np.ndarray, np.ndarray] | None:
    def extract_bounds(raw_stats: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if not isinstance(raw_stats, dict):
            return None
        if "min" not in raw_stats or "max" not in raw_stats:
            return None
        return (
            np.asarray(raw_stats["min"], dtype=np.float32),
            np.asarray(raw_stats["max"], dtype=np.float32),
        )

    current = dataset
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        stats = getattr(current, "stats", None)
        if isinstance(stats, dict):
            bounds = extract_bounds(stats.get("actions_orig") or stats.get("actions"))
            if bounds is not None:
                return bounds

        root = getattr(current, "root", None)
        metadata = getattr(current, "metadata", None)
        inner_dataset = getattr(current, "dataset", None)
        if metadata is None and inner_dataset is not None:
            metadata = getattr(inner_dataset, "meta", None)
        if root is None and inner_dataset is not None:
            root = getattr(inner_dataset, "root", None)
        if metadata is not None:
            meta_stats = getattr(metadata, "stats", None)
            if isinstance(meta_stats, dict):
                bounds = extract_bounds(meta_stats.get("actions_orig") or meta_stats.get("actions"))
                if bounds is not None:
                    return bounds
        if root is None and metadata is not None:
            root = getattr(metadata, "root", None)
        if root is not None:
            stats_path = Path(root) / "meta" / "stats.json"
            if stats_path.exists():
                file_stats = json.loads(stats_path.read_text())
                bounds = extract_bounds(file_stats.get("actions_orig") or file_stats.get("actions"))
                if bounds is not None:
                    return bounds
        if hasattr(current, "dataset"):
            current = current.dataset
        elif hasattr(current, "base_dataset"):
            current = current.base_dataset
        else:
            current = None
    return None


class RoboMMEOnlinePolicy:
    def __init__(
        self,
        model: "BasePolicy",
        data_cfg: "DataConfig",
        input_transforms: Sequence["TransformFn"],
        output_transforms: Sequence["TransformFn"],
        *,
        inference_steps: int,
        chunk_size: int | None,
        use_bf16: bool,
        action_min: np.ndarray | None,
        action_max: np.ndarray | None,
    ) -> None:
        self._model = model
        self._data_cfg = data_cfg
        self._input_transforms = input_transforms
        self._output_transforms = output_transforms
        self._num_steps = int(inference_steps)
        self._chunk_size = chunk_size
        self._device = next(model.parameters()).device
        self._use_bf16 = use_bf16 and self._device.type == "cuda"
        self._action_min = action_min
        self._action_max = action_max
        self.reset()

    def reset(self) -> None:
        self._front_history: list[np.ndarray] = []
        self._wrist_history: list[np.ndarray] = []
        self._state_history: list[np.ndarray] = []

    def _uses_vision_token_memory(self) -> bool:
        cfg = getattr(self._model, "config", None)
        return bool(getattr(cfg, "use_vision_token_memory", False))

    def _append_inputs(self, inputs: Dict[str, Any]) -> None:
        if bool(inputs.get("is_first_step", False)):
            self.reset()
        front = list(inputs.get("front_rgb_list") or [])
        wrist = list(inputs.get("wrist_rgb_list") or [])
        joints = list(inputs.get("joint_state_list") or [])
        grippers = list(inputs.get("gripper_state_list") or [])
        if not (len(front) == len(wrist) == len(joints) == len(grippers)):
            raise ValueError(
                "front/wrist/joint/gripper histories must have equal length: "
                f"{len(front)}, {len(wrist)}, {len(joints)}, {len(grippers)}"
            )
        self._front_history.extend(np.asarray(img, dtype=np.uint8) for img in front)
        self._wrist_history.extend(np.asarray(img, dtype=np.uint8) for img in wrist)
        self._state_history.extend(
            pack_state(joint, gripper) for joint, gripper in zip(joints, grippers)
        )

    def _history_indices(self) -> list[int]:
        if self._uses_vision_token_memory():
            return list(range(len(self._front_history)))
        frame_history = max(1, int(getattr(self._data_cfg, "frame_history", 1) or 1))
        frame_stride = max(1, int(getattr(self._data_cfg, "frame_stride", 1) or 1))
        mode = getattr(self._data_cfg, "history_sampling", "recent")
        indices = list(range(len(self._front_history)))
        if mode == "even":
            return cast(list[int], select_even(indices, frame_history))
        return cast(list[int], select_recent(indices, frame_history, frame_stride))

    def _build_sample(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._append_inputs(inputs)
        if not self._front_history:
            raise ValueError("No RGB observations received.")

        prompt_source = getattr(self._data_cfg, "prompt_source", "prompt") or "prompt"
        task_goal = inputs.get("task_goal") or inputs.get("prompt") or ""
        if prompt_source in {"simple_subgoal", "grounded_subgoal"} and inputs.get(prompt_source):
            task = str(inputs[prompt_source])
        else:
            task = str(task_goal[0]) if isinstance(task_goal, (list, tuple)) and task_goal else str(task_goal)

        hist_indices = self._history_indices()
        front_frames = [image_chw(self._front_history[i]) for i in hist_indices]
        wrist_frames = [image_chw(self._wrist_history[i]) for i in hist_indices]
        if len(front_frames) == 1:
            front_tensor = front_frames[0]
            wrist_tensor = wrist_frames[0]
        else:
            front_tensor = torch.stack(front_frames, dim=0)
            wrist_tensor = torch.stack(wrist_frames, dim=0)

        state_history = max(1, int(getattr(self._data_cfg, "state_history", 1) or 1))
        latest_state = torch.as_tensor(self._state_history[-1], dtype=torch.float32)
        state_tensor = latest_state.reshape(1, -1).repeat(state_history, 1)

        return {
            FRONT_IMAGE_KEY: front_tensor,
            WRIST_IMAGE_KEY: wrist_tensor,
            PROCESSED_STATE_KEY: state_tensor,
            TASK_KEY: task,
            GENERATION_PROMPT_KEY: "",
            GENERATION_ANSWER_KEY: "",
            "episode_id": "online",
        }

    @torch.inference_mode()
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sample = self._build_sample(inputs)
        data_sample: Any = sample
        for transform in self._input_transforms:
            data_sample = transform.compute(data_sample)
        data_sample = data_sample.to(self._device).unsqueeze(0)

        autocast_ctx = (
            torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)
            if self._use_bf16
            else nullcontext()
        )
        with autocast_ctx:
            actions = self._model.sample_actions(
                data_sample.observation,
                num_steps=self._num_steps,
            )

        output: Dict[str, Any] = {PROCESSED_ACTION_KEY: actions.squeeze(0).cpu()}
        for transform in self._output_transforms:
            output = transform.compute(output)

        action_array = np.asarray(output[PROCESSED_ACTION_KEY], dtype=np.float32)
        if self._action_min is not None and self._action_max is not None:
            action_array = np.clip(action_array, self._action_min, self._action_max)
        if self._chunk_size is not None:
            action_array = action_array[: int(self._chunk_size)]
        return {"actions": action_array}
