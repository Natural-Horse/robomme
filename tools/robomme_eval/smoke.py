from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from vla_scratch.robomme_eval.adapter import as_hwc_frame_list
from vla_scratch.transforms.data_keys import (
    FRONT_IMAGE_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
    WRIST_IMAGE_KEY,
)


def build_smoke_inputs(dataset: Any, index: int) -> Dict[str, Any]:
    base_dataset = getattr(dataset, "base_dataset", dataset)
    raw = base_dataset[int(index)]
    front_frames = as_hwc_frame_list(raw[FRONT_IMAGE_KEY])
    wrist_frames = as_hwc_frame_list(raw[WRIST_IMAGE_KEY])
    state = np.asarray(torch.as_tensor(raw[PROCESSED_STATE_KEY])[-1], dtype=np.float32)
    return {
        "task_goal": [str(raw.get(TASK_KEY, ""))],
        "is_first_step": True,
        "front_rgb_list": front_frames,
        "wrist_rgb_list": wrist_frames,
        "joint_state_list": [state[:7].copy() for _ in front_frames],
        "gripper_state_list": [state[7:8].copy() for _ in front_frames],
    }

