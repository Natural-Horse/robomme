from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from vla_scratch.robomme_eval.methods import TASKS_3, get_method


@dataclass(frozen=True)
class EvalProtocol:
    split: str = "test"
    action_space: str = "joint_angle"
    max_steps: int = 1300
    obs_horizon: int = 16
    action_chunk_size: Optional[int] = None
    success_status: str = "success"
    default_tasks: tuple[str, ...] = TASKS_3
    episode_limit: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalRunSpec:
    method: str
    task: str
    policy: str
    data: str
    subgoal_type: Optional[str]
    checkpoint_path: Optional[str]
    split: str
    action_space: str
    max_steps: int
    obs_horizon: int
    action_chunk_size: Optional[int]
    episode_limit: Optional[int]

    def serve_overrides(self) -> list[str]:
        overrides = [
            f"policy={self.policy}",
            f"data={self.data}",
        ]
        if self.action_chunk_size is not None:
            overrides.append(f"chunk_size={self.action_chunk_size}")
        if self.checkpoint_path:
            overrides.append(f"checkpoint_path={self.checkpoint_path}")
        return overrides


def build_run_spec(
    *,
    method_name: str,
    task: str,
    checkpoint_path: Optional[str] = None,
    protocol: EvalProtocol | None = None,
) -> EvalRunSpec:
    protocol = protocol or EvalProtocol()
    method = get_method(method_name)
    return EvalRunSpec(
        method=method.name,
        task=task,
        policy=method.policy,
        data=method.data_for_task(task),
        subgoal_type=method.subgoal_type,
        checkpoint_path=checkpoint_path,
        split=protocol.split,
        action_space=protocol.action_space,
        max_steps=protocol.max_steps,
        obs_horizon=protocol.obs_horizon,
        action_chunk_size=protocol.action_chunk_size,
        episode_limit=protocol.episode_limit,
    )
