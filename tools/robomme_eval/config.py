from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from vla_scratch.datasets.config import DataConfig
from vla_scratch.policies.config import PolicyConfig


@dataclass
class RoboMMEServeConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"policy": "pi-smol"},
            {"data": "robomme-patternlock-ep5"},
        ]
    )

    host: str = "0.0.0.0"
    port: int = 8001
    inference_steps: int = 10
    chunk_size: Optional[int] = None
    use_bf16: bool = True
    smoke_once: bool = False
    smoke_dataset_index: int = 0

    data: DataConfig = MISSING
    policy: PolicyConfig = MISSING
    checkpoint_path: Optional[str] = None
    merge_policy_cfg: bool = False


def register_config() -> None:
    ConfigStore.instance().store(name="serve_robomme", node=RoboMMEServeConfig())

