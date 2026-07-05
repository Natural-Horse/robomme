#!/usr/bin/env python3
from __future__ import annotations

import logging
from typing import cast

import art
import emoji
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from setproctitle import setproctitle

import vla_scratch.configs  # noqa: F401
from vla_scratch.helpers.data import build_input_transforms, build_output_transforms
from vla_scratch.robomme_eval.adapter import (
    RoboMMEOnlinePolicy,
    initialize_policy_dims,
    resolve_action_bounds,
)
from vla_scratch.robomme_eval.config import RoboMMEServeConfig, register_config
from vla_scratch.robomme_eval.server import make_server
from vla_scratch.robomme_eval.smoke import build_smoke_inputs
from vla_scratch.transforms.common import ToNumpy, ToTorch
from vla_scratch.utils.checkpoint import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    merge_policy_cfg_from_checkpoint,
)

logger = logging.getLogger(__name__)
register_config()


def _metadata(cfg: RoboMMEServeConfig) -> dict:
    return {
        "policy": cfg.policy.__class__.__name__,
        "data": cfg.data.__class__.__name__,
        "action_horizon": cfg.policy.action_horizon,
        "action_dim": cfg.policy.action_dim,
        "frame_history": getattr(cfg.data, "frame_history", None),
        "history_sampling": getattr(cfg.data, "history_sampling", None),
        "transport": "http",
    }


@hydra.main(config_name="serve_robomme", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    art.tprint("VLA-SCRATCH", font="big")
    setproctitle("vla-robomme-serve")

    if (checkpoint_path := cfg.get("checkpoint_path")) is not None:
        cfg.checkpoint_path = find_latest_checkpoint(checkpoint_path)
    if cfg.get("merge_policy_cfg", False):
        cfg = merge_policy_cfg_from_checkpoint(cfg, cfg.get("checkpoint_path"))
        OmegaConf.resolve(cfg)

    serve_cfg = cast(RoboMMEServeConfig, OmegaConf.to_object(cfg))
    for i, spec in enumerate(list(serve_cfg.data.input_transforms or [])):
        if isinstance(spec, dict) and "enable_aug" in spec:
            spec.update({"enable_aug": False})
            serve_cfg.data.input_transforms[i] = spec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = initialize_policy_dims(serve_cfg.data, serve_cfg.policy)
    with torch.device(device):
        model = serve_cfg.policy.instantiate()

    if (ckpt := serve_cfg.checkpoint_path) is not None:
        print(emoji.emojize(":package: Loading checkpoint..."))
        missing, unexpected = load_model_from_checkpoint(model, ckpt, device, strict=False)
        print(emoji.emojize(":package: Checkpoint loaded."))
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    model.eval()
    use_bf16 = serve_cfg.use_bf16 and device.type == "cuda"
    if use_bf16:
        model.bfloat16()

    input_transforms = [ToTorch()] + list(build_input_transforms(serve_cfg.data, serve_cfg.policy))
    output_transforms = list(build_output_transforms(serve_cfg.data, serve_cfg.policy)) + [ToNumpy()]
    action_bounds = resolve_action_bounds(dataset)

    policy = RoboMMEOnlinePolicy(
        model,
        serve_cfg.data,
        input_transforms,
        output_transforms,
        inference_steps=serve_cfg.inference_steps,
        chunk_size=serve_cfg.chunk_size,
        use_bf16=use_bf16,
        action_min=action_bounds[0] if action_bounds is not None else None,
        action_max=action_bounds[1] if action_bounds is not None else None,
    )

    if serve_cfg.smoke_once:
        smoke_inputs = build_smoke_inputs(dataset, serve_cfg.smoke_dataset_index)
        smoke_output = policy.infer(smoke_inputs)
        actions = np.asarray(smoke_output["actions"])
        print("[smoke] actions shape:", tuple(actions.shape), "dtype:", actions.dtype)
        return

    server = make_server(serve_cfg.host, int(serve_cfg.port), policy, _metadata(serve_cfg))
    print(
        emoji.emojize(
            f":rocket: RoboMME HTTP policy listening at http://{serve_cfg.host}:{serve_cfg.port}"
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server loop.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
