# RoboMME VLA Memory

This repository contains the VLA-scratch branch used for the RoboMME/LIBERO
memory experiments.  It starts from `EGalahad/vla-scratch` and adds a compact
SmolVLM memory path for FrameSamp and TokenDrop experiments.

## What Changed

- SmolVLM bridge can build history vision-token memory from `mem-k4` inputs.
- FrameSamp keeps evenly sampled history frames after connector-space pooling.
- TokenDrop scores connector tokens by RGB patch change and keeps the highest
  budgeted tokens in temporal order.
- DiT action expert can attend to early VLM layers with
  `policy.vlm_layer_selection=first` and
  `policy.action_expert_cfg.only_attend_to_final_layer=false`.
- IPEC LIBERO v30 dataset configs include goal/spatial/object local datasets
  and grouped 4-history-frame sampling.
- JSON launch configs keep training/eval queues reproducible.

## Main Results

Full100/no-video LIBERO eval summary from the local worklog:

| method | goal | spatial | object | avg |
|---|---:|---:|---:|---:|
| baseline | 46.0 | 36.0 | 56.0 | 46.0 |
| framesamp | 49.0 | 31.0 | 64.0 | 48.0 |
| tokendrop | 48.0 | 48.0 | 60.0 | 52.0 |
| mem5 | n/a | 8.0 | n/a | 8.0 |

Notes:

- The current TokenDrop row excludes older runs with the wrong final-layer-only
  architecture.
- TokenDrop goal/spatial best checkpoints were `checkpoint_30`; spatial
  `checkpoint_50` later measured lower.
- See `docs/train_results.md` for the original result table.

## Repository Layout

```text
vla_scratch/                         # trainable package
  policies/pi/                       # Pi policy + config registry
  policies/modules/vlm_bridge/       # SmolVLM bridge and memory outputs
    smolvlm/vision_memory.py         # FrameSamp / TokenDrop token selection
    smolvlm/video_mem_encoder.py     # MEM video encoder path
  datasets/libero_global/            # IPEC LIBERO v30 loader/configs
configs/experiments/                 # reproducible queue configs
scripts/launch_vla_experiment_from_config.py
tools/robomme_eval/                  # RoboMME serving/eval helpers
docs/                                # worklog reports kept for provenance
weights/README.md                    # checkpoint manifest and storage policy
```

## Training Entry

Example on the server:

```bash
cd /root/autodl-tmp/res/VLA-scratch
/root/autodl-tmp/res/envs/vla/bin/python scripts/launch_vla_experiment_from_config.py \
  --config configs/experiments/smolvla_memory_layerwise.json
```

The corrected TokenDrop-only queue is:

```bash
/root/autodl-tmp/res/envs/vla/bin/python scripts/launch_vla_experiment_from_config.py \
  --config configs/experiments/smolvla_tokendrop_layerwise.json
```

## Push To Your Remote

Create an empty GitHub repo first, then run:

```bash
cd robomme-vla-memory
git init
git add .
git commit -m "Add RoboMME VLA memory experiments"
git branch -M main
git remote add origin git@github.com:<your-account>/<repo>.git
git push -u origin main
```

Large checkpoints should not be committed through plain git.  Use Git LFS or a
release artifact; see `weights/README.md`.
