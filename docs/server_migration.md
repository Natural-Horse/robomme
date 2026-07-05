# Server Migration Notes

This page keeps the migration steps without storing credentials.

## Minimum Workspace

```text
/root/autodl-tmp/res/VLA-scratch
/root/autodl-tmp/res/envs/vla
/root/autodl-tmp/res/install_logs
~/.cache/huggingface/lerobot_official_v30_http
~/.cache/huggingface/hub
```

## Bootstrap

```bash
git clone <remote-url> /root/autodl-tmp/res/VLA-scratch
cd /root/autodl-tmp/res/VLA-scratch
uv sync
```

If using the existing Python environment:

```bash
source /root/autodl-tmp/res/envs/vla/bin/activate
python -m pip install -e .
```

## Data

The experiment configs expect local IPEC LIBERO v30 datasets under:

```text
~/.cache/huggingface/lerobot_official_v30_http/IPEC-COMMUNITY/
  libero_goal_no_noops_1.0.0_lerobot
  libero_spatial_no_noops_1.0.0_lerobot
  libero_object_no_noops_1.0.0_lerobot
```

Use the HuggingFace mirror on servers where direct `huggingface.co` access is
unstable.

## Training Queue

```bash
python scripts/launch_vla_experiment_from_config.py \
  --config configs/experiments/smolvla_memory_layerwise.json
```

For the corrected TokenDrop queue:

```bash
python scripts/launch_vla_experiment_from_config.py \
  --config configs/experiments/smolvla_tokendrop_layerwise.json
```

TensorBoard is configured in each JSON file.  Forward from the local machine:

```bash
ssh -L 6007:127.0.0.1:6007 <server-alias>
```

Then open `http://127.0.0.1:6007`.

## Checkpoints

Keep checkpoints outside git unless Git LFS is explicitly enabled.  The expected
layout is documented in `weights/README.md`.
