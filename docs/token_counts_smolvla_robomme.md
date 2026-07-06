# SmolVLA / RoboMME / VLA-scratch image token counts

Created: 2026-06-10

## SmolVLA / SmolVLM2

- Official LeRobot SmolVLA uses `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` by default and truncates the VLM text stack to the first 16 layers.
- SmolVLM2 config:
  - `vision_config.image_size = 512`
  - `vision_config.patch_size = 16`
  - raw SigLIP patch grid per 512 image/tile: `32 x 32 = 1024`
  - `scale_factor = 4`
  - connector output per 512 image/tile: `1024 / 4^2 = 64` continuous image tokens
- The processor defaults to `do_image_splitting=true`, `size.longest_edge=2048`, `max_image_size.longest_edge=512`. For a normal single 512 robot image/tile, the VLM-side image token count is 64.
- The action expert consumes continuous VLM hidden/KV information through the interleaved VLM/expert layers, not discrete image token IDs.

Sources:
- `https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/raw/main/config.json`
- `https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/raw/main/preprocessor_config.json`
- `_third_party/lerobot-v0.4.2/src/lerobot/policies/smolvla/configuration_smolvla.py`
- `_third_party/lerobot-v0.4.2/src/lerobot/policies/smolvla/smolvlm_with_expert.py`

## RoboMME

- RoboMME uses SigLIP `So400m/14` with `pool_type="none"` for image token features.
- Its dataset visualization and token-drop bookkeeping use an `8 x 8 = 64` spatial grid, with `32 x 32` pixel cells.
- RoboMME configs set:
  - `budget = 512`
  - frame sampling: `token_per_image = 16`
  - token dropping: `token_per_image = 64`

Sources:
- `robomme_local_stage/robomme_policy_learning/src/mme_vla_suite/models/integration/history_pi0.py`
- `robomme_local_stage/robomme_policy_learning/src/mme_vla_suite/dataset_builder/build_robomme_dataset.py`
- `robomme_local_stage/robomme_policy_learning/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml`
- `robomme_local_stage/robomme_policy_learning/src/mme_vla_suite/models/config/robomme/perceptual-tokendrop-context.yaml`

## Our Current VLA-scratch Run

Current running command uses:

- `policy.transforms.0.image_size_longest_edge=256`
- SmolVLM2-256M, also `patch_size=16`, `scale_factor=4`
- raw SigLIP patch grid per image: `16 x 16 = 256`
- connector output per image: `256 / 4^2 = 16` continuous image tokens
- framesamp override: `policy.vision_memory_token_per_image=16`
- tokendrop override: `policy.vision_memory_token_per_image=64`

Important mismatch:

- At 256px, the SmolVLM connector only yields 16 tokens per image. This matches our framesamp setting, but it does not match SmolVLA/RoboMME's 64-token-per-image spatial granularity.
- Earlier code only handled the 4D image-feature layout for TokenDrop. The current `vision_memory.py` restores the common 3D layout `[B*T*V, P, D]` to `[B, T*V, P, D]` before FrameSamp/TokenDrop selection.
- The remaining limitation is spatial granularity: with 256px inputs, connector-space memory has 16 tokens per image. A true 64-token TokenDrop setting requires larger image features or pre-connector token selection.

## Practical Difference Summary

| System | Image/tile size | Vision patch grid | Connector / memory tokens per image | Notes |
| --- | ---: | ---: | ---: | --- |
| SmolVLA official SmolVLM2 | 512 | 32x32 raw SigLIP = 1024 | 64 after connector | action expert attends through VLM/expert layers |
| RoboMME tokendrop | 256 visualization grid | 8x8 memory grid | 64 memory tokens | 32px cells in dataset/token-drop bookkeeping |
| RoboMME framesamp | same feature source | selected/downsampled | 16 tokens/image | config-specific |
| Our current framesamp | 256 | 16x16 raw SigLIP = 256 | 16 after connector | effective current memory = 128 tokens total |
| Our configured tokendrop | 256 | 16x16 raw SigLIP = 256 | only 16 after connector | 64-token setting does not match actual connector output |
