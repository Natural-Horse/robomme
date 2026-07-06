# RoboMME VLA Memory

```text
robomme/
+-- vla_scratch/
|   +-- policies/
|   |   +-- pi/
|   |   +-- modules/
|   |       +-- action_expert/
|   |       +-- vlm_bridge/
|   |           +-- smolvlm/
|   |               +-- bridge.py
|   |               +-- vision_memory.py
|   |               +-- video_mem_encoder.py
|   |               +-- processor.py
|   |               +-- utils.py
|   +-- datasets/
+-- configs/
+-- scripts/
+-- tools/
+-- docs/
+-- weights/
```

核心代码在 `vla_scratch/policies/modules/vlm_bridge/smolvlm` 下。

| 方法 | Goal SR | Spatial SR | Object SR | 平均 SR |
|---|---:|---:|---:|---:|
| baseline | 46.0% | 36.0% | 56.0% | 46.0% |
| framesamp | 49.0% | 31.0% | 64.0% | 48.0% |
| tokendrop | 48.0% | 48.0% | 60.0% | 52.0% |
| mem5 | N/A | 8.0% | N/A | 8.0% |
