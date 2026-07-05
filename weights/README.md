# Checkpoint Manifest

Do not commit model weights with plain git.  Store them with Git LFS, a GitHub
release, HuggingFace Hub, or a separate object store, then keep paths and hashes
here.

## Known Local State

The current machine does not contain the full corrected checkpoint set.  The
local `G:\robomme_transfer` directory only contains transfer logs; the old
`perceptual-framesamp-modul/79999.zip` mentioned in the worklog is not present
as a file at scan time.

## Expected Checkpoints

| name | suite | epoch | source run | notes |
|---|---|---:|---|---|
| baseline | goal/spatial/object | 50 | `per_suite_0603_full/baseline` | full100 result row in `docs/train_results.md` |
| framesamp | goal/spatial/object | 50 | `smolvla_memory_layerwise_0609/framesamp` | corrected layerwise memory config |
| tokendrop | goal | 30 | `smolvla_tokendrop_layerwise_0612/tokendrop/goal` | best full100 goal checkpoint |
| tokendrop | spatial | 30 | `smolvla_tokendrop_layerwise_0612/tokendrop/spatial` | best full100 spatial checkpoint |
| tokendrop | object | 50 | unresolved locally | object row is retained from the available result summary |
| mem5 | spatial | 50 | `mem5_per_suite_0606/mem5/spatial` | measured low in full100 rerun |

## Suggested Layout

```text
checkpoints/
  baseline/{goal,spatial,object}/checkpoint_50/
  framesamp/{goal,spatial,object}/checkpoint_50/
  tokendrop/goal/checkpoint_30/
  tokendrop/spatial/checkpoint_30/
  tokendrop/object/checkpoint_50/
  mem5/spatial/checkpoint_50/
```

If Git LFS is enabled:

```bash
git lfs install
git lfs track "*.pt" "*.safetensors" "*.bin"
git add .gitattributes checkpoints/
git commit -m "Add experiment checkpoints"
```

For normal GitHub use, prefer uploading checkpoint directories as release assets
and keep the repository itself source-only.
