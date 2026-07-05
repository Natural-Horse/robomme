#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/res/VLA-scratch}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

PY="${PY:-/root/autodl-tmp/res/envs/vla/bin/python}"
RUN_DATE="${RUN_DATE:-2026-06-09}"
RUN_ID="${RUN_ID:-smolvla_memory_layerwise_0609}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${RUN_DATE}}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/res/install_logs/${RUN_ID}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval/${RUN_DATE}/${RUN_ID}}"
EPOCHS="${EPOCHS:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
EVAL_RENDERED="${EVAL_RENDERED:-10}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
TB_PORT="${TB_PORT:-6007}"

mkdir -p "$LOG_DIR"

SUITES=(goal spatial object)
DATA_NAMES=(
  libero-ipec-goal-local-v30-mem-k4
  libero-ipec-spatial-local-v30-mem-k4
  libero-ipec-object-local-v30-mem-k4
)

start_tensorboard() {
  local out_root="$1"
  if [[ "$START_TENSORBOARD" != "1" ]]; then
    return
  fi
  pkill -f "[t]ensorboard.*--port[ =]${TB_PORT}" || true
  nohup "$PY" -m tensorboard.main \
    --logdir "$out_root" \
    --host 127.0.0.1 \
    --port "$TB_PORT" \
    > "${LOG_DIR}/tensorboard_${RUN_ID}_${TB_PORT}.log" 2>&1 < /dev/null &
  echo $! > "${LOG_DIR}/tensorboard_${RUN_ID}_${TB_PORT}.pid"
  echo "[tensorboard] logdir=${out_root} port=${TB_PORT} pid=$(cat "${LOG_DIR}/tensorboard_${RUN_ID}_${TB_PORT}.pid")"
}

method_policy() {
  case "$1" in
    framesamp) echo "pi-smol-vismem-smolvla-k5" ;;
    tokendrop) echo "pi-smol-tokendrop-smolvla-k5" ;;
    *) echo "unknown method $1" >&2; exit 2 ;;
  esac
}

memory_overrides() {
  case "$1" in
    framesamp)
      printf '%s\n' \
        policy.vision_memory_selection=even \
        policy.vision_memory_max_tokens=512 \
        policy.vision_memory_token_per_image=16 \
        policy.vision_memory_add_pos_emb=true
      ;;
    tokendrop)
      printf '%s\n' \
        policy.vision_memory_selection=tokendrop \
        policy.vision_memory_max_tokens=512 \
        policy.vision_memory_candidate_tokens=2048 \
        policy.vision_memory_token_drop_stride=8 \
        policy.vision_memory_token_per_image=64 \
        policy.vision_memory_add_pos_emb=true
      ;;
  esac
}

train_one() {
  local method="$1"
  local suite="$2"
  local data_name="$3"
  local port="$4"
  local policy_name
  policy_name="$(method_policy "$method")"
  local run_dir="${OUTPUT_ROOT}/${RUN_ID}/${method}/${suite}"
  local exp_name="${RUN_ID}_${method}_${suite}_b${BATCH_SIZE}_e${EPOCHS}"
  local log_path="${LOG_DIR}/train_${method}_${suite}.log"
  local final_checkpoint="${run_dir}/checkpoint_${EPOCHS}/model.pt"

  if [[ -f "$final_checkpoint" ]]; then
    echo "[$(date '+%F %T')] SKIP_TRAIN ${method}/${suite} existing=${final_checkpoint}"
    return
  fi

  echo "[$(date '+%F %T')] START ${method}/${suite} policy=${policy_name} data=${data_name}"
  mkdir -p "$run_dir"
  mapfile -t mem_args < <(memory_overrides "$method")
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$port" \
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/train_policy.py \
    policy="$policy_name" \
    data="$data_name" \
    data.video_backend=pyav \
    eval_data=none \
    checkpoint_path=null \
    load_optimizer=false \
    policy.use_state=true \
    policy.state_history=1 \
    policy.action_horizon=30 \
    policy.transforms.0.max_length=256 \
    policy.transforms.0.image_size_longest_edge=256 \
    policy.transforms.0.max_image_size_longest_edge=256 \
    policy.freeze_vlm=true \
    policy.num_obs_registers=0 \
    policy.expert_only_use_register=false \
    policy.vlm_layer_selection=first \
    policy.action_expert_cfg.only_attend_to_final_layer=false \
    policy.action_expert_cfg.hidden_size=384 \
    policy.action_expert_cfg.intermediate_size=1536 \
    policy.action_expert_cfg.num_attention_heads=12 \
    policy.action_expert_cfg.num_key_value_heads=12 \
    policy.action_expert_cfg.head_dim=32 \
    policy.action_expert_cfg.num_hidden_layers=8 \
    policy.use_mem_video_encoder=false \
    policy.mem_video_frame_history=1 \
    policy.use_vision_token_memory=true \
    "${mem_args[@]}" \
    batch_size="$BATCH_SIZE" \
    num_workers="$NUM_WORKERS" \
    prefetch_factor="$PREFETCH_FACTOR" \
    use_ddp=true \
    low_mem=false \
    epochs="$EPOCHS" \
    log_interval=20 \
    eval_interval=1000 \
    save_interval="$SAVE_INTERVAL" \
    wandb.mode=disabled \
    exp_name="$exp_name" \
    hydra.run.dir="$run_dir" \
    2>&1 | tee "$log_path"
  echo "[$(date '+%F %T')] DONE ${method}/${suite}"
}

eval_one() {
  local method="$1"
  local suite="$2"
  local data_name="$3"
  local policy_name
  policy_name="$(method_policy "$method")"
  local run_dir="${OUTPUT_ROOT}/${RUN_ID}/${method}/${suite}"
  local eval_dir="${EVAL_ROOT}/${method}_${suite}_e${EPOCHS}_1"
  local log_path="${LOG_DIR}/eval_${method}_${suite}.log"
  if [[ "$RUN_EVAL" != "1" ]]; then
    return
  fi
  if [[ -f "${eval_dir}/eval_info.json" ]]; then
    echo "[$(date '+%F %T')] SKIP_EVAL ${method}/${suite} existing=${eval_dir}/eval_info.json"
    return
  fi
  echo "[$(date '+%F %T')] EVAL ${method}/${suite} checkpoint=${run_dir}"
  "$PY" examples/libero_lerobot/eval_libero_lerobot.py \
    policy="$policy_name" \
    data="$data_name" \
    data.video_backend=pyav \
    checkpoint_path="$run_dir" \
    merge_policy_cfg=true \
    env_task="libero_${suite}" \
    eval_n_episodes="$EVAL_N_EPISODES" \
    eval_batch_size=1 \
    max_episodes_rendered="$EVAL_RENDERED" \
    output_dir="$eval_dir" \
    2>&1 | tee "$log_path"
  "$PY" examples/libero_lerobot/rename_eval_videos.py "$eval_dir" --suite "libero_${suite}" \
    2>&1 | tee -a "$log_path" || true
}

main() {
  echo "[launch] run=${RUN_ID}"
  echo "[launch] output=${OUTPUT_ROOT}/${RUN_ID}"
  echo "[launch] batch=${BATCH_SIZE} epochs=${EPOCHS} save_interval=${SAVE_INTERVAL}"
  start_tensorboard "${OUTPUT_ROOT}/${RUN_ID}"

  local methods="${METHODS:-framesamp tokendrop}"
  for method in $methods; do
    for i in "${!SUITES[@]}"; do
      train_one "$method" "${SUITES[$i]}" "${DATA_NAMES[$i]}" "$((29900 + i))"
      eval_one "$method" "${SUITES[$i]}" "${DATA_NAMES[$i]}"
    done
  done
}

main "$@"
