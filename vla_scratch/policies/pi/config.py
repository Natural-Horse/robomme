from dataclasses import dataclass, field
import copy
import os
import glob
from pathlib import Path
from hydra.core.config_store import ConfigStore
from vla_scratch.policies.config import PolicyConfig
from vla_scratch.policies.modules.action_expert.cross_attention_dit import (
    DiTConfig,
)


def _has_local_qwen_weights(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_index = os.path.isfile(os.path.join(path, "model.safetensors.index.json"))
    has_shards = len(glob.glob(os.path.join(path, "*.safetensors"))) > 0
    return has_index or has_shards

_REPO_ROOT = Path(__file__).resolve().parents[3]
_QWEN_LOCAL_DIR = str(_REPO_ROOT / "checkpoints" / "Qwen3-VL-2B-Instruct")
_QWEN_MODEL_ID = (
    _QWEN_LOCAL_DIR if _has_local_qwen_weights(_QWEN_LOCAL_DIR) else "Qwen/Qwen3-VL-2B-Instruct"
)


@dataclass
class PiConfig(PolicyConfig):
    vlm_type: str
    model_id: str

    action_expert_cfg: DiTConfig = field(
        default_factory=lambda: DiTConfig(
            # hidden size
            hidden_size=1024,
            intermediate_size=4096,
            # attention size
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            # layers
            num_hidden_layers=12,
            cross_attention_every=2,
            only_attend_to_final_layer=True,
        )
    )

    # architecture
    num_obs_registers: int = 4
    expert_only_use_register: bool = True
    suffix_add_pos_emb: bool = True
    use_state: bool = False

    # noising
    num_noise_per_sample: int = 2
    time_dist_alpha: float = 1.0
    time_dist_beta: float = 1.5

    # training
    detach_encoder_output: bool = False
    ce_loss_weight: float = 0.1
    freeze_vlm: bool = False

    # misc
    obs_register_init_gain: float = 0.02
    suffix_pos_emb_init_gain: float = 0.02
    zero_pos_id_for_obs_register: bool = True
    causal_mask_obs_register: bool = True

    # MEM-style video encoder toggles (default off for backward compatibility)
    use_mem_video_encoder: bool = False
    mem_video_every_n_layers: int = 4
    mem_video_frame_history: int = 1
    mem_video_num_cameras: int = 2
    mem_video_temporal_min_period: float = 1.0
    mem_video_temporal_max_period: float = 10000.0
    mem_video_drop_past: bool = True
    use_vision_token_memory: bool = False
    vision_memory_max_tokens: int = 128
    vision_memory_selection: str = "even"
    vision_memory_integration: str = "context"
    vision_memory_candidate_tokens: int = 512
    vision_memory_token_drop_stride: int = 1
    vision_memory_token_per_image: int = 0
    vision_memory_add_pos_emb: bool = False
    vlm_layer_selection: str = "last"


pi_paligemma_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    model_id="google/paligemma-3b-mix-224",
    vlm_type="PaliGemmaForConditionalGeneration",
    state_history=1,
    action_horizon=10,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma-3b-mix-224",
            "max_length": 550,
            "target_size": (224, 224),
        }
    ],
)

pi_paligemma2_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    model_id="google/paligemma2-3b-mix-224",
    vlm_type="PaliGemmaForConditionalGeneration",
    state_history=1,
    action_horizon=10,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.paligemma.processor.PaligemmaProcessor",
            "processor_class": "PaliGemmaProcessor",
            "model_id": "google/paligemma2-3b-mix-224",
            "max_length": 550,
            "target_size": (224, 224),
        }
    ],
)

pi_qwen_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id=_QWEN_MODEL_ID,
    vlm_type="Qwen3VLForConditionalGeneration",
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.qwen.processor.QwenProcessor",
            "processor_class": "Qwen3VLProcessor",
            "model_id": _QWEN_MODEL_ID,
            "max_length": 180,
            # WARN: select this based on your image sizes and prompt lengths, try to make it minimum as possible because if impacts iteration time a lot!
            "padding": "max_length",
        }
    ],
)

pi_smolvlm_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

pi_smolvlm_mem_k4_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    use_mem_video_encoder=True,
    mem_video_every_n_layers=4,
    mem_video_frame_history=5,
    mem_video_num_cameras=2,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

pi_smolvlm_mem_k2_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    use_mem_video_encoder=True,
    mem_video_every_n_layers=4,
    mem_video_frame_history=2,
    mem_video_num_cameras=2,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

pi_smolvlm_mem_k8_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    use_mem_video_encoder=True,
    mem_video_every_n_layers=4,
    mem_video_frame_history=8,
    mem_video_num_cameras=2,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

pi_smolvlm_vismem_k5_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    use_vision_token_memory=True,
    vision_memory_max_tokens=128,
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

pi_smolvlm_tokendrop_k5_config = PiConfig(
    _target_="vla_scratch.policies.pi.policy.PiPolicy",
    state_history=1,
    action_horizon=10,
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    vlm_type="SmolVLMForConditionalGeneration",
    use_vision_token_memory=True,
    vision_memory_max_tokens=128,
    vision_memory_selection="tokendrop",
    transforms=[
        {
            "_target_": "vla_scratch.policies.modules.vlm_bridge.smolvlm.processor.SmolVLMProcessor",
            "processor_class": "SmolVLMProcessor",
            "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            "max_length": 180,
            "padding": "max_length",
            "image_size_longest_edge": 512,
            "max_image_size_longest_edge": 512,
        }
    ],
)

pi_smolvlm_vismem_modulator_k5_config = copy.deepcopy(pi_smolvlm_vismem_k5_config)
pi_smolvlm_vismem_modulator_k5_config.vision_memory_integration = "modulator"

pi_smolvlm_vismem_expert_k5_config = copy.deepcopy(pi_smolvlm_vismem_k5_config)
pi_smolvlm_vismem_expert_k5_config.vision_memory_integration = "expert"

pi_smolvlm_tokendrop_modulator_k5_config = copy.deepcopy(
    pi_smolvlm_tokendrop_k5_config
)
pi_smolvlm_tokendrop_modulator_k5_config.vision_memory_integration = "modulator"

pi_smolvlm_tokendrop_expert_k5_config = copy.deepcopy(
    pi_smolvlm_tokendrop_k5_config
)
pi_smolvlm_tokendrop_expert_k5_config.vision_memory_integration = "expert"

pi_smolvlm_vismem_smolvla_k5_config = copy.deepcopy(pi_smolvlm_vismem_k5_config)
pi_smolvlm_vismem_smolvla_k5_config.vision_memory_max_tokens = 512
pi_smolvlm_vismem_smolvla_k5_config.vision_memory_token_per_image = 16
pi_smolvlm_vismem_smolvla_k5_config.vision_memory_add_pos_emb = True
pi_smolvlm_vismem_smolvla_k5_config.vlm_layer_selection = "first"
pi_smolvlm_vismem_smolvla_k5_config.action_expert_cfg.only_attend_to_final_layer = False

pi_smolvlm_tokendrop_smolvla_k5_config = copy.deepcopy(pi_smolvlm_tokendrop_k5_config)
pi_smolvlm_tokendrop_smolvla_k5_config.vision_memory_max_tokens = 512
pi_smolvlm_tokendrop_smolvla_k5_config.vision_memory_candidate_tokens = 2048
pi_smolvlm_tokendrop_smolvla_k5_config.vision_memory_token_drop_stride = 8
pi_smolvlm_tokendrop_smolvla_k5_config.vision_memory_token_per_image = 64
pi_smolvlm_tokendrop_smolvla_k5_config.vision_memory_add_pos_emb = True
pi_smolvlm_tokendrop_smolvla_k5_config.vlm_layer_selection = "first"
pi_smolvlm_tokendrop_smolvla_k5_config.action_expert_cfg.only_attend_to_final_layer = False

cs = ConfigStore.instance()
cs.store(name="pi-paligemma", node=pi_paligemma_config, group="policy")
cs.store(name="pi-paligemma2", node=pi_paligemma2_config, group="policy")
cs.store(name="pi-qwen", node=pi_qwen_config, group="policy")
cs.store(name="pi-smol", node=pi_smolvlm_config, group="policy")
cs.store(name="pi-smol-mem-k2", node=pi_smolvlm_mem_k2_config, group="policy")
cs.store(name="pi-smol-mem-k4", node=pi_smolvlm_mem_k4_config, group="policy")
cs.store(name="pi-smol-mem-k8", node=pi_smolvlm_mem_k8_config, group="policy")
cs.store(name="pi-smol-vismem-k5", node=pi_smolvlm_vismem_k5_config, group="policy")
cs.store(name="pi-smol-tokendrop-k5", node=pi_smolvlm_tokendrop_k5_config, group="policy")
cs.store(
    name="pi-smol-vismem-modulator-k5",
    node=pi_smolvlm_vismem_modulator_k5_config,
    group="policy",
)
cs.store(
    name="pi-smol-vismem-expert-k5",
    node=pi_smolvlm_vismem_expert_k5_config,
    group="policy",
)
cs.store(
    name="pi-smol-tokendrop-modulator-k5",
    node=pi_smolvlm_tokendrop_modulator_k5_config,
    group="policy",
)
cs.store(
    name="pi-smol-tokendrop-expert-k5",
    node=pi_smolvlm_tokendrop_expert_k5_config,
    group="policy",
)
cs.store(
    name="pi-smol-vismem-smolvla-k5",
    node=pi_smolvlm_vismem_smolvla_k5_config,
    group="policy",
)
cs.store(
    name="pi-smol-tokendrop-smolvla-k5",
    node=pi_smolvlm_tokendrop_smolvla_k5_config,
    group="policy",
)
