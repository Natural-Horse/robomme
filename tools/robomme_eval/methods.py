from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


TASKS_3 = ("PatternLock", "StopCube", "MoveCube")
FULL_ROBOMME_TASKS = (
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)

TASK_DATA_CONFIGS = {
    "prompt": {
        "PatternLock": "robomme-patternlock-ep5",
        "StopCube": "robomme-stopcube-ep3",
        "MoveCube": "robomme-movecube-ep3",
    },
    "prompt-even-k5": {
        "PatternLock": "robomme-patternlock-ep5-even-k5",
        "StopCube": "robomme-stopcube-ep3-even-k5",
        "MoveCube": "robomme-movecube-ep3-even-k5",
    },
    "simple-subgoal": {
        "PatternLock": "robomme-patternlock-ep5-simple-subgoal",
        "StopCube": "robomme-stopcube-ep3-simple-subgoal",
        "MoveCube": "robomme-movecube-ep3-simple-subgoal",
    },
    "grounded-subgoal": {
        "PatternLock": "robomme-patternlock-ep5-grounded-subgoal",
        "StopCube": "robomme-stopcube-ep3-grounded-subgoal",
        "MoveCube": "robomme-movecube-ep3-grounded-subgoal",
    },
    "grounded-even-k5": {
        "PatternLock": "robomme-patternlock-ep5-grounded-even-k5",
        "StopCube": "robomme-stopcube-ep3-grounded-even-k5",
        "MoveCube": "robomme-movecube-ep3-grounded-even-k5",
    },
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    policy: str
    data_family: str
    subgoal_type: Optional[str] = None
    memory_kind: str = "none"
    description: str = ""

    def data_for_task(self, task: str) -> str:
        try:
            return TASK_DATA_CONFIGS[self.data_family][task]
        except KeyError as exc:
            raise KeyError(
                f"Method {self.name!r} has no data config for task {task!r}. "
                f"Available tasks for family {self.data_family!r}: "
                f"{sorted(TASK_DATA_CONFIGS.get(self.data_family, {}))}"
            ) from exc


METHODS: Dict[str, MethodSpec] = {
    "baseline_prompt": MethodSpec(
        name="baseline_prompt",
        policy="pi-smol",
        data_family="prompt",
        memory_kind="none",
        description="No explicit memory; original task prompt only.",
    ),
    "symbolic_simple": MethodSpec(
        name="symbolic_simple",
        policy="pi-smol",
        data_family="simple-subgoal",
        subgoal_type="simple_subgoal",
        memory_kind="symbolic",
        description="Simple subgoal text as policy condition.",
    ),
    "symbolic_grounded": MethodSpec(
        name="symbolic_grounded",
        policy="pi-smol",
        data_family="grounded-subgoal",
        subgoal_type="grounded_subgoal",
        memory_kind="symbolic",
        description="Grounded subgoal text as policy condition.",
    ),
    "perceptual_framesamp": MethodSpec(
        name="perceptual_framesamp",
        policy="pi-smol-vismem-k5",
        data_family="prompt-even-k5",
        memory_kind="perceptual",
        description="Pure perceptual FrameSamp history, no subgoal text.",
    ),
    "perceptual_tokendrop": MethodSpec(
        name="perceptual_tokendrop",
        policy="pi-smol-tokendrop-k5",
        data_family="prompt-even-k5",
        memory_kind="perceptual",
        description="Pure perceptual TokenDrop history, no subgoal text.",
    ),
    "fusion_grounded_framesamp": MethodSpec(
        name="fusion_grounded_framesamp",
        policy="pi-smol-vismem-k5",
        data_family="grounded-even-k5",
        subgoal_type="grounded_subgoal",
        memory_kind="fusion",
        description="Grounded subgoal plus FrameSamp perceptual memory.",
    ),
    "fusion_grounded_tokendrop": MethodSpec(
        name="fusion_grounded_tokendrop",
        policy="pi-smol-tokendrop-k5",
        data_family="grounded-even-k5",
        subgoal_type="grounded_subgoal",
        memory_kind="fusion",
        description="Grounded subgoal plus TokenDrop perceptual memory.",
    ),
    "framesamp_context": MethodSpec(
        name="framesamp_context",
        policy="pi-smol-vismem-k5",
        data_family="prompt-even-k5",
        memory_kind="integration",
        description="FrameSamp memory injected as VLM prefix/context.",
    ),
    "framesamp_expert": MethodSpec(
        name="framesamp_expert",
        policy="pi-smol-vismem-expert-k5",
        data_family="prompt-even-k5",
        memory_kind="integration",
        description="FrameSamp memory injected into action expert cross-attention.",
    ),
    "framesamp_modulator": MethodSpec(
        name="framesamp_modulator",
        policy="pi-smol-vismem-modulator-k5",
        data_family="prompt-even-k5",
        memory_kind="integration",
        description="FrameSamp pooled memory injected as timestep modulator.",
    ),
    "tokendrop_context": MethodSpec(
        name="tokendrop_context",
        policy="pi-smol-tokendrop-k5",
        data_family="prompt-even-k5",
        memory_kind="integration",
        description="TokenDrop memory injected as VLM prefix/context.",
    ),
    "tokendrop_expert": MethodSpec(
        name="tokendrop_expert",
        policy="pi-smol-tokendrop-expert-k5",
        data_family="prompt-even-k5",
        memory_kind="integration",
        description="TokenDrop memory injected into action expert cross-attention.",
    ),
    "tokendrop_modulator": MethodSpec(
        name="tokendrop_modulator",
        policy="pi-smol-tokendrop-modulator-k5",
        data_family="prompt-even-k5",
        memory_kind="integration",
        description="TokenDrop pooled memory injected as timestep modulator.",
    ),
}


PAPER_PRIMARY_METHODS = (
    "baseline_prompt",
    "symbolic_simple",
    "symbolic_grounded",
    "perceptual_framesamp",
    "perceptual_tokendrop",
)

EXPERIMENTAL_FUSION_METHODS = (
    "fusion_grounded_framesamp",
    "fusion_grounded_tokendrop",
)


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown method {name!r}. Available: {sorted(METHODS)}") from exc


def iter_methods(names: Iterable[str] | None = None) -> Iterable[MethodSpec]:
    if names is None:
        return METHODS.values()
    return (get_method(name) for name in names)
