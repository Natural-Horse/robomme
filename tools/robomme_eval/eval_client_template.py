"""Template text used by the host-side eval driver.

The actual simulator client still runs inside the existing RoboMME Docker image
so that the scratch Python environment does not need simulator dependencies.
"""

MINI_CLIENT = r'''
import functools
import urllib.request

import msgpack
import numpy as np

from robomme.robomme_env import *  # noqa: F401,F403
from robomme.env_record_wrapper import BenchmarkEnvBuilder

PORT = {port}
TASK = {task!r}
MAX_STEPS = {max_steps}
EPISODE_LIMIT = {episode_limit}
SUBGOAL_TYPE = {subgoal_type!r}
BASE = f"http://127.0.0.1:{{PORT}}"


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {{obj.dtype}}")
    if isinstance(obj, np.ndarray):
        return {{b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}}
    if isinstance(obj, np.generic):
        return {{b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def post(path, payload, timeout=90):
    req = urllib.request.Request(
        BASE + path,
        data=packb(payload),
        headers={{"Content-Type": "application/msgpack", "Accept": "application/msgpack"}},
        method="POST",
    )
    return unpackb(urllib.request.urlopen(req, timeout=timeout).read())


def current_inputs(obs, info, *, is_first_step, subgoal=None):
    task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]
    if isinstance(subgoal, list):
        subgoal = subgoal[0] if subgoal else None
    inputs = {{
        "task_goal": [task_goal],
        "is_first_step": is_first_step,
        "front_rgb_list": [np.asarray(x, dtype=np.uint8) for x in obs["front_rgb_list"]],
        "wrist_rgb_list": [np.asarray(x, dtype=np.uint8) for x in obs["wrist_rgb_list"]],
        "joint_state_list": [np.asarray(x, dtype=np.float32) for x in obs["joint_state_list"]],
        "gripper_state_list": [np.asarray(x, dtype=np.float32) for x in obs["gripper_state_list"]],
    }}
    if subgoal is not None:
        inputs["simple_subgoal"] = subgoal
        inputs["grounded_subgoal"] = subgoal
    return inputs


print("health", urllib.request.urlopen(BASE + "/healthz", timeout=5).read().decode().strip())
print("metadata", unpackb(urllib.request.urlopen(BASE + "/metadata", timeout=5).read()))

builder = BenchmarkEnvBuilder(env_id=TASK, dataset="test", action_space="joint_angle", gui_render=False, max_steps=MAX_STEPS)
episode_num = builder.get_episode_num()
if EPISODE_LIMIT is not None:
    episode_num = min(episode_num, int(EPISODE_LIMIT))
print("episodes", episode_num)

results = {{}}
for episode_id in range(episode_num):
    post("/reset", {{"reset": True}}, timeout=10)
    env = builder.make_env_for_episode(episode_id)
    try:
        obs, info = env.reset()
        task_goal = info["task_goal"][0] if isinstance(info["task_goal"], list) else info["task_goal"]
        subgoal = None
        if SUBGOAL_TYPE == "simple_subgoal":
            subgoal = info.get("simple_subgoal_online")
        elif SUBGOAL_TYPE == "grounded_subgoal":
            subgoal = info.get("grounded_subgoal_online")
        outcome = "unknown"
        step_count = 0
        is_first_step = True
        while step_count < MAX_STEPS:
            resp = post("/infer", current_inputs(obs, info, is_first_step=is_first_step, subgoal=subgoal), timeout=120)
            actions = np.asarray(resp["actions"], dtype=np.float32)
            print(
                "episode",
                episode_id,
                "goal",
                task_goal,
                "step",
                step_count,
                "action_chunk",
                actions.shape,
                resp.get("server_timing"),
            )
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise RuntimeError(f"Invalid action chunk shape: {{actions.shape}}")
            is_first_step = False
            for action in actions:
                obs, _, terminated, truncated, info = env.step(action)
                step_count += 1
                outcome = info.get("status", "unknown")
                if terminated or truncated or step_count >= MAX_STEPS:
                    break
            if terminated or truncated:
                break
        results[str(episode_id)] = outcome
        print("episode_result", episode_id, outcome, "steps", step_count)
    finally:
        env.close()

success = sum(1 for v in results.values() if v == "success")
total = max(1, len(results))
print("summary", {{"success": success, "total": total, "success_rate": success / total, "results": results}})
'''
