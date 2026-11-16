"""Check whether force-width probe readings reflect settled contact geometry."""

import json
from dataclasses import asdict
from pathlib import Path

from rebranch.eval.runner import load_task_backend
from rebranch.exec.sandbox import run_policy


def main():
    task = load_task_backend("t2_cup")
    records = []
    for mode in ("empty", "sand"):
        for seed in range(3):
            theta = task.param_space.nominal()
            theta["cup_type"] = mode
            episode = task.make_episode(theta, seed=seed)
            p = episode.get_object_pose("cup")["pos"]
            episode.open_gripper(0.07)
            episode.move_to([p[0], p[1], p[2] + 0.08], speed=0.06)
            episode.move_to(p, speed=0.02)
            for force in (1.0, 3.0):
                episode.close_gripper(force)
                for elapsed, steps in ((0.3, 0), (0.8, 250), (1.8, 500)):
                    episode.step_idle(steps)
                    records.append({
                        "mode": mode, "seed": seed, "force_n": force,
                        "seconds_after_close_start": elapsed,
                        "true_width_m": float(episode.data.qpos[6] - episode.data.qpos[7]),
                        "measured_width_m": episode.get_gripper_state()["width"],
                        "object_z_m": episode.get_object_pose("cup")["pos"][2],
                    })
    out = Path("outputs/grounding_audit/probe_settling.json")
    out.write_text(json.dumps(records, indent=2))
    for mode in ("empty", "sand"):
        for force in (1.0, 3.0):
            row = [r for r in records if r["mode"] == mode and r["force_n"] == force and r["seed"] == 0]
            print(mode, force, [(r["seconds_after_close_start"], round(r["true_width_m"], 5)) for r in row], flush=True)

    source = Path("outputs/grounding_audit/t2_cup_width_feedback.py").read_text()
    traces = []
    summary = []
    for mode in ("empty", "sand"):
        for seed in range(3):
            for extra_steps in (0, 250):
                theta = task.param_space.nominal()
                theta["cup_type"] = mode
                episode = task.make_episode(theta, seed=seed)
                original_close = episode.close_gripper

                def delayed_close(force=8.0):
                    original_close(force)
                    episode.step_idle(extra_steps)

                episode.close_gripper = delayed_close
                trace = run_policy(source, episode, theta, seed=seed)
                traces.append({"extra_wait_seconds": extra_steps * episode.dt, "trace": asdict(trace)})
                summary.append({"mode": mode, "seed": seed, "extra_wait_seconds": extra_steps * episode.dt, "success": trace.success, "failure": trace.failure})
    Path("outputs/grounding_audit/probe_timing_control.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    Path("outputs/grounding_audit/probe_timing_control_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
