"""Fixed-structure, non-privileged controls for T10 hidden-mass balance."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Sequence

from rebranch.baselines import BaselineResult
from rebranch.exec.evaluator import Evaluator
from rebranch.sim.backend import TaskBackend
from rebranch.sim.kinder_scene import (
    BEAM_HALF_THICKNESS,
    BLOCK_HALF_SIZE,
    BLOCK_NAMES,
    HINGE_FRICTIONLOSS_NM,
    HINGE_STIFFNESS_NM_RAD,
    PAIR_ARM,
    PAIR_GAP,
    ZONE_HALF_Y,
    ZONE_X,
)

_SITE_Z_OFFSET_M = 0.0003
_ZONE_CENTER_X = (ZONE_X[0] + ZONE_X[1]) / 2
_ZONE_HALF_X = (ZONE_X[1] - ZONE_X[0]) / 2
_DX_LIMIT = _ZONE_HALF_X - BLOCK_HALF_SIZE
_DY_LIMIT = ZONE_HALF_Y - BLOCK_HALF_SIZE
_X_MIN = ZONE_X[0] + BLOCK_HALF_SIZE
_X_MAX = ZONE_X[1] - BLOCK_HALF_SIZE
_PROBE_ARM_M = 0.04
_GRAVITY_M_S2 = 9.81


def parameterized_blind_source() -> str:
    """Return one blind fixed-layout program with only legal landing parameters."""
    assignments = (
        ("block_a", -1.0, _ZONE_CENTER_X - (PAIR_ARM + PAIR_GAP / 2), 0.0),
        ("block_b", 1.0, PAIR_ARM - _ZONE_CENTER_X, 0.0),
        ("block_c", -1.0, _ZONE_CENTER_X - (PAIR_ARM - PAIR_GAP / 2), 0.0),
    )
    params = []
    placements = []
    for name, side, dx, dy in assignments:
        key = name.upper()
        params.extend((
            f"{key}_SIDE = {side:.1f}  # [-1.0, 1.0]",
            f"{key}_DX = {dx:.4f}  # [{-_DX_LIMIT:.4f}, {_DX_LIMIT:.4f}]",
            f"{key}_DY = {dy:.4f}  # [{-_DY_LIMIT:.4f}, {_DY_LIMIT:.4f}]",
        ))
        placements.append(
            f"    zone = 'zone_left' if {key}_SIDE < 0.0 else 'zone_right'\n"
            f"    _place(api, {name!r}, zone, {key}_DX, {key}_DY, tcp_quat)"
        )
    return "\n".join(params) + "\n" + _policy_helpers() + "\n" + (
        "def run(api):\n"
        "    tcp_quat = api.get_ee_pose()['quat']\n"
        + "\n".join(placements)
        + "\n    _withdraw(api, tcp_quat)\n"
    )


def fixed_blind_source() -> str:
    """Return the zero-development member of the parameter-only policy family."""
    return parameterized_blind_source()


def mass_balance_source() -> str:
    """Return a Track A program that measures and exactly balances 1-vs-2 layouts."""
    return _mass_balance_program(
        "    masses = {name: api.get_mass(name) for name in names}\n",
        tilt_feedback=False,
    )


def tilt_feedback_source() -> str:
    """Estimate each mass with a beam-angle probe, then correct the last placement."""
    measurement = (
        "    rest = _withdraw(api, tcp_quat)\n"
        "    api.move_compliant(rest, quat=tcp_quat, duration=0.8)\n"
        "    masses = {}\n"
        "    for name in names:\n"
        "        start_pos = api.get_object_pose(name)['pos']\n"
        "        masses[name] = _probe_mass(api, name, tcp_quat, start_pos)\n"
    )
    return _mass_balance_program(measurement, tilt_feedback=True)


def _mass_balance_program(measurement: str, *, tilt_feedback: bool) -> str:
    """Share legal partition and moment calculations between mass-aware controls."""
    source = (
        "def run(api):\n"
        "    names = ('block_a', 'block_b', 'block_c')\n"
        "    tcp_quat = api.get_ee_pose()['quat']\n"
        + measurement
        + "    best_width = -1.0\n"
        + "    best = None\n"
        + "    for mask in range(1, 7):\n"
        + "        left = [name for i, name in enumerate(names) if mask & (1 << i)]\n"
        + "        right = [name for i, name in enumerate(names) if not mask & (1 << i)]\n"
        + "        mass_left = sum(masses[name] for name in left)\n"
        + "        mass_right = sum(masses[name] for name in right)\n"
        + "        if len(left) == 2:\n"
        + "            left_low = " + repr(_X_MIN) + " + " + repr(PAIR_GAP) + " * masses[left[0]] / mass_left\n"
        + "            left_high = " + repr(_X_MAX) + " - " + repr(PAIR_GAP) + " * masses[left[1]] / mass_left\n"
        + "        else:\n"
        + "            left_low, left_high = " + repr(_X_MIN) + ", " + repr(_X_MAX) + "\n"
        + "        if len(right) == 2:\n"
        + "            right_low = " + repr(_X_MIN) + " + " + repr(PAIR_GAP) + " * masses[right[0]] / mass_right\n"
        + "            right_high = " + repr(_X_MAX) + " - " + repr(PAIR_GAP) + " * masses[right[1]] / mass_right\n"
        + "        else:\n"
        + "            right_low, right_high = " + repr(_X_MIN) + ", " + repr(_X_MAX) + "\n"
        + "        lo = max(left_low * mass_left, right_low * mass_right)\n"
        + "        hi = min(left_high * mass_left, right_high * mass_right)\n"
        + "        if lo <= hi and hi - lo > best_width:\n"
        + "            target_moment = 0.5 * (lo + hi)\n"
        + "            best_width = hi - lo\n"
        + "            best = (left, right, target_moment / mass_left, target_moment / mass_right, mass_left, mass_right)\n"
        + "    if best is None:\n"
        + "        raise ValueError('measured load response has no legal 1-vs-2 layout')\n"
        + "    left, right, left_x, right_x, mass_left, mass_right = best\n"
        + "    arms = {}\n"
        + "    if len(left) == 2:\n"
        + "        arms[left[0]] = left_x + " + repr(PAIR_GAP) + " * masses[left[1]] / mass_left\n"
        + "        arms[left[1]] = left_x - " + repr(PAIR_GAP) + " * masses[left[0]] / mass_left\n"
        + "    else:\n"
        + "        arms[left[0]] = left_x\n"
        + "    if len(right) == 2:\n"
        + "        arms[right[0]] = right_x + " + repr(PAIR_GAP) + " * masses[right[1]] / mass_right\n"
        + "        arms[right[1]] = right_x - " + repr(PAIR_GAP) + " * masses[right[0]] / mass_right\n"
        + "    else:\n"
        + "        arms[right[0]] = right_x\n"
        + "    pair, singleton = (left, right) if len(left) == 2 else (right, left)\n"
        + "    order = [pair[0], singleton[0], pair[1]]\n"
        + "    positions = {name: (-arms[name] if name in left else arms[name]) for name in names}\n"
        + "    for name in order[:2]:\n"
        + "        x = positions[name]\n"
        + "        zone = 'zone_left' if x < 0.0 else 'zone_right'\n"
        + f"        dx = {_ZONE_CENTER_X:.8f} + x if x < 0.0 else x - {_ZONE_CENTER_X:.8f}\n"
        + "        _place(api, name, zone, dx, 0.0, tcp_quat)\n"
    )
    if tilt_feedback:
        source += (
            "    rest = _withdraw(api, tcp_quat)\n"
            "    api.move_compliant(rest, quat=tcp_quat, duration=0.8)\n"
            "    angle = _beam_angle(api)\n"
            "    name = order[2]\n"
            "    friction = " + repr(HINGE_FRICTIONLOSS_NM) + " if angle > 0 else -" + repr(HINGE_FRICTIONLOSS_NM) + " if angle < 0 else 0.0\n"
            "    beam = api.get_object_pose('beam')\n"
            "    height_moment = sum(masses[n] * _local_position(beam, api.get_object_pose(n)['pos'])[2] for n in order[:2])\n"
            "    current_moment = (" + repr(HINGE_STIFFNESS_NM_RAD) + " * angle + friction) / (" + repr(_GRAVITY_M_S2) + " * math.cos(angle)) - height_moment * math.tan(angle)\n"
            "    x = -current_moment / masses[name]\n"
            "    side = -1.0 if x < 0.0 else 1.0\n"
            "    x = side * min(max(abs(x), " + repr(_X_MIN) + "), " + repr(_X_MAX) + ")\n"
            "    positions[name] = x\n"
            "    zone = 'zone_left' if x < 0.0 else 'zone_right'\n"
            f"    dx = {_ZONE_CENTER_X:.8f} + x if x < 0.0 else x - {_ZONE_CENTER_X:.8f}\n"
            "    _place(api, name, zone, dx, 0.0, tcp_quat)\n"
            "    _withdraw(api, tcp_quat)\n"
        )
        return _policy_helpers() + "\n" + _tilt_helpers() + "\n" + _probe_helpers() + "\n" + source
    source += (
        "    name = order[2]\n"
        "    x = positions[name]\n"
        "    zone = 'zone_left' if x < 0.0 else 'zone_right'\n"
        f"    dx = {_ZONE_CENTER_X:.8f} + x if x < 0.0 else x - {_ZONE_CENTER_X:.8f}\n"
        "    _place(api, name, zone, dx, 0.0, tcp_quat)\n"
        "    _withdraw(api, tcp_quat)\n"
    )
    return _policy_helpers() + "\n" + source


def _policy_helpers() -> str:
    return f'''\
def _rotate(q, v):
    w, x, y, z = q
    a, b, c = v
    return [
        (1-2*(y*y+z*z))*a + 2*(x*y-z*w)*b + 2*(x*z+y*w)*c,
        2*(x*y+z*w)*a + (1-2*(x*x+z*z))*b + 2*(y*z-x*w)*c,
        2*(x*z-y*w)*a + 2*(y*z+x*w)*b + (1-2*(x*x+y*y))*c,
    ]

def _world(site, dx, dy, dz):
    d = _rotate(site['quat'], [dx, dy, dz])
    p = site['pos']
    return [p[0]+d[0], p[1]+d[1], p[2]+d[2]]

def _place_at(api, name, site_name, dx, dy, target_z, tcp_quat):
    obj = api.get_object_pose(name)['pos']
    api.open_gripper(0.068)
    api.move_to([obj[0], obj[1], obj[2]+0.08], quat=tcp_quat, speed=0.05)
    api.move_to(obj, quat=tcp_quat, speed=0.03)
    api.close_gripper(force=10.0)
    api.move_to([obj[0], obj[1], obj[2]+0.09], quat=tcp_quat, speed=0.04)
    held = api.get_object_pose(name)['pos']
    tcp = api.get_ee_pose()['pos']
    grasp_offset = [held[i]-tcp[i] for i in range(3)]
    site = api.get_object_pose(site_name)
    target = _world(site, dx, dy, target_z)
    tcp_target = [target[i]-grasp_offset[i] for i in range(3)]
    api.move_to([tcp_target[0], tcp_target[1], tcp_target[2]+0.07], quat=tcp_quat, speed=0.04)
    held = api.get_object_pose(name)['pos']
    tcp = api.get_ee_pose()['pos']
    grasp_offset = [held[i]-tcp[i] for i in range(3)]
    tcp_target = [target[i]-grasp_offset[i] for i in range(3)]
    api.move_to(tcp_target, quat=tcp_quat, speed=0.02)
    for _ in range(2):
        actual = api.get_object_pose(name)['pos']
        current_site = api.get_object_pose(site_name)
        desired = _world(current_site, dx, dy, target_z)
        tcp = api.get_ee_pose()['pos']
        tcp_target = [tcp[i]+desired[i]-actual[i] for i in range(3)]
        api.move_to(tcp_target, quat=tcp_quat, speed=0.02)
    api.open_gripper(0.068)
    api.move_to([tcp_target[0], tcp_target[1], tcp_target[2]+0.07], quat=tcp_quat, speed=0.04)

def _place(api, name, zone_name, dx, dy, tcp_quat):
    _place_at(api, name, zone_name, dx, dy, {BLOCK_HALF_SIZE - _SITE_Z_OFFSET_M:.7f}, tcp_quat)

def _place_world(api, name, target, tcp_quat):
    obj = api.get_object_pose(name)['pos']
    api.open_gripper(0.068)
    api.move_to([obj[0], obj[1], obj[2]+0.08], quat=tcp_quat, speed=0.05)
    api.move_to(obj, quat=tcp_quat, speed=0.03)
    api.close_gripper(force=10.0)
    api.move_to([obj[0], obj[1], obj[2]+0.09], quat=tcp_quat, speed=0.04)
    held = api.get_object_pose(name)['pos']
    tcp = api.get_ee_pose()['pos']
    grasp_offset = [held[i]-tcp[i] for i in range(3)]
    tcp_target = [target[i]-grasp_offset[i] for i in range(3)]
    api.move_to([tcp_target[0], tcp_target[1], tcp_target[2]+0.07], quat=tcp_quat, speed=0.04)
    api.move_to(tcp_target, quat=tcp_quat, speed=0.02)
    api.open_gripper(0.068)
    api.move_to([tcp_target[0], tcp_target[1], tcp_target[2]+0.07], quat=tcp_quat, speed=0.04)

def _withdraw(api, tcp_quat):
    beam = api.get_object_pose('beam')['pos']
    api.move_to([beam[0], beam[1], beam[2]+0.12], quat=tcp_quat, speed=0.04)
    rest = [beam[0], beam[1], 0.35]
    api.move_to(rest, quat=tcp_quat, speed=0.04)
    return rest
'''

def _probe_helpers() -> str:
    return f'''\\
def _probe_mass(api, name, tcp_quat, start_pos):
    _place_at(api, name, 'beam', {_PROBE_ARM_M:.6f}, 0.0, {BEAM_HALF_THICKNESS + BLOCK_HALF_SIZE:.6f}, tcp_quat)
    rest = _withdraw(api, tcp_quat)
    api.move_compliant(rest, quat=tcp_quat, duration=0.8)
    after = _beam_angle(api)
    obj = api.get_object_pose(name)['pos']
    beam = api.get_object_pose('beam')
    local = _local_position(beam, obj)
    friction = {HINGE_FRICTIONLOSS_NM:.6f} if after > 0 else -{HINGE_FRICTIONLOSS_NM:.6f} if after < 0 else 0.0
    mass = ({HINGE_STIFFNESS_NM_RAD:.6f} * after + friction) / ({_GRAVITY_M_S2:.4f} * (local[0]*math.cos(after) + local[2]*math.sin(after)))
    _place_world(api, name, start_pos, tcp_quat)
    rest = _withdraw(api, tcp_quat)
    api.move_compliant(rest, quat=tcp_quat, duration=0.8)
    return mass
'''


def _tilt_helpers() -> str:
    return f'''\
def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw]

def _beam_angle(api):
    pivot = api.get_object_pose('balance_support')['quat']
    beam = api.get_object_pose('beam')['quat']
    relative = _qmul([pivot[0], -pivot[1], -pivot[2], -pivot[3]], beam)
    w, x, y, z = relative
    return math.atan2(2*(x*z+w*y), 1-2*(x*x+y*y))

def _local_position(beam, world_pos):
    delta = [world_pos[i]-beam['pos'][i] for i in range(3)]
    q = beam['quat']
    return _rotate([q[0], -q[1], -q[2], -q[3]], delta)
'''


def run_t10_control_development(
    backend: TaskBackend,
    method: str,
    cases: Sequence[dict[str, Any] | tuple[dict[str, Any], int]],
    *,
    seed: int,
    workers: int,
    max_rollouts: int,
) -> BaselineResult:
    """Evaluate one fixed T10 control on the caller's common development cases."""
    sources = {
        "t10_fixed_blind": fixed_blind_source,
        "t10_mass_balance": mass_balance_source,
        "t10_tilt_feedback": tilt_feedback_source,
    }
    source = sources[method]()
    if method == "t10_mass_balance" and getattr(backend, "measurement_track") != "A":
        raise ValueError("t10_mass_balance requires Track A")
    extra_primitives = ("get_mass",) if getattr(backend, "measurement_track") == "A" else ()
    with Evaluator(
        backend,
        workers=workers,
        budget_limit=max_rollouts,
        extra_primitives=extra_primitives,
    ) as evaluator:
        traces = evaluator.run_many(source, cases, seed=seed * 1000)

    failures = Counter((trace.failure or "none", trace.error or "none") for trace in traces)
    return BaselineResult(
        method=method,
        policy_source=source,
        n_rollouts=evaluator.n_rollouts,
        dev_llm_calls=0,
        dev_prompt_tokens=0,
        dev_completion_tokens=0,
        round_logs=[{
            "round": 0,
            "kind": "fixed_structure_control",
            "development_rollouts": evaluator.n_rollouts,
            "success_count": sum(trace.success for trace in traces),
            "mean_score": sum(trace.score for trace in traces) / len(traces) if traces else 0.0,
            "failure_counts": {
                f"{failure} | {error}": count
                for (failure, error), count in sorted(failures.items())
            },
        }],
        program_snapshots=[{"round": 0, "source": source}],
        extra={
            "baseline_type": "custom_control",
            "dev_rollout_budget": max_rollouts,
            "actual_dev_rollouts": evaluator.n_rollouts,
            "dev_sim_time_s": sum(trace.sim_time for trace in traces),
            "dev_api_calls": sum(len(trace.calls) for trace in traces),
            "dev_mass_calls": sum(
                call.name == "get_mass" for trace in traces for call in trace.calls
            ),
            "dev_traces": [asdict(trace) for trace in traces],
            "selection_rule": "fixed_control_no_parameter_selection",
        },
    )
