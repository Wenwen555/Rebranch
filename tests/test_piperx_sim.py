"""Tests for PiPER-X MJCF asset, 6-DOF OSC / admittance control, MujocoEpisode, and real driver."""
from __future__ import annotations

import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rebranch.api.primitives import EpisodeTerminated, RobotAPI
from rebranch.api.proprioception import DEFAULT_PIPERX_XML, ProprioceptiveEstimator
from rebranch.exec.sandbox import run_policy
from rebranch.sim.backend import SNAPSHOT_KEYS, Episode, EpisodeResult

Q_HOME = np.array([0.0, 1.48, -1.38, 1.30, 0.0, 0.0], dtype=float)


def test_piperx_xml_requirements_and_compile_speed() -> None:
    xml_text = DEFAULT_PIPERX_XML.read_text(encoding="utf-8")
    assert "gravcomp" not in xml_text
    assert "actuatorgravcomp" not in xml_text
    assert "<force" not in xml_text
    assert "<torque" not in xml_text
    assert ".dae" not in xml_text.lower()

    root = ET.fromstring(xml_text)
    j6 = root.find(".//joint[@name='joint6']")
    assert j6 is not None
    lo, hi = (float(v) for v in j6.attrib["range"].split())
    assert lo == pytest.approx(-math.pi, abs=1e-6)
    assert hi == pytest.approx(math.pi, abs=1e-6)

    tcp = root.find(".//body[@name='gripper_base']/site[@name='tcp']")
    assert tcp is not None
    assert [float(v) for v in tcp.attrib["pos"].split()] == pytest.approx([0.0, 0.0, 0.148])

    pad_l = root.find(".//body[@name='gripper_link1']/geom[@name='pad_left']")
    pad_r = root.find(".//body[@name='gripper_link2']/geom[@name='pad_right']")
    assert pad_l is not None and pad_r is not None
    assert [float(v) for v in pad_l.attrib["pos"].split()] == pytest.approx([0.0, 0.010, 0.002])
    assert [float(v) for v in pad_l.attrib["size"].split()] == pytest.approx([0.009, 0.014, 0.002])
    assert [float(v) for v in pad_r.attrib["pos"].split()] == pytest.approx([0.0, 0.010, 0.002])
    assert [float(v) for v in pad_r.attrib["size"].split()] == pytest.approx([0.009, 0.014, 0.002])

    excl = root.find(".//contact/exclude[@body1='gripper_link1'][@body2='gripper_link2']")
    assert excl is not None

    # MjSpec.compile() speed < 15 ms (check warm median after filesystem cache)
    times_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        spec = mujoco.MjSpec.from_file(str(DEFAULT_PIPERX_XML))
        model = spec.compile()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    assert min(times_ms) < 15.0
    assert sorted(times_ms)[2] < 15.0

    # Inner pad gap equals qpos[6] - qpos[7] across [0.0, 0.07]
    data = mujoco.MjData(model)
    for width in (0.0, 0.025, 0.05, 0.07):
        data.qpos[:6] = Q_HOME
        data.qpos[6] = 0.5 * width
        data.qpos[7] = -0.5 * width
        mujoco.mj_forward(model, data)
        gap = float(data.geom("pad_right").xpos[1] - data.geom("pad_left").xpos[1] - 0.004)
        assert gap == pytest.approx(width, abs=1e-5)

    # Workspace joint margin >= 9 deg across x in [0.22, 0.48], y in [-0.22, 0.22] at z=0.06..0.12
    tcp_id = model.site("tcp").id
    data.qpos[:6] = Q_HOME
    mujoco.mj_forward(model, data)
    R_home = data.site_xmat[tcp_id].reshape(3, 3).copy()
    j_lo, j_hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    min_margin_deg = 999.0
    for x in np.linspace(0.22, 0.48, 5):
        for y in np.linspace(-0.22, 0.22, 5):
            yaw = math.atan2(y, x)
            Rz = np.array(
                [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
            )
            R_des = Rz @ R_home
            q = Q_HOME.copy()
            for _ in range(100):
                data.qpos[:6] = q
                mujoco.mj_forward(model, data)
                mujoco.mj_jacSite(model, data, jp, jr, tcp_id)
                J = np.vstack([jp[:, :6], jr[:, :6]])
                R_err = R_des @ data.site_xmat[tcp_id].reshape(3, 3).T
                qt = np.zeros(4)
                mujoco.mju_mat2Quat(qt, np.ascontiguousarray(R_err).ravel())
                if qt[0] < 0:
                    qt *= -1.0
                sh = float(np.linalg.norm(qt[1:]))
                rot_err = (2.0 * math.atan2(sh, float(qt[0])) / sh) * qt[1:] if sh > 1e-10 else 2.0 * qt[1:]
                err = np.concatenate([np.array([x, y, 0.08]) - data.site_xpos[tcp_id], rot_err])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
                q = np.clip(q + 0.5 * dq, j_lo, j_hi)
            min_margin_deg = min(min_margin_deg, float(np.rad2deg(np.min(np.minimum(q - j_lo, j_hi - q)))))
    assert min_margin_deg >= 9.0


def test_gripper_clamping_force_calibration() -> None:
    from rebranch.sim.control import gripper_control_torques, osc_torque_6dof

    est = ProprioceptiveEstimator()
    pos_home, _ = est.fk(Q_HOME)

    spec = mujoco.MjSpec.from_file(str(DEFAULT_PIPERX_XML))
    spec.worldbody.add_geom(
        name="pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.06, 0.06, 0.02],
        pos=[pos_home[0], pos_home[1], pos_home[2] - 0.035],
    )
    box = spec.worldbody.add_body(name="test_box", pos=list(pos_home))
    box.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    box.mass = 0.20
    box.ipos = [0.0, 0.0, 0.0]
    box.inertia = [1e-4, 1e-4, 1e-4]
    box.add_geom(
        name="test_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.02, 0.015],
        friction=[1.2, 0.005, 0.0001],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    tcp_id = model.site("tcp").id
    pad_l = model.geom("pad_left").id
    pad_r = model.geom("pad_right").id
    box_g = model.geom("test_box_geom").id

    for f_cmd in (1.0, 4.0, 8.0, 14.0, 20.0):
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = Q_HOME
        data.qpos[6:8] = [0.025, -0.025]
        mujoco.mj_forward(model, data)
        x0 = data.site_xpos[tcp_id].copy()
        R0 = data.site_xmat[tcp_id].reshape(3, 3).copy()
        for _ in range(250):
            data.ctrl[:6] = osc_torque_6dof(model, data, tcp_id, x0, R0)
            data.ctrl[6:8] = gripper_control_torques(model, data, "force", target_force=f_cmd)
            mujoco.mj_step(model, data)
        fn_l = fn_r = 0.0
        cf = np.zeros(6)
        for ci in range(data.ncon):
            c = data.contact[ci]
            if {c.geom1, c.geom2} == {pad_l, box_g}:
                mujoco.mj_contactForce(model, data, ci, cf)
                fn_l += float(cf[0])
            elif {c.geom1, c.geom2} == {pad_r, box_g}:
                mujoco.mj_contactForce(model, data, ci, cf)
                fn_r += float(cf[0])
        assert fn_l == pytest.approx(f_cmd, rel=0.10)
        assert fn_r == pytest.approx(f_cmd, rel=0.10)


@pytest.mark.parametrize("control_mode", ["mit_osc", "pos_admittance"])
def test_unloaded_move_to_accuracy(control_mode: str) -> None:
    from rebranch.sim.mujoco_episode import MujocoEpisode

    ep = MujocoEpisode(control_mode=control_mode, seed=0)  # type: ignore[arg-type]
    assert isinstance(ep, Episode)
    assert isinstance(ep, RobotAPI)

    for target in ([0.38, 0.10, 0.08], [0.26, -0.12, 0.07]):
        ep.move_to(target, speed=0.08)
        pos = ep.get_ee_pose()["pos"]
        err = float(np.linalg.norm(np.array(pos) - np.array(target)))
        assert err < 0.002


def test_300g_payload_lift_wrench() -> None:
    from rebranch.sim.mujoco_episode import MujocoEpisode

    def add_300g_box(spec: mujoco.MjSpec) -> None:
        spec.worldbody.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.36, 0.0, 0.0],
            size=[0.18, 0.25, 0.02],
            friction=[1.0, 0.005, 0.0001],
        )
        box = spec.worldbody.add_body(name="payload_300g", pos=[0.35, 0.0, 0.045])
        box.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        box.mass = 0.300
        box.ipos = [0.0, 0.0, 0.0]
        box.inertia = [1e-4, 1e-4, 1e-4]
        box.add_geom(
            name="payload_300g_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.025],
            friction=[1.2, 0.005, 0.0001],
        )

    ep = MujocoEpisode(spec_mutator=add_300g_box, noise_enabled=True, seed=7)
    ep.zero_ee_wrench()
    ep.open_gripper(0.065)
    ep.move_to([0.35, 0.0, 0.10], speed=0.08)
    ep.move_to([0.35, 0.0, 0.045], speed=0.06)
    ep.close_gripper(14.0)
    gs = ep.get_gripper_state()
    assert gs["grasped"] is True
    ep.move_to([0.35, 0.0, 0.14], speed=0.05)
    ep.step_idle(60)

    w = ep.get_ee_wrench("world")
    assert w["force"][2] == pytest.approx(-2.943, rel=0.20)


@pytest.mark.parametrize("control_mode", ["mit_osc", "pos_admittance"])
def test_move_compliant_hinged_door_arc(control_mode: str) -> None:
    from rebranch.sim.control import admittance_step, osc_torque_6dof
    from rebranch.sim.mujoco_episode import MujocoEpisode

    def add_door(spec: mujoco.MjSpec) -> None:
        spec.worldbody.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.36, 0.0, -0.02],
            size=[0.20, 0.25, 0.02],
        )
        door = spec.worldbody.add_body(name="door", pos=[0.41, 0.18, 0.10])
        door.add_joint(
            name="door_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-1.57, 1.57],
            damping=0.4,
            frictionloss=0.2,
        )
        door.mass = 0.5
        door.ipos = [0.0, -0.09, 0.0]
        door.inertia = [1e-3, 1e-3, 2e-3]
        door.add_geom(
            name="door_panel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, -0.09, 0.0],
            size=[0.01, 0.07, 0.04],
            contype=0,
            conaffinity=0,
        )
        door.add_geom(
            name="door_handle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, -0.18, 0.0],
            size=[0.012, 0.008, 0.02],
            friction=[1.5, 0.005, 0.0001],
        )

    ep = MujocoEpisode(spec_mutator=add_door, control_mode=control_mode, seed=1)  # type: ignore[arg-type]
    handle = ep.get_object_pose("door_handle")["pos"]
    ep.open_gripper(0.06)
    ep.move_to([handle[0] - 0.04, handle[1], handle[2]], speed=0.08)
    ep.move_to(handle, speed=0.06)
    ep.close_gripper(15.0)
    ep.move_compliant([handle[0] - 0.08, handle[1], handle[2]], free_axes="Z", speed=0.05)
    angle_deg = abs(math.degrees(float(ep.data.joint("door_hinge").qpos[0])))
    assert angle_deg >= 15.0

    # Unit check: osc_torque_6dof stiffness zeroing along free_axes & admittance_step sign
    R0 = np.eye(3)
    tau_free = osc_torque_6dof(
        ep.model,
        ep.data,
        ep.tcp_id,
        ep.data.site_xpos[ep.tcp_id] + np.array([0.0, 0.0, 0.2]),
        R0,
        free_axes="zXYZ",
    )
    tau_free2 = osc_torque_6dof(
        ep.model,
        ep.data,
        ep.tcp_id,
        ep.data.site_xpos[ep.tcp_id] + np.array([0.0, 0.0, -0.2]),
        R0,
        free_axes="zXYZ",
    )
    assert np.allclose(tau_free, tau_free2, atol=1e-8)

    x_next, _ = admittance_step(
        np.array([0.3, 0.0, 0.1]), R0, np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), "z", 200.0, 0.002
    )
    assert x_next[2] > 0.1
    assert x_next[0] == pytest.approx(0.3)


def test_rollout_speed_determinism_snapshot_and_callbacks() -> None:
    from rebranch.sim.mujoco_episode import MujocoEpisode

    # 1. Single 20s rollout (10,000 steps) < 1.0 s
    ep = MujocoEpisode(seed=0)
    t0 = time.perf_counter()
    ep.step_idle(10000)
    wall = time.perf_counter() - t0
    assert ep.sim_time == pytest.approx(20.0, abs=1e-6)
    assert wall < 1.0

    # 2. Determinism & strict SNAPSHOT_KEYS via run_policy
    policy = """
def run(api):
    api.zero_ee_wrench()
    p = api.get_ee_pose()["pos"]
    api.open_gripper(0.05)
    api.move_to([p[0] + 0.03, p[1] + 0.02, p[2] - 0.04], speed=0.08)
    w = api.get_ee_wrench("world")
    gs = api.get_gripper_state()
"""
    t1 = run_policy(policy, MujocoEpisode(seed=42), theta={"x": 1}, seed=42)
    t2 = run_policy(policy, MujocoEpisode(seed=42), theta={"x": 1}, seed=42)
    assert t1.success and t2.success
    assert t1.lines == t2.lines
    assert t1.path == t2.path
    assert [c.obs for c in t1.calls] == [c.obs for c in t2.calls]
    for call in t1.calls:
        assert tuple(call.obs.keys()) == SNAPSHOT_KEYS

    # 3. move_until_contact detects tabletop contact
    def add_table(spec: mujoco.MjSpec) -> None:
        spec.worldbody.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.35, 0.0, 0.0],
            size=[0.20, 0.25, 0.02],
        )

    ep_c = MujocoEpisode(spec_mutator=add_table, seed=3)
    ep_c.open_gripper(0.06)
    res_c = ep_c.move_until_contact([0.35, 0.0, -0.02], speed=0.04, max_force=5.0, max_torque=0.8)
    assert res_c["contact"] is True
    assert res_c["pos"][2] > 0.015

    # 4. step_callback termination
    def crush_cb(model: mujoco.MjModel, data: mujoco.MjData) -> str | None:
        if data.time > 0.05:
            return "crushed"
        return None

    ep_cb = MujocoEpisode(step_callback=crush_cb, seed=0)
    t_cb = run_policy("def run(api):\n    api.open_gripper(0.05)\n", ep_cb, theta={}, seed=0)
    assert not t_cb.success
    assert t_cb.failure == "crushed"


def test_real_driver_and_calibrate_script_sim_check() -> None:
    from rebranch.real import PiperXDriver, PiperXEpisode, PiperXHardwareUnavailable

    drv = PiperXDriver(dry_run=False)
    with pytest.raises(PiperXHardwareUnavailable):
        drv.read_joints()

    ep_real = PiperXEpisode(
        dry_run=True,
        object_pose_fn=lambda name: {"pos": [0.35, 0.0, 0.05], "quat": [1.0, 0.0, 0.0, 0.0]},
    )
    ep_real.zero_ee_wrench()
    ep_real.open_gripper(0.06)
    ep_real.move_to([0.35, 0.02, 0.12], speed=0.08)
    snap = ep_real.snapshot()
    assert tuple(snap.keys()) == SNAPSHOT_KEYS
    assert ep_real.evaluate() == EpisodeResult(success=True, score=1.0)

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "calibrate_piperx.py"), "--sim-check"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"payload_300g_fz_n"' in proc.stdout
    assert '"gripper_forces_n"' in proc.stdout
