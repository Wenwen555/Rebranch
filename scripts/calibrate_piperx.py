#!/usr/bin/env python3
"""PiPER-X proprioception & gripper calibration script (with --sim-check dry-run mode)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

from rebranch.api.noise import DEFAULT_CALIB_PATH, JointNoiseModel
from rebranch.api.proprioception import DEFAULT_PIPERX_XML, ProprioceptiveEstimator
from rebranch.real.piperx_driver import PiperXDriver
from rebranch.real.piperx_episode import PiperXEpisode
from rebranch.sim.control import Q_HOME, gripper_control_torques, osc_torque_6dof
from rebranch.sim.mujoco_episode import MujocoEpisode


def run_sim_check(calib_path: Path, model_path: Path) -> dict[str, object]:
    """Validate PiPER-X kinematics, 300g payload wrench, noise model, and gripper force."""
    est = ProprioceptiveEstimator(model_path=model_path)
    pos_home, _ = est.fk(Q_HOME)

    # 1. Workspace joint limit margin check
    m = est.model
    d = mujoco.MjData(m)
    d.qpos[:6] = Q_HOME
    mujoco.mj_forward(m, d)
    R_home = d.site_xmat[est.site_id].reshape(3, 3).copy()
    lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
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
                d.qpos[:6] = q
                mujoco.mj_forward(m, d)
                mujoco.mj_jacSite(m, d, jp, jr, est.site_id)
                J = np.vstack([jp[:, :6], jr[:, :6]])
                err = np.concatenate([np.array([x, y, 0.08]) - d.site_xpos[est.site_id], np.zeros(3)])
                # full 6D orientation error
                R_err = R_des @ d.site_xmat[est.site_id].reshape(3, 3).T
                qt = np.zeros(4)
                mujoco.mju_mat2Quat(qt, np.ascontiguousarray(R_err).ravel())
                if qt[0] < 0:
                    qt *= -1.0
                sh = float(np.linalg.norm(qt[1:]))
                err[3:] = (2.0 * math.atan2(sh, float(qt[0])) / sh) * qt[1:] if sh > 1e-10 else 2.0 * qt[1:]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
                q = np.clip(q + 0.5 * dq, lo, hi)
            margin = float(np.rad2deg(np.min(np.minimum(q - lo, hi - q))))
            min_margin_deg = min(min_margin_deg, margin)

    # 2. Unloaded move_to accuracy + 300g payload lift wrench check
    def add_300g_box(spec: mujoco.MjSpec) -> None:
        # Tabletop at z = 0.02 in front of robot base
        spec.worldbody.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.36, 0.0, 0.0],
            size=[0.20, 0.25, 0.02],
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

    ep_unloaded = MujocoEpisode(model_path=model_path, calib_path=calib_path, seed=0)
    target = [0.38, 0.10, 0.08]
    ep_unloaded.move_to(target, speed=0.08)
    ee_after = ep_unloaded.get_ee_pose()["pos"]
    unloaded_err_m = float(np.linalg.norm(np.array(ee_after) - np.array(target)))

    ep_300g = MujocoEpisode(
        model_path=model_path,
        calib_path=calib_path,
        spec_mutator=add_300g_box,
        seed=0,
    )
    ep_300g.zero_ee_wrench()
    ep_300g.open_gripper(0.065)
    ep_300g.move_to([0.35, 0.0, 0.10], speed=0.08)
    ep_300g.move_to([0.35, 0.0, 0.045], speed=0.06)
    ep_300g.close_gripper(14.0)
    ep_300g.move_to([0.35, 0.0, 0.14], speed=0.05)
    ep_300g.step_idle(60)
    w_300g = ep_300g.get_ee_wrench("world")

    # 3. Gripper clamping force calibration check across [1.0, 20.0] N
    spec_g = mujoco.MjSpec.from_file(str(model_path))
    spec_g.worldbody.add_geom(
        name="pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.06, 0.06, 0.02],
        pos=[pos_home[0], pos_home[1], pos_home[2] - 0.035],
    )
    box_b = spec_g.worldbody.add_body(name="calib_box", pos=list(pos_home))
    box_b.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    box_b.mass = 0.20
    box_b.ipos = [0.0, 0.0, 0.0]
    box_b.inertia = [1e-4, 1e-4, 1e-4]
    box_b.add_geom(
        name="calib_box_g",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.02, 0.015],
        friction=[1.2, 0.005, 0.0001],
    )
    mg = spec_g.compile()
    dg = mujoco.MjData(mg)
    pad_l = mg.geom("pad_left").id
    pad_r = mg.geom("pad_right").id
    box_g = mg.geom("calib_box_g").id
    grip_forces: dict[float, tuple[float, float]] = {}
    for f_cmd in (1.0, 8.0, 20.0):
        mujoco.mj_resetData(mg, dg)
        dg.qpos[:6] = Q_HOME
        dg.qpos[6:8] = [0.025, -0.025]
        mujoco.mj_forward(mg, dg)
        x0 = dg.site_xpos[mg.site("tcp").id].copy()
        R0 = dg.site_xmat[mg.site("tcp").id].reshape(3, 3).copy()
        for _ in range(250):
            dg.ctrl[:6] = osc_torque_6dof(mg, dg, mg.site("tcp").id, x0, R0)
            dg.ctrl[6:8] = gripper_control_torques(mg, dg, "force", target_force=f_cmd)
            mujoco.mj_step(mg, dg)
        fn_l = fn_r = 0.0
        cf = np.zeros(6)
        for ci in range(dg.ncon):
            c = dg.contact[ci]
            if {c.geom1, c.geom2} == {pad_l, box_g}:
                mujoco.mj_contactForce(mg, dg, ci, cf)
                fn_l += float(cf[0])
            elif {c.geom1, c.geom2} == {pad_r, box_g}:
                mujoco.mj_contactForce(mg, dg, ci, cf)
                fn_r += float(cf[0])
        grip_forces[f_cmd] = (round(fn_l, 3), round(fn_r, 3))

    # 4. Dry-run PiperXEpisode check
    dry_ep = PiperXEpisode(dry_run=True, model_path=model_path)
    dry_snap = dry_ep.snapshot()
    noise = JointNoiseModel(calib_path=calib_path, enabled=True)
    cfg = yaml.safe_load(calib_path.read_text())

    return {
        "q_home": Q_HOME.tolist(),
        "tcp_home": [round(float(v), 4) for v in pos_home],
        "min_workspace_margin_deg": round(min_margin_deg, 2),
        "unloaded_move_to_err_mm": round(unloaded_err_m * 1000.0, 3),
        "payload_300g_fz_n": round(float(w_300g["force"][2]), 3),
        "gripper_forces_n": grip_forces,
        "calib_lpf_hz": noise.lpf_cutoff_hz,
        "calib_keys": sorted(cfg.keys()),
        "dry_run_snapshot_keys": len(dry_snap),
    }


def run_hardware_calibration(can_name: str, calib_out: Path) -> None:
    """Measure zero-bias & noise statistics on live PiPER-X hardware."""
    drv = PiperXDriver(can_name=can_name, dry_run=False)
    drv.connect()
    try:
        est = ProprioceptiveEstimator()
        drv.send_joint_positions(Q_HOME)
        drv.sleep_dt(2.0)
        samples = []
        for _ in range(100):
            q, qd, tau = drv.read_joints()
            samples.append(tau - est._g_hat(q, qd))
            drv.sleep_dt(0.01)
        residuals = np.asarray(samples)
        print("Measured mean joint torque bias (Nm):", np.mean(residuals, axis=0).round(4))
        print("Measured std joint torque noise (Nm):", np.std(residuals, axis=0).round(4))
        print(f"Inspect and update {calib_out} with measured values if needed.")
    finally:
        drv.disconnect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate and validate PiPER-X model and noise parameters.")
    parser.add_argument("--sim-check", action="store_true", help="Run simulation calibration validation check.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --sim-check (no physical hardware).")
    parser.add_argument("--can", default="can0", help="CAN interface name for real hardware.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH, help="Calibration YAML path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_PIPERX_XML, help="PiPER-X MJCF XML path.")
    args = parser.parse_args(argv)

    if args.sim_check or args.dry_run:
        report = run_sim_check(args.calib, args.model)
        print(json.dumps(report, indent=2))
        return 0

    run_hardware_calibration(args.can, args.calib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
