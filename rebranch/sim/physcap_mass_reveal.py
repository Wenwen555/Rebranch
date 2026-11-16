"""Gated MuJoCo mass reveal adapted from official PhysCaP Appendix A.4.

Upstream source: physcap/physcap, commit 556401e30b10c22e9853b224e7e3955f16f1dd05.
Copyright (c) 2026 Max Fu and The PhysCaP Authors; licensed under MIT.
See baselines/physcap_appendix_assets/LICENSE and SOURCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


_GRIPPER_GEOM_TOKENS = ("finger", "gripper", "pad")


@dataclass
class _MassGate:
    name: str
    body_id: int
    geom_ids: set[int]
    reference_z: float
    mass_kg: float
    grip_s: float = 0.0
    lifted_s: float = 0.0
    revealed: bool = False


class PhysCaPMassReveal:
    """Track actual gripper contact and object height after each MuJoCo step."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        grip_duration_s: float = 0.5,
        lift_duration_s: float = 0.5,
        lift_threshold_m: float = 0.03,
    ) -> None:
        self.model = model
        self.data = data
        self.dt = float(model.opt.timestep)
        self.grip_duration_s = float(grip_duration_s)
        self.lift_duration_s = float(lift_duration_s)
        self.lift_threshold_m = float(lift_threshold_m)
        self._gripper_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if any(
                token in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
                for token in _GRIPPER_GEOM_TOKENS
            )
        }
        self._gates: dict[str, _MassGate] = {}
        for body_id in range(1, model.nbody):
            joint_start = int(model.body_jntadr[body_id])
            joint_count = int(model.body_jntnum[body_id])
            has_free_joint = any(
                int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
                for joint_id in range(joint_start, joint_start + joint_count)
            )
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not has_free_joint or not name:
                continue
            body_ids = self._subtree(body_id)
            geom_ids = {
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in body_ids
            }
            if not geom_ids:
                continue
            mass = sum(float(model.body_mass[item]) for item in body_ids)
            self._gates[name] = _MassGate(
                name=name,
                body_id=body_id,
                geom_ids=geom_ids,
                reference_z=float(data.xpos[body_id][2]),
                mass_kg=mass,
            )

    def _subtree(self, root_id: int) -> set[int]:
        body_ids: set[int] = set()
        for body_id in range(self.model.nbody):
            current = body_id
            while current > 0:
                if current == root_id:
                    body_ids.add(body_id)
                    break
                current = int(self.model.body_parentid[current])
        return body_ids

    def _gripped(self, gate: _MassGate) -> bool:
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if (
                geom1 in gate.geom_ids and geom2 in self._gripper_geom_ids
            ) or (
                geom2 in gate.geom_ids and geom1 in self._gripper_geom_ids
            ):
                return True
        return False

    def update(self) -> None:
        """Update continuous duration gates after one real MuJoCo step."""
        for gate in self._gates.values():
            is_gripped = self._gripped(gate)
            is_lifted = (
                float(self.data.xpos[gate.body_id][2]) - gate.reference_z
                >= self.lift_threshold_m
            )
            gate.grip_s = gate.grip_s + self.dt if is_gripped else 0.0
            gate.lifted_s = gate.lifted_s + self.dt if is_lifted else 0.0
            if (
                gate.grip_s >= self.grip_duration_s
                and gate.lifted_s >= self.lift_duration_s
            ):
                gate.revealed = True

    def read(self, name: str) -> float | None:
        """Return exact subtree mass only after both appendix gates pass."""
        gate = self._gates.get(str(name))
        return float(gate.mass_kg) if gate is not None and gate.revealed else None

    def state(self, name: str) -> dict[str, float | bool] | None:
        """Internal diagnostic state; never exposed as a policy primitive."""
        gate = self._gates.get(str(name))
        if gate is None:
            return None
        return {
            "grip_duration_s": gate.grip_s,
            "lift_duration_s": gate.lifted_s,
            "revealed": gate.revealed,
        }
