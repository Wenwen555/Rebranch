"""Contact insertion into an unfixed socket with public geometric pose feedback."""
from __future__ import annotations

import itertools
import math
from typing import Any

import mujoco
import numpy as np

from rebranch.sim.backend import EpisodeResult, TaskBackend
from rebranch.sim.mujoco_episode import MujocoEpisode
from rebranch.sim.params import Continuous, ParamSpace
from rebranch.tasks._common import DEFAULT_OBS_API_MAP, DEFAULT_OBS_SCALE, TABLE_Z, add_table

PEG_HALF = (0.01225, 0.01225, 0.032)  # Physical collision extents, metres.
PEG_START = (0.30, -0.09, TABLE_Z + PEG_HALF[2])
BOX_HALF = (0.055, 0.055, 0.019)
BOX_START = (0.38, 0.07, TABLE_Z + BOX_HALF[2])
HOLE_HALF = 0.0154
BASE_THICKNESS = 0.004
HOLE_FLOOR_Z = -BOX_HALF[2] + BASE_THICKNESS
HOLD_SECONDS = 1.0


def insertion_state(ep: MujocoEpisode) -> dict[str, Any]:
    """Read private geometry and load-bearing contacts for scoring and diagnostics."""
    model, data = ep.model, ep.data
    peg_id, box_id = model.body('peg').id, model.body('socket_box').id
    peg_R = data.xmat[peg_id].reshape(3, 3)
    box_R = data.xmat[box_id].reshape(3, 3)
    bottom = np.array([(x, y, -PEG_HALF[2]) for x, y in
                       itertools.product((-PEG_HALF[0], PEG_HALF[0]),
                                         (-PEG_HALF[1], PEG_HALF[1]))])
    bottom_world = bottom @ peg_R.T + data.xpos[peg_id]
    bottom_local = (bottom_world - data.xpos[box_id]) @ box_R
    hole_xy = model.site_pos[model.site('hole').id, :2]
    inside = bool(np.all(np.abs(bottom_local[:, :2] - hole_xy) <= HOLE_HALF))
    depth = BOX_HALF[2] - float(np.max(bottom_local[:, 2]))
    tilt = math.degrees(math.acos(float(np.clip(np.dot(peg_R[:, 2], box_R[:, 2]), -1, 1))))
    base_id, table_id = model.geom('socket_base').id, model.geom('table').id
    peg_gid = model.geom('peg_col').id
    peg_supported, box_supported = False, False
    robot_contacts, contacts = [], []
    force = np.zeros(6)  # Contact-normal force scratch, N.
    for index in range(data.ncon):
        c = data.contact[index]
        a, b = int(c.geom1), int(c.geom2)
        ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if ba not in (peg_id, box_id) and bb not in (peg_id, box_id):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] <= 1e-6:
            continue
        names = (model.geom(a).name, model.geom(b).name)
        contacts.append({'geoms': names, 'normal_force_n': float(force[0]),
                         'friction': c.friction.tolist()})
        pair = {a, b}
        peg_supported |= pair == {peg_gid, base_id}
        box_supported |= pair == {base_id, table_id}
        if ((ba in (peg_id, box_id) and bb not in (0, peg_id, box_id)) or
                (bb in (peg_id, box_id) and ba not in (0, peg_id, box_id))):
            robot_contacts.append(names)
    speeds = {}
    for name in ('peg', 'socket_box'):
        dof = model.jnt_dofadr[model.joint(f'{name}_free').id]
        speeds[name] = {'linear_m_s': float(np.linalg.norm(data.qvel[dof:dof + 3])),
                        'angular_rad_s': float(np.linalg.norm(data.qvel[dof + 3:dof + 6]))}
    full_corners = np.array(list(itertools.product(*[(-v, v) for v in PEG_HALF])))
    top_z = float(np.max((full_corners @ peg_R.T + data.xpos[peg_id])[:, 2]))
    on_pad = bool(np.all(np.abs(data.xpos[box_id, :2] - np.array(BOX_START[:2])) <= .035))
    checks = {
        'insertion_depth': depth >= .020,
        'peg_inside_opening': inside,
        'relative_tilt': tilt <= 5.0,
        'peg_on_socket_base': bool(peg_supported),
        'box_on_table': bool(box_supported),
        'robot_clear': not robot_contacts,
        'released': not ep.get_gripper_state()['grasped'],
        'withdrawn': float(data.site_xpos[ep.tcp_id, 2]) >= top_z + .080,
        'box_on_pad': on_pad,
        'settled': all(v['linear_m_s'] <= .010 and v['angular_rad_s'] <= .050
                       for v in speeds.values()),
    }
    return {'sim_time_s': ep.sim_time, 'checks': checks, 'depth_m': depth,
            'relative_tilt_deg': tilt, 'speeds': speeds, 'contacts': contacts,
            'box_pose': ep.get_object_pose('socket_box'),
            'peg_pose': ep.get_object_pose('peg'), 'tcp_pose': ep.get_ee_pose(),
            'lower_face_box_xyz': bottom_local.tolist(), 'robot_contacts': robot_contacts}


class ContactInsertionTask(TaskBackend):
    """Seat a peg in a freely moving socket and leave both supported after release."""
    name = 'contact_insertion'
    requires_branching = False  # Whether active branching helps is an experimental question.
    probe_config = None
    task_description = (
        "Insert the 24.5 x 24.5 x 64 mm peg 'peg' into the 30.8 mm square opening 'hole' "
        "of the unfixed 110 x 110 x 38 mm 'socket_box'. The opening pose is public and "
        "tracks the current box: its origin is the cavity floor, 34 mm below the box top. "
        "All named body/geom/site poses are public at the same precision. Box mass and "
        "contact friction are unknown; contact can move the box or peg inside the gripper. "
        "The table is at world z=.020 m, and the box starts centered at (.38,.07) m. "
        "After inserting, open the gripper and retreat at least 80 mm above the peg top. "
        "For a one-second hold the peg must be supported on the box's cavity floor, "
        "inserted at least 20 mm with its lower face inside the opening and relative tilt "
        "at most 5 degrees. Box must remain on the table within +/-35 mm of its initial "
        "x/y center; neither body may contact the robot. Linear/angular speeds must stay "
        "below .010 m/s and .050 rad/s. Use the same 200-call budget. A 24 mm upper-side "
        "grasp offset leaves pad clearance above the rim; robot motion is not guaranteed "
        "peg motion. No special force threshold models irreversible part damage."
    )

    def __init__(self) -> None:
        self.param_space = ParamSpace([
            Continuous('hole_offset_x', -.008, .008, nominal=0.0),
            Continuous('hole_offset_y', -.008, .008, nominal=0.0),
            Continuous('box_mass_kg', .20, .85, nominal=.525),
            Continuous('base_friction', .25, .65, nominal=.45),
            Continuous('grip_friction', .25, 1.20, nominal=.60),
        ])
        self.obs_scale = dict(DEFAULT_OBS_SCALE)
        self.obs_api_map = dict(DEFAULT_OBS_API_MAP)
        self.draft_policy = self.conservative_policy()

    def make_episode(self, theta: dict[str, Any], seed: int = 0) -> MujocoEpisode:
        """Build the free box, collision opening and explicit material-pair contacts."""
        ox, oy = float(theta['hole_offset_x']), float(theta['hole_offset_y'])
        mass = float(theta['box_mass_kg'])

        def mutate(spec: mujoco.MjSpec) -> None:
            add_table(spec, friction=(.7, .005, .0001))
            spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                   size=[1, 1, .1], rgba=[.25, .25, .25, 1])
            box = spec.worldbody.add_body(name='socket_box', pos=BOX_START)
            box.add_joint(name='socket_box_free', type=mujoco.mjtJoint.mjJNT_FREE)
            box.mass, box.ipos = mass, [0, 0, 0]
            box.inertia = [mass * (BOX_HALF[(i + 1) % 3] ** 2 +
                                   BOX_HALF[(i + 2) % 3] ** 2) / 3 for i in range(3)]
            box.add_geom(name='socket_base', type=mujoco.mjtGeom.mjGEOM_BOX,
                         size=[.055, .055, .002], pos=[0, 0, -.017],
                         friction=[.35, .005, .0001], rgba=[.35, .40, .48, 1])
            r, outer = HOLE_HALF, BOX_HALF[0]
            # Four non-overlapping wall boxes leave the translated square opening visible.
            walls = [
                ('xp', [(ox + r + outer) / 2, 0, .002], [(outer - ox - r) / 2, outer, .017]),
                ('xn', [(ox - r - outer) / 2, 0, .002], [(outer + ox - r) / 2, outer, .017]),
                ('yp', [ox, (oy + r + outer) / 2, .002], [r, (outer - oy - r) / 2, .017]),
                ('yn', [ox, (oy - r - outer) / 2, .002], [r, (outer + oy - r) / 2, .017]),
            ]
            for name, pos, size in walls:
                box.add_geom(name=f'hole_wall_{name}', type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=pos, size=size, friction=[.35, .005, .0001],
                             rgba=[.35, .40, .48, 1])
            s, c = math.sin(math.pi / 8), math.cos(math.pi / 8)
            offset, z = r + .0018, BOX_HALF[2] + .0018
            chamfers = [
                ('xp', [ox + offset, oy, z], [c, 0, -s, 0], [.002, r + .006, .002]),
                ('xn', [ox - offset, oy, z], [c, 0, s, 0], [.002, r + .006, .002]),
                ('yp', [ox, oy + offset, z], [c, s, 0, 0], [r + .006, .002, .002]),
                ('yn', [ox, oy - offset, z], [c, -s, 0, 0], [r + .006, .002, .002]),
            ]
            for name, pos, quat, size in chamfers:
                box.add_geom(name=f'chamfer_{name}', type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=pos, quat=quat, size=size, friction=[.35, .005, .0001],
                             rgba=[.48, .52, .60, 1])
            box.add_site(name='hole', pos=[ox, oy, HOLE_FLOOR_Z], size=[.001, .001, .001],
                         rgba=[1, .8, .2, 0])
            peg = spec.worldbody.add_body(name='peg', pos=PEG_START)
            peg.add_joint(name='peg_free', type=mujoco.mjtJoint.mjJNT_FREE)
            peg.mass, peg.ipos, peg.inertia = .16, [0, 0, 0], [1e-4, 1e-4, 1e-4]
            peg.add_geom(name='peg_col', type=mujoco.mjtGeom.mjGEOM_BOX, size=PEG_HALF,
                         friction=[.35, .005, .0001], rgba=[.84, .62, .22, 1])
            # Pair-specific friction avoids the max(geom friction) combination rule.
            for pad in ('pad_left', 'pad_right'):
                mu = float(theta['grip_friction'])
                spec.add_pair(name=f'{pad}_peg', geomname1=pad, geomname2='peg_col',
                              condim=3, friction=[mu, mu, .005, .0001, .0001],
                              solref=[.0125, 1], solimp=[.925, .97, .001, .5, 2])
            mu = float(theta['base_friction'])
            spec.add_pair(name='box_table', geomname1='socket_base', geomname2='table',
                          condim=3, friction=[mu, mu, .005, .0001, .0001])

        def evaluate(ep: MujocoEpisode) -> EpisodeResult:
            """Require actual released support throughout the one-second final hold."""
            rows = []
            passed = {key: True for key in insertion_state(ep)['checks']}
            for step in range(round(HOLD_SECONDS / ep.dt)):
                ep.step_idle(1)
                row = insertion_state(ep)
                for key, value in row['checks'].items():
                    passed[key] &= value
                if step % 25 == 0:
                    rows.append(row)
            ep._contact_insertion_diagnostics = {'checks_all_hold': passed,
                                                  'hold_samples': rows,
                                                  'terminal': row}
            failed = [key for key, value in passed.items() if not value]
            return EpisodeResult(success=not failed, score=.9 if not failed else -.5,
                                 failure=', '.join(failed) if failed else None)

        episode = MujocoEpisode(theta=theta, seed=seed, spec_mutator=mutate,
                                evaluate_fn=evaluate)
        episode.protocol_version = 'contact-insertion-v1'
        return episode

    def oracle_policy(self, theta: dict[str, Any]) -> str:
        """Use the real offset with live public transforms and normal physical actions."""
        from rebranch.baselines.contact_insertion_controls import control_source
        return control_source('oracle', theta=theta)

    def conservative_policy(self) -> str:
        """Expose the fixed compliant program as the initial task policy."""
        from rebranch.baselines.contact_insertion_controls import control_source
        return control_source('fixed_compliant')

    def latent_mode(self, theta: dict[str, Any]) -> str:
        """No ground-truth discrete modes are asserted for continuous contact dynamics."""
        return 'continuous_contact_parameters'
