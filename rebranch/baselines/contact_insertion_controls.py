"""Four explicit controls for the unfixed-socket mechanism acceptance batch."""
from __future__ import annotations

from textwrap import dedent, indent
from typing import Any

_HELPERS = '''
GRIP_FORCE = 14.0  # [1.0, 20.0], same ordinary grip across all four controls.
GRASP_HEIGHT = 0.024  # m above peg center, to keep pads above the socket rim.

def _rotate(q, v):
    # Rotate an object-local displacement into world coordinates.
    w, x, y, z = q
    a, b, c = v
    return [(1-2*(y*y+z*z))*a + 2*(x*y-z*w)*b + 2*(x*z+y*w)*c,
            2*(x*y+z*w)*a + (1-2*(x*x+z*z))*b + 2*(y*z-x*w)*c,
            2*(x*z-y*w)*a + 2*(y*z+x*w)*b + (1-2*(x*x+y*y))*c]

def _world(pose, local):
    # Named hole origin is the physical cavity floor, not the top surface.
    delta = _rotate(pose['quat'], local)
    return [pose['pos'][i] + delta[i] for i in range(3)]

def _conjugate(q):
    # Public pose quaternions are unit rotations in wxyz order.
    return [q[0], -q[1], -q[2], -q[3]]

def _multiply(a, b):
    # Compose rotations: apply b first, then a.
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]

def _tcp_pose(peg_target, peg_quat, peg, ee):
    # Transport the measured rigid peg-in-TCP transform into the desired peg pose.
    inverse_tcp = _conjugate(ee['quat'])
    grasp_quat = _multiply(inverse_tcp, peg['quat'])
    grasp_pos = _rotate(inverse_tcp, [peg['pos'][i]-ee['pos'][i] for i in range(3)])
    target_quat = _multiply(peg_quat, _conjugate(grasp_quat))
    offset = _rotate(target_quat, grasp_pos)
    return {'pos': [peg_target[i]-offset[i] for i in range(3)], 'quat': target_quat}

def _pick(api):
    # Grip the upper side and lift clear of the table before approaching the box.
    api.zero_ee_wrench()
    p = api.get_object_pose('peg')['pos']
    q = api.get_ee_pose()['quat']
    api.open_gripper(.060)
    api.move_to([p[0], p[1], p[2] + GRASP_HEIGHT + .070], quat=q, speed=.08)
    api.move_to([p[0], p[1], p[2] + GRASP_HEIGHT], quat=q, speed=.04)
    api.close_gripper(force=GRIP_FORCE)
    api.move_to([p[0], p[1], p[2] + GRASP_HEIGHT + .110], quat=q, speed=.05)

def _prepare(api):
    # All controls align above the visible hole and remeasure the actual grasp there.
    hole = api.get_object_pose('hole')
    _pick(api)
    peg = api.get_object_pose('peg')
    ee = api.get_ee_pose()
    target = _tcp_pose(_world(hole, [0, 0, .112]), hole['quat'], peg, ee)
    api.move_to(target['pos'], quat=target['quat'], speed=.05)
    hole = api.get_object_pose('hole')
    peg = api.get_object_pose('peg')
    ee = api.get_ee_pose()
    return hole, peg, ee

def _release(api):
    # Small release clears the peg; withdraw above the rim before opening fully.
    api.open_gripper(.033)
    p = api.get_ee_pose()['pos']
    api.move_to([p[0], p[1], p[2]+.120], speed=.05)
    api.open_gripper(.060)

def _entered(peg, hole):
    # Infer insertion from the public peg face in the live box frame.
    q = hole['quat']
    inv = [q[0], -q[1], -q[2], -q[3]]
    lower = []
    for x in (-.01225, .01225):
        for y in (-.01225, .01225):
            v = _world(peg, [x, y, -.032])
            lower.append(_rotate(inv, [v[i] - hole['pos'][i] for i in range(3)]))
    normal = _rotate(peg['quat'], [0, 0, 1])
    axis = _rotate(hole['quat'], [0, 0, 1])
    return (.034 - max(p[2] for p in lower) >= .020
            and all(abs(p[0]) <= .0154 and abs(p[1]) <= .0154 for p in lower)
            and sum(normal[i]*axis[i] for i in range(3)) >= math.cos(math.radians(5)))
'''


def control_source(method: str, theta: dict[str, Any] | None = None) -> str:
    """Generate one declared controller; only the oracle uses true local hole offset."""
    if method == 'oracle':
        body = f'''
        def run(api):
            hole, peg, ee = _prepare(api)
            box = api.get_object_pose('socket_box')
            hole = {{'pos': _world(box, [{float(theta['hole_offset_x'])!r},
                                      {float(theta['hole_offset_y'])!r}, -.015]),
                    'quat': box['quat']}}
            target = _tcp_pose(_world(hole, [0, 0, .033]), hole['quat'], peg, ee)
            api.move_to(target['pos'], quat=target['quat'], speed=.02)
            _release(api)
        '''
    elif method == 'fixed_compliant':
        body = '''
        def run(api):
            hole, peg, ee = _prepare(api)
            target = _tcp_pose(_world(hole, [0, 0, .033]), hole['quat'], peg, ee)
            api.move_compliant(target['pos'], quat=target['quat'], free_axes='xyXYZ', speed=.02)
            _release(api)
        '''
    elif method in ('world_search_cached', 'peck_retrack'):
        # Nine locations are a finite recovery search, shared by both versions.
        grid = [(0, 0), (-.008, 0), (0, -.008), (0, .008), (.008, 0),
                (-.008, -.008), (-.008, .008), (.008, -.008), (.008, .008)]
        refresh = ("        hole = api.get_object_pose('hole')\n"
                   "        peg = api.get_object_pose('peg')\n"
                   "        ee = api.get_ee_pose()\n") if method == 'peck_retrack' else ''
        body = dedent(f'''
        def run(api):
            hole, peg, ee = _prepare(api)
            for index, (dx, dy) in enumerate({grid!r}):
                if index:
                    target = _tcp_pose(_world(hole, [dx, dy, .112]), hole['quat'], peg, ee)
                    api.move_to(target['pos'], quat=target['quat'], speed=.05)
        ''')
        body += indent(dedent('''
                target = _tcp_pose(_world(hole, [dx, dy, .033]), hole['quat'], peg, ee)
                api.move_until_contact(target['pos'], quat=target['quat'],
                                       speed=.02, max_force=6.0, max_torque=.8)
                observed_peg = api.get_object_pose('peg')
                observed_hole = api.get_object_pose('hole')
                if _entered(observed_peg, observed_hole):
                    _release(api)
                    return
                pose = api.get_ee_pose()
                position = pose['pos']
                api.move_to([position[0], position[1], position[2]+.080],
                            quat=pose['quat'], speed=.05)
        ''').lstrip(), '        ')
        # Re-anchor after the failed contact has been unloaded, before the next candidate.
        body += refresh
        body += "    _release(api)\n"
        return dedent(_HELPERS).lstrip() + '\n' + body
    else:
        raise ValueError(f'Unknown contact insertion control: {method}')
    return dedent(_HELPERS).lstrip() + '\n' + dedent(body).lstrip()
