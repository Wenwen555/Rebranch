"""PiPER-X assembly of native KinDER Dynamic3D MJCF objects."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco

from rebranch.api.proprioception import DEFAULT_PIPERX_XML

BEAM_CENTER = (0.35, 0.0)
SUPPORT_Z = 0.04  # Public tabletop surface height in metres.
PIVOT_HEIGHT = 0.085
BEAM_CLEARANCE = 0.0025
BEAM_HALF_THICKNESS = 0.005
BEAM_HALF_LENGTH = 0.18
BEAM_HALF_WIDTH = 0.035
BLOCK_HALF_SIZE = 0.012
BLOCK_NAMES = ("block_a", "block_b", "block_c")
BLOCK_STARTS = ((0.26, -0.16), (0.26, 0.0), (0.26, 0.16))
ZONE_X = (0.06, 0.168)
ZONE_HALF_Y = 0.027
PAIR_ARM = 0.13
PAIR_GAP = 0.028
# Beam stiffness used by the successful nominal physical diagnostic.
HINGE_STIFFNESS_NM_RAD = 0.8
HINGE_FRICTIONLOSS_NM = 0.001
WITNESS_MARGIN_M = 0.001  # Clearance from the public loading-region boundary.
MASS_RATIO_BOUNDS = (
    PAIR_ARM / (ZONE_X[1] - BLOCK_HALF_SIZE - WITNESS_MARGIN_M),
    PAIR_ARM / (ZONE_X[0] + BLOCK_HALF_SIZE + WITNESS_MARGIN_M),
)
PERMUTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
)
_NATIVE_SEESAW = None


def masses_by_block(theta: dict[str, float]) -> dict[str, float]:
    """Map continuous latent masses to indistinguishable named blocks."""
    permutation = PERMUTATIONS[int(theta["mass_permutation"])]
    latent = (
        float(theta["pair_mass_a_kg"]),
        float(theta["pair_mass_b_kg"]),
        float(theta["mass_ratio"]) * (
            float(theta["pair_mass_a_kg"]) + float(theta["pair_mass_b_kg"])
        ),
    )
    return {name: latent[role] for name, role in zip(BLOCK_NAMES, permutation)}


def nonstack_feasibility_witness(theta: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Construct an interior 2D no-stack placement; the evaluator never uses it."""
    permutation = PERMUTATIONS[int(theta["mass_permutation"])]
    name_by_role = {
        role: BLOCK_NAMES[block_index]
        for block_index, role in enumerate(permutation)
    }
    pair_a = float(theta["pair_mass_a_kg"])
    pair_b = float(theta["pair_mass_b_kg"])
    pair_total = pair_a + pair_b
    ratio = float(theta["mass_ratio"])
    # Separate the same-side pair along beam length while preserving its weighted arm.
    role0_arm = PAIR_ARM + PAIR_GAP * pair_b / pair_total
    role1_arm = PAIR_ARM - PAIR_GAP * pair_a / pair_total
    return {
        name_by_role[0]: (-role0_arm, 0.0),
        name_by_role[1]: (-role1_arm, 0.0),
        name_by_role[2]: (PAIR_ARM / ratio, 0.0),
    }


def build_hidden_mass_balance_model(theta: dict[str, float]) -> mujoco.MjModel:
    """Compile native seesaw/cube MJCF into the existing PiPER-X scene."""
    from rebranch.tasks._common import add_table
    from kinder.envs.dynamic3d.objects.generated_objects import GeneratedSeesaw
    from kinder.envs.dynamic3d.objects.primitive_objects import Cube

    spec = mujoco.MjSpec.from_file(str(DEFAULT_PIPERX_XML))
    # Keep dropped cubes in the physical scene; otherwise gravity sends later
    # object-pose targets arbitrarily far below the robot.
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(1.0, 1.0, 0.1), rgba=(0.2, 0.2, 0.2, 1.0),
    )
    add_table(spec, pos=(0.35, 0.0, 0.0), size=(0.18, 0.25, SUPPORT_Z))
    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    compiler.set("meshdir", str(DEFAULT_PIPERX_XML.parent / "meshes"))
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    global _NATIVE_SEESAW
    if _NATIVE_SEESAW is None:
        _NATIVE_SEESAW = GeneratedSeesaw(
            "balance_support",
            env=None,
            options={
                "beam_length": 0.36,
                "beam_width": 0.07,
                "beam_thickness": 0.01,
                "beam_clearance": BEAM_CLEARANCE,
                "pivot_height": PIVOT_HEIGHT,
                "pivot_width": 0.035,
                "beam_mass": 0.1,
                "pivot_mass": 0.18,
                "damping": 0.02,
                "hinge_frictionloss": HINGE_FRICTIONLOSS_NM,
                "hinge_stiffness": HINGE_STIFFNESS_NM_RAD,
                "hinge_range": [-0.7, 0.7],
                "beam_friction": [1.8, 0.025, 0.002],
                "pivot_friction": [1.5, 0.025, 0.002],
            },
        )
    seesaw = _NATIVE_SEESAW
    for elem in deepcopy(seesaw.get_assets()):
        asset.append(elem)
    support = deepcopy(seesaw.xml_element)
    support.set("pos", f"{BEAM_CENTER[0]} {BEAM_CENTER[1]} {SUPPORT_Z}")
    # Rotate the native local X beam into the published world-Y workspace.
    support.set("quat", "0.7071067811865476 0 0 0.7071067811865475")
    support.remove(support.find("freejoint"))
    # Fixing the native free-root to world changes parent/child collision filtering.
    # Preserve the original articulated-component exclusion for its pivot and beam.
    contact = root.find("contact")
    ET.SubElement(contact, "exclude", body1="balance_support", body2="balance_support_beam")
    beam = support.find("./body")
    beam_top = BEAM_HALF_THICKNESS
    ET.SubElement(beam, "site", name="beam", pos="0 0 0", size="0.003",
                  rgba="0.7 0.7 0.7 0")
    zone_x_mid = (ZONE_X[0] + ZONE_X[1]) / 2
    zone_x_half = (ZONE_X[1] - ZONE_X[0]) / 2
    for name, sign, color in (
        ("zone_left", -1, "0.20 0.55 0.85 0.28"),
        ("zone_right", 1, "0.95 0.63 0.18 0.28"),
    ):
        ET.SubElement(
            beam, "site", name=name, type="box",
            pos=f"{sign * zone_x_mid} 0 {beam_top + 0.0003}",
            size=f"{zone_x_half} {ZONE_HALF_Y} 0.0003",
            rgba=color,
        )
    worldbody.append(support)

    block_masses = masses_by_block(theta)
    for name, (x, y) in zip(BLOCK_NAMES, BLOCK_STARTS):
        block = Cube(
            name,
            env=None,
            options={
                "size": [BLOCK_HALF_SIZE] * 3,
                "rgba": [0.24, 0.62, 0.38, 1.0],
                "mass": block_masses[name],
            },
        )
        body = block.xml_element
        body.set("pos", f"{x} {y} {SUPPORT_Z + BLOCK_HALF_SIZE}")
        body.find("freejoint").set("name", f"{name}_free")
        body.find("geom").set("name", f"{name}_geom")
        worldbody.append(body)

    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode")).compile()
