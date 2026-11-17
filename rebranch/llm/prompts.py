"""Shared prompt templates ensuring zero ground-truth theta parameter leakage."""

from __future__ import annotations

from rebranch.api.primitives import API_DOC

SHARED_PROMPT_VERSION = "stage1d-common-control-v2"

SYSTEM_PROMPT = (
    "You are an expert robotics manipulation engineer writing safe, deterministic "
    "Python control policies for a 6-DOF PiPER-X arm using only the provided RobotAPI. "
    "Return ONLY valid Python code inside a ```python ... ``` block defining `def run(api):` "
    "with top-level uppercase tunable constants annotated with range comments `# [lo, hi]`. "
    "Never use `import` statements (`math` is pre-injected). "
    "Respect the documented per-policy API-call budget. Avoid speculative preflight "
    "checks, unjustified early exits, and generic recovery or return-to-home routines; use feedback only as needed "
    "for the stated task."
)

PHYSCAP_EXTRA_DOC = """\
Additional Online-Probe Proxy Sensing Primitives (not a PhysCaP reproduction):
The wrapper records an unloaded paired reference after each unloaded move_to using
12 samples 20 ms apart (0.24 s hold). These holds advance and are included in sim_time;
they are measurement actions, not cost-free getters.
11. `api.get_mass() -> float | None`
    Paired unloaded/loaded z-force estimate at the same EE pose (within 2 mm). May make small SDK pose-feedback corrections to compensate load deflection, with object pose confirming >=2 cm lift. Returns None unless these observations exist.
12. `api.get_stiffness() -> float | None`
    Active 1 N / 3 N settled squeeze secant (N/m), restores prior force. Moves the gripper; allowed only at the previously observed support height, never on a suspended payload. Returns None when width change is below 0.5 mm resolution or no grasp. None is not zero or soft.
"""


def build_repair_prompt(
    task_description: str,
    current_code: str,
    failure_summary: str,
    extra_notes: str = "",
    api_doc: str = API_DOC,
) -> str:
    """Prompt to repair a policy on a single observed failure trace (no theta values)."""
    notes_block = f"\nAdditional Notes:\n{extra_notes}\n" if extra_notes else ""
    return (
        f"Task:\n{task_description}\n\n"
        f"{api_doc}\n\n"
        f"Current Policy:\n```python\n{current_code.strip()}\n```\n\n"
        f"Observed Execution Failure (public execution and terminal evaluation feedback):\n{failure_summary.strip()}\n"
        f"{notes_block}\n"
        "Revise the policy so it succeeds without violating safety limits. "
        "Keep tunable constants at module top level with `# [lo, hi]` comments. "
        "Return ONLY the complete updated Python code in a ```python``` block."
    )


def build_branch_prompt(
    task_description: str,
    current_code: str,
    conflict_summary: str,
    distinguishing_observables: str = "",
    failure_feedback: str = "",
    api_doc: str = API_DOC,
) -> str:
    """Prompt to insert an explicit sensory probe and conditional branch for conflicting cases."""
    obs_block = (
        f"\nDistinguishing SDK Observables Across Conflicting Regimes:\n"
        f"{distinguishing_observables.strip()}\n"
        if distinguishing_observables
        else ""
    )
    fb_block = (
        f"\nPrevious Candidate Verification Feedback:\n{failure_feedback.strip()}\n"
        if failure_feedback
        else ""
    )
    return (
        f"Task:\n{task_description}\n\n"
        f"{api_doc}\n\n"
        f"Current Policy:\n```python\n{current_code.strip()}\n```\n\n"
        "Constant retuning did not produce a policy that passed these cases within the evaluation budget.\n"
        f"Conflict Evidence (SDK traces only):\n{conflict_summary.strip()}\n"
        f"{obs_block}{fb_block}\n"
        "Insert a safe, lightweight sensory probe (e.g. `zero_ee_wrench()`, light grasp / "
        "micro-lift, `move_until_contact()`, or `move_compliant()`), read SDK observables "
        "(`get_ee_wrench()`, `get_gripper_state()`, `get_ee_pose()`, `get_object_pose()`), "
        "and branch cleanly into distinct `if / elif / else` strategies (scaling any "
        "continuous compensation smoothly with the measured probe signal when variation is "
        "continuous) so each regime succeeds safely.\n"
        "Declare tunable thresholds and branch parameters as top-level uppercase constants "
        "with `# [lo, hi]` comments. Return ONLY the complete Python code in a ```python``` block."
    )


def build_single_twin_prompt(
    task_description: str,
    current_code: str,
    failure_summary: str,
    api_doc: str = API_DOC,
) -> str:
    """Prompt for Single-Twin + CaP baseline debugging."""
    return build_repair_prompt(
        task_description=task_description,
        current_code=current_code,
        failure_summary=failure_summary,
        api_doc=api_doc,
    )


def build_uniform_dr_prompt(
    task_description: str,
    current_code: str,
    failure_summary: str,
    api_doc: str = API_DOC,
    rejected_attempt: str = "",
) -> str:
    """Prompt for Uniform-DR + CaP baseline summarizing multi-twin failures."""
    rejected_context = f"Previous Rejected Attempt:\n{rejected_attempt}\n\n" if rejected_attempt else ""
    return (
        f"Task:\n{task_description}\n\n"
        f"{api_doc}\n\n"
        f"Current Policy:\n```python\n{current_code.strip()}\n```\n\n"
        f"Failure Summary Across Sampled Trials (public execution and terminal evaluation feedback):\n"
        f"{failure_summary.strip()}\n\n"
        f"{rejected_context}"
        "Improve the policy to maximize success across all observed failure modes. "
        "You may adjust parameters, add safe sensory probes, or add conditional branches. "
        "Keep tunable constants at top level with `# [lo, hi]` comments. "
        "Return ONLY the complete Python code in a ```python``` block."
    )


def build_physcap_prompt(
    task_description: str,
    current_code: str,
    obs_summary: str,
    extra_primitives_doc: str = PHYSCAP_EXTRA_DOC,
    api_doc: str = API_DOC,
) -> str:
    """Prompt for PhysCaP baseline with SDK-derived mass/stiffness primitives."""
    return (
        f"Task:\n{task_description}\n\n"
        f"{api_doc}\n\n"
        f"{extra_primitives_doc.strip()}\n\n"
        f"Current Policy:\n```python\n{current_code.strip()}\n```\n\n"
        f"Recent Execution Observations:\n{obs_summary.strip()}\n\n"
        "Write an updated policy using the available primitives (including `get_mass()` "
        "and `get_stiffness()` where helpful) to handle the physical variation robustly. "
        "Return ONLY the complete Python code in a ```python``` block."
    )
