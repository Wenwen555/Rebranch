"""PiPER-X MuJoCo manipulation task suite (T1-T8)."""

from __future__ import annotations

from rebranch.sim.backend import TaskBackend
from rebranch.tasks.t1_eccentric_box import T1EccentricBoxTask
from rebranch.tasks.t2_cup import T2CupTask
from rebranch.tasks.t3_push_can import T3PushCanTask
from rebranch.tasks.t4_cabinet import T4CabinetTask
from rebranch.tasks.t5_drawer import T5DrawerTask
from rebranch.tasks.t6_valve import T6ValveTask
from rebranch.tasks.t7_peg import T7PegTask
from rebranch.tasks.t8_card import T8CardTask
from rebranch.tasks.t9_supported_tray import T9SupportedTrayTask
from rebranch.tasks.contact_insertion import ContactInsertionTask
from rebranch.tasks.hidden_mass_balance_beam import (
    HiddenMassBalanceBeamATask,
    HiddenMassBalanceBeamBTask,
)

TASKS: dict[str, type[TaskBackend]] = {
    "t1_eccentric_box": T1EccentricBoxTask,
    "t2_cup": T2CupTask,
    "t3_push_can": T3PushCanTask,
    "t4_cabinet": T4CabinetTask,
    "t5_drawer": T5DrawerTask,
    "t6_valve": T6ValveTask,
    "t7_peg": T7PegTask,
    "t8_card": T8CardTask,
    "t9_supported_tray": T9SupportedTrayTask,
    "contact_insertion": ContactInsertionTask,
    "t10_hidden_mass_balance_beam_a": HiddenMassBalanceBeamATask,
    "t10_hidden_mass_balance_beam_b": HiddenMassBalanceBeamBTask,
}

_ALIASES: dict[str, str] = {
    "t1": "t1_eccentric_box",
    "t2": "t2_cup",
    "t3": "t3_push_can",
    "t4": "t4_cabinet",
    "t5": "t5_drawer",
    "t6": "t6_valve",
    "t7": "t7_peg",
    "t8": "t8_card",
}


def make_task(name: str) -> TaskBackend:
    """Instantiate a registered PiPER-X task backend by name (or T1..T8 alias)."""
    key = name.strip().lower()
    canonical = _ALIASES.get(key, key)
    if canonical not in TASKS:
        raise KeyError(f"Unknown task {name!r}; expected one of {sorted(TASKS)}")
    return TASKS[canonical]()


__all__ = [
    "TASKS",
    "T1EccentricBoxTask",
    "T2CupTask",
    "T3PushCanTask",
    "T4CabinetTask",
    "T5DrawerTask",
    "T6ValveTask",
    "T7PegTask",
    "T8CardTask",
    "T9SupportedTrayTask",
    "ContactInsertionTask",
    "HiddenMassBalanceBeamATask",
    "HiddenMassBalanceBeamBTask",
    "make_task",
]
