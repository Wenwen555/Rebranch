"""Constant lifting, in-place instantiation, and inline literal hoisting."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import io
import re
import tokenize
from typing import Any, Mapping

from rebranch.api.primitives import PHYSCAP_EXTRA_PRIMITIVES, PRIMITIVES


_RANGE_RE = re.compile(
    r"#\s*\[\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\]"
)
_SET_RE = re.compile(r"#\s*\{([^}]+)\}")

_HOIST_KWARG_RANGES: dict[str, tuple[float, float]] = {
    "force": (1.0, 20.0),
    "max_force": (1.0, 20.0),
    "max_torque": (0.1, 5.0),
    "speed": (0.01, 0.20),
}
_ALL_PRIMITIVES: frozenset[str] = frozenset(PRIMITIVES) | frozenset(
    PHYSCAP_EXTRA_PRIMITIVES
)


@dataclass(frozen=True)
class ConstSpec:
    """Specification of a lifted top-level policy constant."""

    name: str
    kind: str  # "continuous" | "discrete"
    initial: int | float | str
    lo: float | None = None
    hi: float | None = None
    choices: tuple[Any, ...] | None = None
    lineno: int = 0
    is_int: bool = False


def _extract_literal(node: ast.AST) -> int | float | str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, int):
            return node.value
        if isinstance(node.value, float):
            return node.value
        if isinstance(node.value, str):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _extract_literal(node.operand)
        if isinstance(inner, (int, float)):
            return -inner
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        inner = _extract_literal(node.operand)
        if isinstance(inner, (int, float)):
            return inner
    return None


def _line_comments(source: str) -> dict[int, str]:
    comments: dict[int, str] = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_str, start, _, _ in tokens:
            if tok_type == tokenize.COMMENT:
                comments[start[0]] = tok_str
    except tokenize.TokenError:
        pass
    return comments


def _parse_annotation(comment: str) -> tuple[str, Any] | None:
    m_range = _RANGE_RE.search(comment)
    if m_range:
        lo = float(m_range.group(1))
        hi = float(m_range.group(2))
        if lo <= hi:
            return ("continuous", (lo, hi))
        return ("continuous", (hi, lo))

    m_set = _SET_RE.search(comment)
    if m_set:
        inner = m_set.group(1).strip()
        try:
            vals = ast.literal_eval("[" + inner + "]")
        except Exception:
            return None
        if isinstance(vals, list) and vals:
            return ("discrete", tuple(vals))
    return None


def _top_level_const_nodes(
    tree: ast.Module,
) -> list[tuple[str, ast.AST]]:
    found: list[tuple[str, ast.AST]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id.isupper() and not tgt.id.startswith("_"):
                found.append((tgt.id, stmt.value))
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            tgt = stmt.target
            if isinstance(tgt, ast.Name) and tgt.id.isupper() and not tgt.id.startswith("_"):
                found.append((tgt.id, stmt.value))
    return found


def lift_constants(source: str) -> list[ConstSpec]:
    """Extract annotated top-level uppercase constants from `source`."""
    tree = ast.parse(source, filename="<policy>")
    comments = _line_comments(source)
    specs: list[ConstSpec] = []

    for name, val_node in _top_level_const_nodes(tree):
        lit = _extract_literal(val_node)
        if lit is None:
            continue
        lineno = int(getattr(val_node, "end_lineno", None) or val_node.lineno)
        comment = comments.get(lineno) or comments.get(val_node.lineno, "")
        if not comment:
            continue
        ann = _parse_annotation(comment)
        if ann is None:
            continue
        kind, payload = ann
        if kind == "continuous" and isinstance(lit, (int, float)):
            lo, hi = payload
            specs.append(
                ConstSpec(
                    name=name,
                    kind="continuous",
                    initial=lit,
                    lo=float(lo),
                    hi=float(hi),
                    lineno=int(val_node.lineno),
                    is_int=isinstance(lit, int) and not isinstance(lit, bool),
                )
            )
        elif kind == "discrete":
            choices = tuple(payload)
            if lit not in choices:
                choices = (lit,) + choices
            specs.append(
                ConstSpec(
                    name=name,
                    kind="discrete",
                    initial=lit,
                    choices=choices,
                    lineno=int(val_node.lineno),
                )
            )
    return specs


def _format_value(val: Any) -> str:
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, int) and not isinstance(val, bool):
        return repr(val)
    if isinstance(val, float):
        return repr(val)
    raise TypeError(f"Unsupported constant value type: {type(val)!r}")


def _replace_spans(
    source: str, replacements: list[tuple[int, int, int, int, str]]
) -> str:
    """Replace `(lineno, col, end_lineno, end_col, text)` spans in `source`."""
    if not replacements:
        return source
    lines = source.splitlines(keepends=True)
    # Sort descending so earlier byte offsets remain valid.
    for lineno, col, end_lineno, end_col, new_text in sorted(
        replacements, key=lambda r: (r[0], r[1]), reverse=True
    ):
        if lineno == end_lineno:
            line_b = lines[lineno - 1].encode("utf-8")
            updated_b = line_b[:col] + new_text.encode("utf-8") + line_b[end_col:]
            lines[lineno - 1] = updated_b.decode("utf-8")
        else:
            start_b = lines[lineno - 1].encode("utf-8")[:col]
            end_b = lines[end_lineno - 1].encode("utf-8")[end_col:]
            lines[lineno - 1 : end_lineno] = [
                (start_b + new_text.encode("utf-8") + end_b).decode("utf-8")
            ]
    return "".join(lines)


def instantiate(source: str, values: Mapping[str, Any]) -> str:
    """Return `source` with top-level constant values replaced in-place.

    Preserves line numbers and `# [lo, hi]` / `# {"a", "b"}` comments.
    """
    if not values:
        return source
    tree = ast.parse(source, filename="<policy>")
    replacements: list[tuple[int, int, int, int, str]] = []
    for name, val_node in _top_level_const_nodes(tree):
        if name not in values:
            continue
        if (
            val_node.lineno is None
            or val_node.col_offset is None
            or val_node.end_lineno is None
            or val_node.end_col_offset is None
        ):
            continue
        replacements.append(
            (
                int(val_node.lineno),
                int(val_node.col_offset),
                int(val_node.end_lineno),
                int(val_node.end_col_offset),
                _format_value(values[name]),
            )
        )
    return _replace_spans(source, replacements)


def _all_identifiers(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                names.add(arg.arg)
    return names


def hoist_inline_literals(
    source: str,
    target_kwargs: tuple[str, ...] = ("force", "max_force", "max_torque", "speed"),
) -> str:
    """Hoist literal primitive kwargs (`force`, `max_force`, `max_torque`, `speed`) to top-level constants."""
    tree = ast.parse(source, filename="<policy>")
    used_names = _all_identifiers(tree)
    target_set = set(target_kwargs)

    hoist_items: list[tuple[int, int, int, int, str, float, float, float]] = []
    # Collect in source order so numbering is deterministic.
    calls: list[ast.Call] = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    calls.sort(key=lambda c: (c.lineno, c.col_offset))

    for call in calls:
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "api"
            and func.attr in _ALL_PRIMITIVES
        ):
            continue
        for kw in call.keywords:
            if kw.arg is None or kw.arg not in target_set:
                continue
            lit = _extract_literal(kw.value)
            if not isinstance(lit, float):
                continue
            base_name = f"{func.attr}_{kw.arg}".upper()
            cand_name = base_name
            suffix = 2
            while cand_name in used_names:
                cand_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(cand_name)

            lo, hi = _HOIST_KWARG_RANGES.get(kw.arg, (min(0.0, lit), max(1.0, lit)))
            lo = min(lo, lit)
            hi = max(hi, lit)
            hoist_items.append(
                (
                    int(kw.value.lineno),
                    int(kw.value.col_offset),
                    int(kw.value.end_lineno),
                    int(kw.value.end_col_offset),
                    cand_name,
                    lit,
                    lo,
                    hi,
                )
            )

    if not hoist_items:
        return source

    replacements = [
        (lineno, col, end_lineno, end_col, name)
        for lineno, col, end_lineno, end_col, name, _, _, _ in hoist_items
    ]
    updated = _replace_spans(source, replacements)

    new_lines = [
        f"{name} = {repr(float(val))}  # [{lo:g}, {hi:g}]\n"
        for _, _, _, _, name, val, lo, hi in hoist_items
    ]

    # Insert new constants right before the first `def ` statement (after any
    # module docstring or existing top-level constants).
    updated_tree = ast.parse(updated, filename="<policy>")
    insert_lineno = 1
    for stmt in updated_tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            insert_lineno = int(stmt.lineno)
            break
    else:
        insert_lineno = len(updated.splitlines()) + 1

    lines = updated.splitlines(keepends=True)
    idx = max(0, insert_lineno - 1)
    prefix = lines[:idx]
    suffix_lines = lines[idx:]
    if prefix and not prefix[-1].endswith("\n"):
        prefix[-1] = prefix[-1] + "\n"
    if suffix_lines and suffix_lines[0].strip():
        new_lines.append("\n")
    return "".join(prefix + new_lines + suffix_lines)
