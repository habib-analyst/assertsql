"""Plan-diff explainer: what changed between repair attempts, and why.

Two layers:
  * diff_ir(old, new): structural diff of the QueryIR, rendered as plain
    sentences ("added LIMIT 5", "flipped ORDER BY salary from ASC to DESC").
  * diff_plans(old_plan, new_plan): line diff of EXPLAIN QUERY PLAN output,
    showing which plan steps appeared or disappeared.
"""

from __future__ import annotations

import difflib

from .ir import QueryIR


def diff_ir(old: QueryIR, new: QueryIR) -> list[str]:
    out: list[str] = []

    if old.distinct != new.distinct:
        out.append("added DISTINCT" if new.distinct else "removed DISTINCT")

    old_sel = {s.key() for s in old.select}
    new_sel = {s.key() for s in new.select}
    for s in new.select:
        if s.key() not in old_sel:
            out.append(f"added output column {describe_select(s)}")
    for s in old.select:
        if s.key() not in new_sel:
            out.append(f"removed output column {describe_select(s)}")

    if old.table != new.table:
        out.append(f"changed base table from {old.table} to {new.table}")
    old_j = {j.key() for j in old.joins}
    for j in new.joins:
        if j.key() not in old_j:
            out.append(f"added {j.kind} JOIN {j.table} ON {j.left} = {j.right}")

    old_w = {c.key() for c in old.where}
    new_w = {c.key() for c in new.where}
    for c in new.where:
        if c.key() not in old_w:
            out.append(f"added filter {c.col} {c.op} {c.value!r}")
    for c in old.where:
        if c.key() not in new_w:
            out.append(f"removed filter {c.col} {c.op} {c.value!r}")

    if set(old.group_by) != set(new.group_by):
        added = [g for g in new.group_by if g not in old.group_by]
        removed = [g for g in old.group_by if g not in new.group_by]
        if added:
            out.append(f"added GROUP BY {', '.join(added)}")
        if removed:
            out.append(f"removed GROUP BY {', '.join(removed)}")

    old_o = dict(old.order_by)
    new_o = dict(new.order_by)
    for c, d in new.order_by:
        if c not in old_o:
            out.append(f"added ORDER BY {c} {d}")
        elif old_o[c] != d:
            out.append(f"flipped ORDER BY {c} from {old_o[c]} to {d}")
    for c in old_o:
        if c not in new_o:
            out.append(f"removed ORDER BY {c}")

    if old.limit != new.limit:
        if new.limit is None:
            out.append("removed LIMIT")
        else:
            out.append(f"set LIMIT {new.limit}")

    return out


def describe_select(s) -> str:
    if s.kind == "count_star":
        return "COUNT(*)"
    if s.kind == "agg":
        return f"{s.agg}({s.col})"
    return s.col or "?"


def diff_plans(old_plan: list[str], new_plan: list[str]) -> list[str]:
    """Line-level diff of two EXPLAIN QUERY PLAN outputs."""
    out = []
    for line in difflib.unified_diff(old_plan, new_plan, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            out.append("plan: now " + line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            out.append("plan: no longer " + line[1:].strip())
    return out


def explain_trace(question: str, steps) -> str:
    """Render a full human-readable narrative of a repair trace."""
    lines = [f"Question: {question}", ""]
    for st in steps:
        lines.append(f"--- Round {st.round} ---")
        lines.append(f"SQL: {st.sql}")
        if st.violations:
            lines.append("Violated assertions (counterexamples):")
            for v in st.violations:
                who = v.assertion.text if v.assertion else "execution"
                lines.append(f"  - {who}: {v.detail}")
                for r in v.failing_rows[:2]:
                    lines.append(f"      e.g. row {r}")
        else:
            lines.append("All assertions hold.")
        if st.diff:
            lines.append("What changed and why:")
            for d in st.diff:
                lines.append(f"  * {d}")
        if st.plan:
            lines.append(f"Plan: {'; '.join(st.plan[:3])}")
        lines.append("")
    return "\n".join(lines)
