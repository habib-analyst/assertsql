"""Counterexample-guided repair loop.

Each round: execute the candidate in the sandbox, check the synthesized
assertions against the materialized result, and turn every violated
assertion into a counterexample (assertion text + minimal failing rows).
One deterministic repair operator is applied per round, chosen by the
violated assertion's kind. Cap: 5 repair rounds after the initial attempt.

No LLM is consulted at any point: the "feedback" is data, not prose.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from .assertions import Assertion, Violation, synthesize, grouping_column
from . import generator as _genmod
from .ir import QueryIR, render
from .plandiff import diff_ir
from .sandbox import Sandbox, SandboxResult
from .schema import Schema

MAX_ROUNDS = 5

_PRIORITY = {"error": 0, "card_exact": 1, "card_max": 2, "single_value": 3,
             "ordered": 4, "unique": 5}


@dataclass
class RepairStep:
    round: int
    sql: str
    plan: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    repair_desc: str = ""        # what the repair operator did going INTO this round
    diff: list[str] = field(default_factory=list)  # IR diff vs previous round
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "sql": self.sql,
            "plan": self.plan,
            "violations": [
                {"assertion": (v.assertion.text if v.assertion else "execution"),
                 "kind": (v.assertion.kind if v.assertion else "error"),
                 "detail": v.detail,
                 "failing_rows": [list(r) for r in v.failing_rows]}
                for v in self.violations],
            "repair_desc": self.repair_desc,
            "diff": self.diff,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class RepairTrace:
    question: str
    assertions: list[Assertion]
    steps: list[RepairStep]
    final_sql: str
    status: str  # fixed | improved | unfixable | error
    rounds_used: int
    n_violations_start: int
    n_violations_end: int

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "assertions": [a.text for a in self.assertions],
            "steps": [s.to_dict() for s in self.steps],
            "final_sql": self.final_sql,
            "status": self.status,
            "rounds_used": self.rounds_used,
            "n_violations_start": self.n_violations_start,
            "n_violations_end": self.n_violations_end,
        }


def _closest_column(bad: str, schema: Schema) -> Optional[str]:
    names = [f"{t}.{c.name}" for t, tab in schema.tables.items()
             for c in tab.columns]
    bare = [c.name for tab in schema.tables.values() for c in tab.columns]
    hit = difflib.get_close_matches(bad, bare, n=1, cutoff=0.6)
    if hit:
        for full in names:
            if full.split(".")[-1].lower() == hit[0].lower():
                return full
    hit = difflib.get_close_matches(bad, names, n=1, cutoff=0.6)
    return hit[0] if hit else None


def _rename_column(ir: QueryIR, bad: str, good: str) -> bool:
    """Replace a bad column reference everywhere in the IR."""
    bad_bare = bad.split(".")[-1].strip('"').lower()
    changed = False

    def fix(ref: str) -> str:
        nonlocal changed
        if ref.split(".")[-1].strip('"').lower() == bad_bare:
            changed = True
            return good
        return ref

    for s in ir.select:
        if s.col:
            s.col = fix(s.col)
    for c in ir.where + ir.having:
        c.col = fix(c.col)
    ir.group_by = [fix(g) for g in ir.group_by]
    ir.order_by = [(fix(c), d) for c, d in ir.order_by]
    return changed


def _repair_op(ir: QueryIR, violation: Violation, schema: Schema,
               err: str = "") -> tuple[bool, str]:
    """Apply one deterministic repair. Returns (changed, description)."""
    kind = violation.assertion.kind if violation.assertion else "error"

    if kind == "error":
        m = re.search(r"no such column: ([\w.\"`]+)", err, re.IGNORECASE)
        if m:
            bad = m.group(1)
            good = _closest_column(bad, schema)
            if good and _rename_column(ir, bad, good):
                return True, (f"replaced unknown column '{bad}' with "
                              f"closest schema match '{good}'")
            return False, f"no schema match for unknown column '{bad}'"
        m = re.search(r"no such table: ([\w.\"`]+)", err, re.IGNORECASE)
        if m:
            return False, f"unknown table '{m.group(1)}' (cannot invent tables)"
        return False, f"unrepairable execution error: {err[:120]}"

    a = violation.assertion
    n_rows_hint = len(violation.failing_rows)

    if kind in ("card_max", "card_exact"):
        n = a.params["n"]
        m = re.search(r"got (\d+)", violation.detail)
        got = int(m.group(1)) if m else n + 1
        if got > n:
            # too many rows: constrain the result
            if ir.limit is None:
                ir.limit = n
                return True, (f"added LIMIT {n} ({violation.detail}; "
                              f"counterexample rows: {n_rows_hint} shown)")
            if not ir.distinct and not any(s.kind == "agg" for s in ir.select):
                ir.distinct = True
                return True, (f"added DISTINCT ({violation.detail}; duplicates "
                              f"in counterexample rows)")
            return False, "too many rows and no constraint left to tune"
        else:
            # too few rows: relax the query
            if ir.where:
                dropped = ir.where.pop()
                return True, (f"removed WHERE {dropped.col} {dropped.op} "
                              f"({violation.detail})")
            if ir.limit is not None:
                ir.limit = None
                return True, f"removed LIMIT ({violation.detail})"
            return False, "too few rows and nothing left to relax"

    if kind == "single_value":
        if ir.limit is None:
            ir.limit = 1
            return True, f"added LIMIT 1 ({violation.detail})"
        if ir.where:
            dropped = ir.where.pop()
            return True, (f"removed WHERE {dropped.col} {dropped.op} "
                          f"(scalar query returned no rows)")
        return False, "scalar shape violated and nothing left to repair"

    if kind == "ordered":
        col, direction = a.params["col"], a.params["direction"]
        ir.order_by = [(col, direction)]
        return True, (f"set ORDER BY {col} {direction} "
                      f"({violation.detail})")

    if kind == "unique":
        col = a.params["col"]
        if any(s.kind == "agg" for s in ir.select):
            if col not in ir.group_by:
                ir.group_by.append(col)
                return True, (f"added GROUP BY {col} ({violation.detail})"
                               )
            return False, "GROUP BY already present but duplicates remain"
        if not ir.distinct:
            ir.distinct = True
            return True, f"added DISTINCT ({violation.detail})"
        return False, "DISTINCT already present but duplicates remain"

    return False, f"no repair operator for {kind}"


def _check_all(assertions: list[Assertion], res: SandboxResult) -> list[Violation]:
    out = []
    for a in assertions:
        v = a.check(res.rows, res.colnames)
        if v:
            out.append(v)
    out.sort(key=lambda v: _PRIORITY.get(v.assertion.kind, 9))
    return out


def repair(question: str, schema: Schema, db_path: str,
           max_rounds: int = MAX_ROUNDS,
           assertions: Optional[list[Assertion]] = None,
           evidence: str = "") -> RepairTrace:
    """Run the counterexample-guided repair loop. Returns the full trace."""
    box = Sandbox(db_path)
    if assertions is None:
        assertions = synthesize(question, schema)

    ir = _genmod.generate(question, schema, evidence=evidence)
    steps: list[RepairStep] = []
    prev_ir: Optional[QueryIR] = None
    prev_sql = ""
    best_idx = 0
    best_violations = 10 ** 9
    status = "unfixable"

    for rnd in range(max_rounds + 1):
        sql = render(ir)
        res = box.run(sql)
        violations: list[Violation] = []
        err = ""
        if not res.ok:
            err = res.error
            violations = [Violation(
                Assertion("error", {}, "query executes without error", question),
                f"execution failed: {err}", [])]
        else:
            violations = _check_all(assertions, res)

        step = RepairStep(round=rnd, sql=sql, plan=res.plan,
                          violations=violations, elapsed_ms=res.elapsed_ms)
        if prev_ir is not None:
            step.diff = diff_ir(prev_ir, ir)
        steps.append(step)

        if len(violations) < best_violations:
            best_violations = len(violations)
            best_idx = rnd

        if not violations:
            status = "fixed" if rnd > 0 else "fixed"
            break

        if rnd == max_rounds:
            break

        target = violations[0]
        new_ir = ir.clone()
        changed, desc = _repair_op(new_ir, target, schema, err=err)
        step.repair_desc = ("next: " + desc) if changed else ("stuck: " + desc)
        if not changed or new_ir.fingerprint() == ir.fingerprint():
            break
        prev_ir, prev_sql = ir, sql
        # attach the "why" to the upcoming round's diff narrative
        ir = new_ir

    # annotate each step's diff with the motivating counterexample
    for i in range(1, len(steps)):
        why = "; ".join(v.assertion.text for v in steps[i - 1].violations[:2])
        if why and steps[i].diff:
            steps[i].diff = [f"because: {why}"] + steps[i].diff

    final = steps[best_idx]
    if best_violations == 0:
        status = "fixed"
    elif best_violations < len(steps[0].violations):
        status = "improved"
    else:
        status = "unfixable"

    return RepairTrace(
        question=question,
        assertions=assertions,
        steps=steps,
        final_sql=final.sql,
        status=status,
        rounds_used=len(steps) - 1,
        n_violations_start=len(steps[0].violations),
        n_violations_end=best_violations,
    )
