"""Evaluation: execution accuracy and the three benchmark arms.

Arms (same candidate generator throughout; only the loop differs):
  * single-shot : generator output, executed once, no assertions.
  * assertsql   : generator + counterexample-guided repair (<=5 rounds).
  * oracle      : generator + repair driven by assertions extracted from the
                  gold query (upper bound for the assertion machinery).

Execution accuracy: candidate result vs gold result as multisets of rows
(order-insensitive unless the gold query has ORDER BY), with NULL-safe
comparison and float rounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .assertions import Assertion, assertions_from_gold, synthesize
from .generator import generate
from .ir import render
from .repair import repair, RepairTrace
from .sandbox import Sandbox
from .schema import Schema, load_schema

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _norm(v):
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def _norm_rows(rows: list[tuple]) -> list[tuple]:
    return sorted(tuple(_norm(v) for v in r) for r in rows)


def execution_match(pred_sql: str, gold_sql: str, db_path: str,
                    timeout_s: float = 10.0) -> tuple[bool, str]:
    """Compare predicted vs gold SQL by executing both. Returns (match, note)."""
    box = Sandbox(db_path, timeout_s=timeout_s)
    gold = box.run(gold_sql)
    if not gold.ok:
        return False, f"gold failed: {gold.error[:80]}"
    pred = box.run(pred_sql)
    if not pred.ok:
        return False, f"pred failed: {pred.error[:80]}"
    if len(gold.colnames) != len(pred.colnames):
        return False, "column count differs"
    g_rows = _norm_rows(gold.rows)
    p_rows = _norm_rows(pred.rows)
    if _ORDER_BY_RE.search(gold_sql or ""):
        # order-sensitive compare when gold orders
        g_ord = [tuple(_norm(v) for v in r) for r in gold.rows]
        p_ord = [tuple(_norm(v) for v in r) for r in pred.rows]
        return (g_ord == p_ord, "order-sensitive" if g_ord == p_ord else "order differs")
    return (g_rows == p_rows, "match" if g_rows == p_rows else "rows differ")


@dataclass
class BenchItem:
    qid: str
    question: str
    db_id: str
    db_path: str
    gold_sql: str
    evidence: str = ""


@dataclass
class BenchResult:
    qid: str
    arm: str
    correct: bool
    note: str = ""
    trace: RepairTrace | None = None


def run_arm(items: list[BenchItem], arm: str,
            on_item=None) -> list[BenchResult]:
    """Run one arm over all items. Arms: single | assertsql | oracle."""
    results: list[BenchResult] = []
    schema_cache: dict[str, Schema] = {}
    for it in items:
        if it.db_path not in schema_cache:
            schema_cache[it.db_path] = load_schema(it.db_path)
        schema = schema_cache[it.db_path]
        try:
            if arm == "single":
                ir = generate(it.question, schema, evidence=it.evidence)
                sql = render(ir)
                ok, note = execution_match(sql, it.gold_sql, it.db_path)
                results.append(BenchResult(it.qid, arm, ok, note))
            elif arm in ("assertsql", "oracle"):
                assertions = (assertions_from_gold(it.gold_sql)
                              if arm == "oracle" else None)
                trace = repair(it.question, schema, it.db_path,
                               assertions=assertions, evidence=it.evidence)
                ok, note = execution_match(trace.final_sql, it.gold_sql,
                                           it.db_path)
                results.append(BenchResult(it.qid, arm, ok, note, trace))
            else:
                raise ValueError(f"unknown arm {arm}")
        except Exception as e:  # never let one item kill the benchmark
            results.append(BenchResult(it.qid, arm, False,
                                       f"harness error: {type(e).__name__}: {e}"))
        if on_item:
            on_item(it, results[-1])
    return results


def accuracy(results: list[BenchResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def assertion_quality(items: list[BenchItem]) -> dict:
    """E2: precision/recall of synthesized assertions vs gold-derived ones."""
    schema_cache: dict[str, Schema] = {}
    tp = fp = fn = 0
    per_kind: dict[str, list[int]] = {}
    for it in items:
        if it.db_path not in schema_cache:
            schema_cache[it.db_path] = load_schema(it.db_path)
        schema = schema_cache[it.db_path]
        synth = synthesize(it.question, schema)
        gold = assertions_from_gold(it.gold_sql)
        matched_gold = set()
        for s in synth:
            hit = None
            for i, g in enumerate(gold):
                if i not in matched_gold and s.matches(g):
                    hit = i
                    break
            if hit is not None:
                matched_gold.add(hit)
                tp += 1
            else:
                fp += 1
            per_kind.setdefault(s.kind, [0, 0])
            per_kind[s.kind][0 if hit is not None else 1] += 1
        fn += len(gold) - len(matched_gold)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
            "per_kind_tp_fp": per_kind}
