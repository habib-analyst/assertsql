"""Tests for the sandbox, generator, and counterexample-guided repair loop."""

from assertsql import assertions as A
from assertsql.generator import generate
from assertsql.ir import render
from assertsql.plandiff import diff_ir, explain_trace
from assertsql.repair import repair
from assertsql.sandbox import Sandbox


def test_sandbox_readonly(db_path):
    box = Sandbox(db_path)
    r = box.run("DELETE FROM employee")
    assert not r.ok and "refusing" in r.error


def test_sandbox_runs_select(db_path):
    box = Sandbox(db_path)
    r = box.run("SELECT COUNT(*) FROM employee")
    assert r.ok and r.rows == [(6,)]
    assert r.plan, "expected EXPLAIN QUERY PLAN lines"


def test_sandbox_error_surfaced(db_path):
    box = Sandbox(db_path)
    r = box.run("SELECT nope FROM employee")
    assert not r.ok and "no such column" in r.error


def test_generator_top_n(schema):
    ir = generate("Who are the top 2 earners?", schema)
    sql = render(ir)
    assert "LIMIT 2" in sql and "DESC" in sql


def test_generator_count(schema):
    ir = generate("How many employees are there?", schema)
    assert "COUNT(*)" in render(ir)


def test_repair_fixes_missing_limit(db_path, schema):
    # craft an IR that forgets the LIMIT, then repair must add it
    from assertsql.ir import QueryIR, SelectItem
    ir = QueryIR(
        select=[SelectItem(kind="col", col="employee.name")],
        table="employee",
        order_by=[("employee.salary", "DESC")])
    import assertsql.generator as G
    orig_gen = G.generate
    G.generate = lambda q, s, evidence="": ir
    try:
        trace = repair("Who are the top 2 earners?", schema, db_path)
    finally:
        G.generate = orig_gen
    assert trace.status == "fixed", trace.steps[-1].sql
    assert "LIMIT 2" in trace.final_sql
    assert trace.rounds_used >= 1


def test_repair_fixes_bad_column(db_path, schema):
    from assertsql.ir import QueryIR, SelectItem
    ir = QueryIR(select=[SelectItem(kind="col", col="employee.salar")],
                 table="employee")
    import assertsql.generator as G
    orig_gen = G.generate
    G.generate = lambda q, s, evidence="": ir
    try:
        trace = repair("List employee salaries.", schema, db_path,
                       assertions=[])
    finally:
        G.generate = orig_gen
    assert trace.status == "fixed"
    assert "salary" in trace.final_sql


def test_repair_loop_cap(db_path, schema):
    # unfixable: assertion contradicts the data (at most 0 rows, table nonempty)
    a = [A.Assertion("card_max", {"n": 0}, "at most 0", "q")]
    trace = repair("List employees.", schema, db_path, assertions=a,
                   max_rounds=5)
    assert trace.rounds_used <= 5
    assert trace.status in ("unfixable", "improved", "fixed")


def test_diff_narrates_change(db_path, schema):
    from assertsql.ir import QueryIR, SelectItem
    ir = QueryIR(
        select=[SelectItem(kind="col", col="employee.name")],
        table="employee",
        order_by=[("employee.salary", "DESC")])
    import assertsql.generator as G
    orig_gen = G.generate
    G.generate = lambda q, s, evidence="": ir
    try:
        trace = repair("Who are the top 2 earners?", schema, db_path)
    finally:
        G.generate = orig_gen
    text = explain_trace(trace.question, trace.steps)
    assert "LIMIT 2" in text
    assert "Round 1" in text


def test_diff_ir_sentences():
    from assertsql.ir import QueryIR, SelectItem
    a = QueryIR(select=[SelectItem(kind="col", col="t.x")], table="t")
    b = a.clone()
    b.limit = 5
    b.order_by = [("t.x", "DESC")]
    d = diff_ir(a, b)
    assert any("LIMIT 5" in s for s in d)
    assert any("ORDER BY t.x DESC" in s for s in d)
