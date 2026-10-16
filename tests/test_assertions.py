"""Tests for the assertion synthesizer and checkers."""

from assertsql.assertions import (
    Assertion, assertions_from_gold, synthesize)


def test_synthesize_top_n(schema):
    a = synthesize("Who are the top 5 earners?", schema)
    kinds = {x.kind: x for x in a}
    assert "card_max" in kinds
    assert kinds["card_max"].params["n"] == 5


def test_synthesize_superlative(schema):
    a = synthesize("Who is the youngest employee?", schema)
    kinds = {x.kind: x for x in a}
    assert kinds["card_exact"].params["n"] == 1
    assert "ordered" in kinds
    assert kinds["ordered"].params["direction"] == "ASC"


def test_synthesize_count(schema):
    a = synthesize("How many employees are there?", schema)
    assert any(x.kind == "single_value" for x in a)


def test_synthesize_per_grouping(schema):
    a = synthesize("For each department, show the average salary.", schema)
    uniq = [x for x in a if x.kind == "unique"]
    assert uniq, "expected a uniqueness assertion for 'for each department'"


def test_synthesize_no_hallucinated_order(schema):
    a = synthesize("List all employees.", schema)
    assert not [x for x in a if x.kind == "ordered"], \
        "must not invent an ordering assertion"


def test_card_max_check():
    a = Assertion("card_max", {"n": 2}, "t", "q")
    assert a.check([(1,), (2,), (3,)], ["x"]) is not None
    assert a.check([(1,), (2,)], ["x"]) is None


def test_unique_check():
    a = Assertion("unique", {"col": "d"}, "t", "q")
    assert a.check([(1,), (1,)], ["d"]) is not None
    assert a.check([(1,), (2,)], ["d"]) is None
    # unprojected column: cannot check, passes vacuously
    assert a.check([(1,)], ["other"]) is None


def test_ordered_check():
    a = Assertion("ordered", {"col": "s", "direction": "DESC"}, "t", "q")
    assert a.check([(1,), (3,), (2,)], ["s"]) is not None
    assert a.check([(3,), (2,), (1,)], ["s"]) is None


def test_single_value_check():
    a = Assertion("single_value", {}, "t", "q")
    assert a.check([(1,), (2,)], ["x"]) is not None
    assert a.check([(1,)], ["x"]) is None
    assert a.check([(1, 2)], ["x", "y"]) is not None


def test_gold_limit_and_order():
    g = assertions_from_gold(
        "SELECT name FROM employee ORDER BY salary DESC LIMIT 5")
    kinds = {x.kind: x for x in g}
    assert kinds["card_max"].params["n"] == 5
    assert kinds["ordered"].params["direction"] == "DESC"


def test_gold_scalar():
    g = assertions_from_gold("SELECT COUNT(*) FROM employee")
    assert any(x.kind == "single_value" for x in g)


def test_gold_group_by_unique():
    g = assertions_from_gold(
        "SELECT dept_id, AVG(salary) FROM employee GROUP BY dept_id")
    assert any(x.kind == "unique" for x in g)
    assert not any(x.kind == "single_value" for x in g)


def test_card_exact_implies_card_max_match():
    s = Assertion("card_exact", {"n": 1}, "t", "q")
    g = Assertion("card_max", {"n": 1}, "t", "gold")
    assert s.matches(g)
