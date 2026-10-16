"""assertsql demo: watch the agent catch its own wrong query.

Builds a tiny in-memory company database, asks a question, and shows the
full repair trace: synthesized assertions, violated assertions with
counterexample rows, each repair, and the plan diff.

Run:  python examples/demo.py ["your question here"]
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assertsql.plandiff import explain_trace
from assertsql.repair import repair
from assertsql.schema import load_schema


def build_demo_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE department (id INTEGER PRIMARY KEY, name TEXT, budget REAL)")
    con.execute("""CREATE TABLE employee (
        id INTEGER PRIMARY KEY, name TEXT, age INTEGER, salary REAL,
        dept_id INTEGER,
        FOREIGN KEY (dept_id) REFERENCES department(id))""")
    con.executemany("INSERT INTO department VALUES (?,?,?)", [
        (1, "eng", 200000.0), (2, "sales", 150000.0), (3, "hr", 90000.0)])
    con.executemany("INSERT INTO employee VALUES (?,?,?,?,?)", [
        (1, "amy", 30, 90000.0, 1), (2, "bo", 45, 120000.0, 1),
        (3, "cy", 28, 70000.0, 2), (4, "di", 52, 110000.0, 2),
        (5, "ed", 35, 95000.0, 3), (6, "fay", 41, 105000.0, 1),
        (7, "gus", 29, 88000.0, 2)])
    con.commit()
    con.close()


QUESTIONS = [
    "Who are the top 3 earners?",
    "Who is the youngest employee?",
    "How many employees are there?",
    "For each department, show the average salary.",
]


def main() -> None:
    questions = sys.argv[1:] or QUESTIONS
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "demo.sqlite")
        build_demo_db(db)
        schema = load_schema(db)
        for q in questions:
            trace = repair(q, schema, db)
            print("=" * 70)
            print(explain_trace(q, trace.steps))
            print(f"STATUS: {trace.status} after {trace.rounds_used} repair round(s)")
            print(f"FINAL SQL: {trace.final_sql}")
            print()


if __name__ == "__main__":
    main()
