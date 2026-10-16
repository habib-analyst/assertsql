# assertsql

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![No LLM](https://img.shields.io/badge/LLM-none-green.svg)]()

**Test-driven development for SQL.** assertsql synthesizes checkable semantic
assertions from your English question ("top 5 earners" → *at most 5 rows*),
runs the candidate query in a sandbox, and repairs it using violated assertions
as counterexamples. No LLM judge, no human — just the database.

## Quickstart

```bash
pip install assertsql
```

```python
from assertsql import load_schema, repair
from assertsql.plandiff import explain_trace

schema = load_schema("company.sqlite")
trace = repair("Who are the top five earners?", schema, "company.sqlite")
print(explain_trace(trace.question, trace.steps))
print("FINAL:", trace.final_sql, "| status:", trace.status)
```

Real output (the generator missed the word-number "five"; the assertion caught it):

```
--- Round 0 ---
SQL: SELECT "employee"."salary" FROM "employee"
Violated assertions (counterexamples):
  - result has at most 5 row(s): expected at most 5 row(s), got 7

--- Round 1 ---
SQL: SELECT "employee"."salary" FROM "employee" LIMIT 5
All assertions hold.
What changed and why:
  * because: result has at most 5 row(s)
  * set LIMIT 5

FINAL: SELECT "employee"."salary" FROM "employee" LIMIT 5 | status: fixed
```

Or run the self-contained demo (no download needed):

```bash
python examples/demo.py "Who are the top 3 earners?"
```

## Measured results

196-question Spider dev subset (4 databases), CPU-only, fully deterministic:

| Arm | Execution accuracy |
|---|---|
| Single-shot (rule-based generator, no repair) | **18.4%** (36/196) |
| **assertsql** (same generator + ≤5 assertion-guided repair rounds) | **20.4%** (40/196) |
| Oracle (repair driven by assertions extracted from gold SQL) | 21.4% (42/196) |

Assertion quality vs. gold-derived assertions (196 questions):

| Metric | Value |
|---|---|
| Precision | **0.911** (112 tp / 11 fp) |
| Recall | 0.675 (112 tp / 54 fn) |

Repair efficiency (13 questions where repair engaged):

| Metric | Value |
|---|---|
| Median repair rounds | 1 |
| Fixed | 12 / 13 |
| Honest unfixable rate | **7.7%** (1 / 13) |

BIRD mini-dev (500 questions): 1.8% single-shot, 1.8% assertsql — the rule-based
generator is out of its depth on BIRD's schemas, and assertion repair cannot fix
semantic (wrong-table) errors. Reported as measured.

**How to read the headline number:** the oracle upper bound (+3.0pp over
single-shot) shows assertion-guided repair is *not* the bottleneck on this
subset — the rule-based generator's semantic errors (wrong tables/columns) are.
The repair loop fixes what it can prove wrong: shape errors. See `PAPER.md`
for the full analysis.

## How it works

```
question + schema
      │
      ▼
┌─────────────┐   ┌──────────────┐
│ synthesizer │──▶│  assertions  │  "at most 5 rows", "exactly 1 row",
└─────────────┘   │  (SQL-checked)│  "no duplicate dept", "ordered by salary DESC"
                  └──────┬───────┘
                         │ violated? ── counterexample rows
┌─────────────┐          ▼
│  generator  │──▶ candidate SQL ──▶ sandbox (read-only SQLite)
└─────────────┘          │                    │
                         │◀── repair op ◀─────┘  (max 5 rounds)
                         ▼
                  plan-diff explainer: EXPLAIN diff rendered as
                  "what changed and why"
```

1. **Assertion synthesizer** (`assertsql/assertions.py`): rule-based,
   from (question, schema). Five assertion kinds: `card_exact`, `card_max`,
   `unique`, `ordered`, `single_value`. Conservative by design (precision 0.91).
2. **Candidate generator** (`assertsql/generator.py`): rule-based text-to-SQL
   (schema linking with synonym expansion, sample-value WHERE matching,
   FK-graph join planning, NOT IN subqueries). The single-shot baseline.
3. **Repair loop** (`assertsql/repair.py`): execute → check assertions →
   one deterministic repair op per violated assertion (add LIMIT/DISTINCT/
   ORDER BY/GROUP BY, relax WHERE, remap bad columns) → repeat, cap 5 rounds.
4. **Plan-diff explainer** (`assertsql/plandiff.py`): structural IR diff +
   EXPLAIN QUERY PLAN line diff → human-readable narrative.

No LLM anywhere: not in generation, not in checking, not in repair.

## Reproduce the benchmarks

```bash
# download Spider dev + BIRD mini-dev (see hidden_files/assertsql_data/MANIFEST.json)
python bench/run_bench.py --all
```

## Citation

```bibtex
@software{rehman2026assertsql,
  author  = {Habib Ur Rehman},
  title   = {assertsql: counterexample-guided text-to-SQL repair with synthesized semantic assertions},
  year    = {2026},
  url     = {https://github.com/habib-analyst/assertsql}
}
```

## License

MIT — see [LICENSE](LICENSE).

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
