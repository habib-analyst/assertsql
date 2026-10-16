"""Sandbox: read-only, time-boxed SQLite execution of candidate queries."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field


_READONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE | re.DOTALL)
_MAX_ROWS = 5000


@dataclass
class SandboxResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    colnames: list[str] = field(default_factory=list)
    error: str = ""
    elapsed_ms: float = 0.0
    plan: list[str] = field(default_factory=list)  # EXPLAIN QUERY PLAN lines


class Sandbox:
    """Executes one SELECT at a time against a database file, read-only."""

    def __init__(self, db_path: str, timeout_s: float = 10.0):
        self.db_path = db_path
        self.timeout_s = timeout_s

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect("file:" + self.db_path + "?mode=ro", uri=True)
        con.execute("PRAGMA query_only = ON")
        deadline = time.time() + self.timeout_s

        def _guard():
            if time.time() > deadline:
                raise TimeoutError("query timed out")
            return 0

        con.set_progress_handler(_guard, 10000)
        return con

    def run(self, sql: str) -> SandboxResult:
        if not _READONLY_RE.match(sql or ""):
            return SandboxResult(ok=False, error="refusing non-SELECT statement")
        t0 = time.time()
        con = self._connect()
        try:
            plan_rows = con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            plan = [" ".join(str(x) for x in r[2:]) for r in plan_rows]
            cur = con.execute(sql)
            colnames = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(_MAX_ROWS)
            return SandboxResult(ok=True, rows=rows, colnames=colnames,
                                 elapsed_ms=(time.time() - t0) * 1000.0,
                                 plan=plan)
        except TimeoutError as e:
            return SandboxResult(ok=False, error=str(e),
                                 elapsed_ms=(time.time() - t0) * 1000.0)
        except sqlite3.Error as e:
            return SandboxResult(ok=False,
                                 error=f"{type(e).__name__}: {e}",
                                 elapsed_ms=(time.time() - t0) * 1000.0)
        finally:
            con.close()
