"""QueryIR: a small, structured representation of SELECT queries.

The repair loop edits this IR (never raw SQL strings), so every repair step
is a well-defined structural change that the plan-diff explainer can narrate.
Only the SELECT/WITH-free single-SELECT subset needed for text-to-SQL repair
is modeled: no subqueries, no UNION, no CTEs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _qident(name: str) -> str:
    # Quote identifiers defensively; pass through dotted names part by part.
    parts = name.split(".")
    out = []
    for p in parts:
        p = p.strip().strip('"')
        if p == "*":
            out.append("*")
        else:
            out.append('"' + p.replace('"', '""') + '"')
    return ".".join(out)


def _qlit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


@dataclass
class SelectItem:
    """One output column: a plain column, an aggregate, or COUNT(*)."""
    kind: str  # "col" | "agg" | "count_star"
    col: Optional[str] = None
    agg: Optional[str] = None  # COUNT | SUM | AVG | MIN | MAX
    distinct: bool = False

    def key(self) -> str:
        return f"{self.kind}:{self.agg}:{self.col}:{self.distinct}"


@dataclass
class Condition:
    """One ANDed predicate. op in =, !=, <, <=, >, >=, LIKE, IN, BETWEEN,
    IS NULL, IS NOT NULL, NOT IN (with subquery)."""
    col: str
    op: str
    value: Any = None
    subquery: Optional["QueryIR"] = None

    def key(self) -> str:
        sub = self.subquery.fingerprint() if self.subquery else ""
        return f"{self.col}{self.op}{self.value!r}{sub}"


@dataclass
class Join:
    table: str
    left: str   # "table.col" on the already-present side
    right: str  # "table.col" on the new table
    kind: str = "INNER"

    def key(self) -> str:
        return f"{self.kind}:{self.table}:{self.left}={self.right}"


@dataclass
class QueryIR:
    select: list[SelectItem] = field(default_factory=list)
    table: str = ""
    joins: list[Join] = field(default_factory=list)
    where: list[Condition] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    having: list[Condition] = field(default_factory=list)
    order_by: list[tuple[str, str]] = field(default_factory=list)  # (col, ASC|DESC)
    limit: Optional[int] = None
    distinct: bool = False

    def fingerprint(self) -> str:
        """Canonical string for change detection between repair rounds."""
        parts = [
            "D" if self.distinct else "",
            ",".join(s.key() for s in self.select),
            self.table,
            ",".join(j.key() for j in self.joins),
            ",".join(c.key() for c in self.where),
            ",".join(self.group_by),
            ",".join(c.key() for c in self.having),
            ",".join(f"{c}:{d}" for c, d in self.order_by),
            str(self.limit),
        ]
        return "|".join(parts)

    def clone(self) -> "QueryIR":
        import copy
        return copy.deepcopy(self)


def _render_cond(c: Condition) -> str:
    col = _qident(c.col)
    op = c.op.upper()
    if op in ("IS NULL", "IS NOT NULL"):
        return f"{col} {op}"
    if op == "NOT IN" and c.subquery is not None:
        return f"{col} NOT IN ({render(c.subquery)})"
    if op in ("IN", "NOT IN"):
        vals = ", ".join(_qlit(v) for v in c.value)
        return f"{col} {op} ({vals})"
    if op == "BETWEEN":
        lo, hi = c.value
        return f"{col} BETWEEN {_qlit(lo)} AND {_qlit(hi)}"
    return f"{col} {op} {_qlit(c.value)}"


def render(ir: QueryIR) -> str:
    """Render the IR to a single SQLite SELECT statement."""
    if not ir.select or not ir.table:
        raise ValueError("QueryIR needs at least one select item and a table")
    sel = []
    for s in ir.select:
        if s.kind == "count_star":
            sel.append("COUNT(*)")
        elif s.kind == "agg":
            d = "DISTINCT " if s.distinct else ""
            sel.append(f"{s.agg}({d}{_qident(s.col)})")
        else:
            d = "DISTINCT " if s.distinct else ""
            sel.append(f"{d}{_qident(s.col)}")
    distinct_kw = "DISTINCT " if ir.distinct else ""
    sql = f"SELECT {distinct_kw}{', '.join(sel)} FROM {_qident(ir.table)}"
    for j in ir.joins:
        sql += (f" {j.kind} JOIN {_qident(j.table)} ON "
                f"{_qident(j.left)} = {_qident(j.right)}")
    if ir.where:
        sql += " WHERE " + " AND ".join(_render_cond(c) for c in ir.where)
    if ir.group_by:
        sql += " GROUP BY " + ", ".join(_qident(g) for g in ir.group_by)
    if ir.having:
        sql += " HAVING " + " AND ".join(_render_cond(c) for c in ir.having)
    if ir.order_by:
        sql += " ORDER BY " + ", ".join(f"{_qident(c)} {d}" for c, d in ir.order_by)
    if ir.limit is not None:
        sql += f" LIMIT {int(ir.limit)}"
    return sql
