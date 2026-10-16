"""Schema introspection from a SQLite database file (read-only)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


import re


def words(name: str) -> set[str]:
    """Split an identifier into lowercase words: snake_case and camelCase.

    Also keeps the joined form ('FName' -> 'fname') so 'first name' style
    synonyms match whole column names, not just fragments.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    out = set(re.split(r"[_\s]+", spaced.lower())) - {""}
    joined = re.sub(r"[^a-z0-9]", "", name.lower())
    if joined:
        out.add(joined)
    return out


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


# Phrase-level synonyms for schema linking (question phrasing -> schema words).
# Shared by the generator and the assertion synthesizer so they agree.
SYNONYMS = {
    "first name": "fname", "given name": "fname", "forename": "fname",
    "last name": "lname", "surname": "lname", "family name": "lname",
    "full name": "name",
    "earners": "salary", "earnings": "salary", "wages": "salary",
    "wage": "salary", "pay": "salary", "paid": "salary", "salaries": "salary",
    "salary": "salary", "income": "salary", "earn": "salary",
    "earned": "salary", "earning": "salary",
    "staff": "employee", "workers": "employee", "employees": "employee",
    "students": "student", "singers": "singer", "concerts": "concert",
    "dogs": "dog", "cats": "cat", "pets": "pet",
    "old": "age", "young": "age", "ages": "age",
    "expensive": "price", "cheap": "price", "prices": "price",
    "weigh": "weight", "weighs": "weight", "weighed": "weight",
    "taller": "height", "shorter": "height",
    "richer": "net_worth", "wealthier": "net_worth",
}


def expand_text(text: str) -> str:
    """Append synonym expansions so token matching sees schema words."""
    low = text.lower()
    extra = [v for k, v in SYNONYMS.items() if k in low]
    return text + " " + " ".join(extra) if extra else text


def link_tables(question: str, schema: "Schema") -> list[tuple[float, str]]:
    """Score tables by name-token overlap with the question (shared by the
    generator and the assertion synthesizer so they agree on the subject)."""
    qt = set(tokens(expand_text(question)))
    scored = []
    for tname, table in schema.tables.items():
        nt = words(tname)
        sing = {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in nt}
        overlap = max(len(nt & qt), len(sing & qt))
        score = overlap * 2.0
        for col in table.columns:
            if words(col.name) & qt:
                score += 0.5
        scored.append((score, tname))
    scored.sort(reverse=True)
    return scored


@dataclass
class Column:
    name: str
    type: str
    notnull: bool
    pk: bool
    table: str = ""
    samples: list[str] = field(default_factory=list)  # distinct text values

    @property
    def numeric(self) -> bool:
        t = (self.type or "").upper()
        return any(k in t for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC"))


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    row_count: int = 0

    def col(self, name: str) -> Column | None:
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None


@dataclass
class ForeignKey:
    from_table: str
    from_col: str
    to_table: str
    to_col: str


@dataclass
class Schema:
    tables: dict[str, Table]
    fks: list[ForeignKey]

    def all_columns(self) -> list[Column]:
        out = []
        for t in self.tables.values():
            out.extend(t.columns)
        return out

    def find_join(self, t1: str, t2: str) -> ForeignKey | None:
        for fk in self.fks:
            if {fk.from_table.lower(), fk.to_table.lower()} == {t1.lower(), t2.lower()}:
                return fk
        return None


def load_schema(db_path: str, sample_values: int = 3) -> Schema:
    """Load tables/columns/FKs from a SQLite file without writing to it."""
    uri = "file:" + db_path + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables: dict[str, Table] = {}
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        for (tname,) in cur.fetchall():
            cols: list[Column] = []
            for cid, name, ctype, notnull, dflt, pk in con.execute(
                    f'PRAGMA table_info("{tname}")'):
                cols.append(Column(name=name, type=ctype or "", notnull=bool(notnull),
                                   pk=bool(pk), table=tname))
            try:
                n = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
            except sqlite3.Error:
                n = 0
            tables[tname] = Table(name=tname, columns=cols, row_count=n)
        fks: list[ForeignKey] = []
        for tname in tables:
            for row in con.execute(f'PRAGMA foreign_key_list("{tname}")'):
                # id, seq, table, from, to, on_update, on_delete, match
                fks.append(ForeignKey(from_table=tname, from_col=row[3],
                                      to_table=row[2], to_col=row[4]))
        if sample_values:
            for tname, table in tables.items():
                for col in table.columns:
                    if col.numeric or col.pk:
                        continue
                    try:
                        rows = con.execute(
                            f'SELECT DISTINCT "{col.name}" FROM "{tname}" '
                            f'WHERE "{col.name}" IS NOT NULL LIMIT {sample_values}'
                        ).fetchall()
                        col.samples = [str(r[0]) for r in rows if r[0] != ""]
                    except sqlite3.Error:
                        pass
        return Schema(tables=tables, fks=fks)
    finally:
        con.close()
