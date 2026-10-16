"""Semantic assertions: SQL-checkable properties of a query result.

Two producers:
  * synthesize(question, schema) -- rule-based derivation from the natural
    language question. These are the assertions the repair loop uses.
  * assertions_from_gold(gold_sql) -- structural assertions extracted from a
    gold query. Used for the oracle upper bound and as ground truth for the
    assertion-quality experiment (E2).

Every assertion is evaluated against the *materialized* candidate result, so
checking is pure Python/SQL over data -- no LLM judge anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import Schema, link_tables, tokens, words


# --------------------------------------------------------------------------
# Assertion model
# --------------------------------------------------------------------------

@dataclass
class Violation:
    assertion: "Assertion"
    detail: str
    failing_rows: list[tuple] = field(default_factory=list)


@dataclass
class Assertion:
    kind: str  # card_exact | card_max | unique | ordered | single_value
    params: dict[str, Any]
    text: str            # human-readable statement of the property
    source_phrase: str    # question fragment that produced it (or "gold")

    def key(self) -> str:
        return f"{self.kind}:{sorted(self.params.items())}"

    # -- evaluation -----------------------------------------------------
    def check(self, rows: list[tuple], colnames: list[str]) -> Violation | None:
        """Return a Violation if the assertion fails on the result, else None."""
        if self.kind == "card_exact":
            n = self.params["n"]
            if len(rows) != n:
                return Violation(self, f"expected exactly {n} row(s), got {len(rows)}",
                                 failing_rows=rows[:3])
        elif self.kind == "card_max":
            n = self.params["n"]
            if len(rows) > n:
                return Violation(
                    self, f"expected at most {n} row(s), got {len(rows)}",
                    failing_rows=rows[:3])
        elif self.kind == "unique":
            col = self.params["col"]
            idx = _col_index(colnames, col)
            if idx is None:
                return None  # column not projected; cannot check
            seen, dupes = set(), []
            for r in rows:
                v = r[idx]
                if v is None:
                    continue
                if v in seen and len(dupes) < 3:
                    dupes.append(r)
                seen.add(v)
            if dupes:
                non_null = sum(1 for r in rows if r[idx] is not None)
                return Violation(
                    self,
                    f"column '{col}' has {non_null - len(seen)} duplicate "
                    f"value(s) across {non_null} non-null rows",
                    failing_rows=dupes)
        elif self.kind == "ordered":
            col = self.params["col"]
            direction = self.params["direction"]
            idx = _col_index(colnames, col)
            if idx is None or len(rows) < 2:
                return None
            vals = [r[idx] for r in rows]
            pairs = [(a, b) for a, b in zip(vals, vals[1:])
                     if a is not None and b is not None]
            if direction == "DESC":
                bad = sum(1 for a, b in pairs if b > a)
            else:
                bad = sum(1 for a, b in pairs if b < a)
            if bad:
                return Violation(
                    self, f"column '{col}' is not sorted {direction} "
                           f"({bad} inversion(s))",
                    failing_rows=rows[:3])
        elif self.kind == "single_value":
            if not (len(rows) == 1 and len(colnames) == 1):
                return Violation(
                    self, f"expected a single scalar value, got "
                           f"{len(rows)} row(s) x {len(colnames)} col(s)",
                    failing_rows=rows[:3])
        else:
            raise ValueError(f"unknown assertion kind {self.kind!r}")
        return None

    def matches(self, other: "Assertion") -> bool:
        """Loose match used for E2 scoring.

        card_exact(n) implies card_max(n); single_value (1x1) implies
        card_exact(1).
        """
        if self.kind == other.kind:
            return _params_equal(self.kind, self.params, other.params)
        if {self.kind, other.kind} == {"card_exact", "card_max"}:
            return self.params.get("n") == other.params.get("n")
        if (self.kind, other.kind) == ("card_exact", "single_value"):
            return self.params.get("n") == 1
        return False


def _col_index(colnames: list[str], col: str) -> Optional[int]:
    want = col.lower().split(".")[-1].strip('"')
    for i, c in enumerate(colnames):
        if c.lower().split(".")[-1].strip('"') == want:
            return i
    return None


def _params_equal(kind: str, a: dict, b: dict) -> bool:
    if kind in ("card_exact", "card_max"):
        return a.get("n") == b.get("n")
    if kind == "unique":
        return (a.get("col", "").lower().split(".")[-1]
                == b.get("col", "").lower().split(".")[-1])
    if kind == "ordered":
        same_col = (a.get("col", "").lower().split(".")[-1]
                    == b.get("col", "").lower().split(".")[-1])
        return same_col and a.get("direction") == b.get("direction")
    if kind == "single_value":
        return True
    return False


# --------------------------------------------------------------------------
# Synthesizer: question -> assertions (rule-based, no LLM)
# --------------------------------------------------------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
}


def _num_word(s: str) -> Optional[int]:
    s = s.lower()
    if s.isdigit():
        return int(s)
    return _WORD_NUM.get(s)


# adjective -> (column-name hints, direction); the noun following the
# adjective is prepended to hints at synthesis time ("greatest height")
_SUPERLATIVES = [
    (r"\byoungest\b", ["age", "birth"], "ASC"),
    (r"\boldest\b", ["age", "birth"], "DESC"),
    (r"\bgreatest\b", ["height", "size", "amount", "count", "number"], "DESC"),
    (r"(?<!at\s)\bleast\b", ["height", "size", "amount", "count", "number"], "ASC"),
    (r"\bhighest\b", ["salary", "wage", "pay", "price", "amount", "score", "grade", "rating", "capacity", "population"], "DESC"),
    (r"\blowest\b", ["salary", "wage", "pay", "price", "amount", "score", "grade", "rating", "capacity", "population"], "ASC"),
    (r"\blargest\b", ["population", "area", "size", "capacity", "amount", "salary"], "DESC"),
    (r"\bsmallest\b", ["population", "area", "size", "capacity", "amount", "salary"], "ASC"),
    (r"\bmost expensive\b", ["price", "cost", "fee", "salary"], "DESC"),
    (r"\bcheapest\b", ["price", "cost", "fee"], "ASC"),
    (r"(?<!at\s)\bmost\b", ["count", "number", "score"], "DESC"),
    (r"\bmaximum\b", ["salary", "price", "amount", "score"], "DESC"),
    (r"\bminimum\b", ["salary", "price", "amount", "score"], "ASC"),
    (r"\bearliest\b", ["date", "year", "time"], "ASC"),
    (r"\blatest\b", ["date", "year", "time"], "DESC"),
    (r"\btallest\b", ["height"], "DESC"),
    (r"\bshortest\b", ["height"], "ASC"),
]

_AGG_WORDS = re.compile(
    r"\b(how many|count of|count the|total number of)\b|"
    r"(?<!most\s)(?<!least\s)(?<!greatest\s)\bnumber of\b")
_SCALAR_WORDS = re.compile(
    r"\b(average|avg|mean|total|sum of|sum the)\b")


def _resolve_column(hints: list[str], question: str, schema: Schema,
                    prefer_table: str | None = None,
                    strict: bool = False) -> Optional[str]:
    """Pick the schema column best matching the hints and question text.

    Conservative: only returns a column when there is a clear textual match.
    Ties are broken toward the question's primary table; a residual tie
    returns None. With strict=True, a hint itself (not just stray question
    words) must match the column name. Returns 'table.col' qualified name.
    """
    q = question.lower()
    scored: list[tuple[int, int, bool, str]] = []  # (hint, total, preferred, qname)
    # singularize hints ("names" -> "name") so plural phrasing matches columns
    raw_hints = []
    for h in hints:
        raw_hints.append(h)
        if h.endswith("s") and len(h) > 3 and not h.endswith("ss"):
            raw_hints.append(h[:-1])
    for col in schema.all_columns():
        cname = col.name.lower()
        hint_score = 0
        for h in raw_hints:
            if h == cname or h in cname or cname in h:
                hint_score += 3
        total = hint_score
        # bonus if the column name (or its tokens) appears in the question
        for tok in words(col.name):
            if len(tok) > 2 and tok in q:
                total += 2
        if total:
            if strict and hint_score < 3:
                continue
            preferred = (prefer_table is not None
                         and col.table.lower() == prefer_table.lower())
            scored.append((hint_score, total, preferred, f"{col.table}.{col.name}"))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    key = scored[0][:3]
    contenders = [s for s in scored if s[:3] == key]
    if len(contenders) > 1:
        return None
    return contenders[0][3]


def _is_table_word(noun: str, schema: Schema) -> bool:
    """Is this plural noun the name of a table (e.g. 'documents')?"""
    sing = noun[:-1]
    for tname in schema.tables:
        tw = words(tname)
        if noun in tw or sing in tw:
            return True
    return False


def _mentioned_column(question: str, schema: Schema,
                      prefer_table: str | None) -> Optional[str]:
    """Best-mentioned column of the subject table (for bare ordering phrases
    like 'in descending order' with no explicit sort key). PKs are skipped
    unless the question asks for ids."""
    qt = set(tokens(question.lower()))
    wants_id = bool({"id", "ids", "identifier"} & qt)
    best: Optional[tuple[int, bool, str]] = None
    for col in schema.all_columns():
        if prefer_table and col.table.lower() != prefer_table.lower():
            continue
        if col.pk and not wants_id:
            continue
        overlap = len(words(col.name) & qt)
        key = (overlap, not col.pk, col.name)
        if overlap and (best is None or key > best[0]):
            best = (key, f"{col.table}.{col.name}")
    return best[1] if best else None


def grouping_column(question: str, schema: Schema) -> Optional[str]:
    """Find the column a 'per X' / 'each X' grouping refers to.

    Two phrasing families:
      P1: explicit "per|each|for each|one row per|grouped by|group by X"
      P2: "X and how many Y" / "X along with the number of Y"
          ("Show all template type codes and number of templates for each")
    Matches against column names first, then table names (resolving to the
    table's name-ish column, else its primary key). Conservative: returns
    None when nothing matches clearly.
    """
    ql = question.lower()
    col = _grouping_p1(ql, schema)
    if col:
        return col
    return _grouping_p2(ql, schema)


def _grouping_p1(ql: str, schema: Schema) -> Optional[str]:
    m = re.search(r"\b(?:per|each|for each|one row per|grouped by|group by)\s+"
                  r"(?:the\s+|a\s+|an\s+)?([a-z_ ]+?)(?:\s+and\b|\s+or\b|[,.?]|$)",
                  ql)
    if not m:
        return None
    phrase = m.group(1).strip()
    if not phrase:
        return None
    best: Optional[tuple[int, str]] = None

    def consider(score: int, qualified: str):
        nonlocal best
        if score and (best is None or score > best[0]):
            best = (score, qualified)

    for col in schema.all_columns():
        cwords = words(col.name)
        cname = " ".join(sorted(cwords))
        if cname == phrase:
            consider(10, f"{col.table}.{col.name}")
        elif cname in phrase or phrase in cname:
            consider(5, f"{col.table}.{col.name}")
        else:
            overlap = cwords & set(phrase.split())
            if overlap:
                consider(len(overlap), f"{col.table}.{col.name}")
    # table-name fallback: "for each department" -> department.name (or PK).
    # Only the phrase head (first two words) may match, so "each student
    # who has any pet" resolves to Student, not Has_Pet.
    head = " ".join(phrase.split()[:2])
    hw = set(head.split())
    for tname, table in schema.tables.items():
        tclean = tname.lower().replace("_", " ")
        tsing = tclean[:-1] if tclean.endswith("s") else tclean
        tw = set(tclean.split())
        if tclean == phrase or tsing == phrase or (tw and tw <= hw):
            target = None
            for c in table.columns:
                if c.name.lower() == "name":
                    target = c
                    break
            if target is None:
                for c in table.columns:
                    if c.pk:
                        target = c
                        break
            if target is None and table.columns:
                target = table.columns[0]
            if target is not None:
                consider(6, f"{tname}.{target.name}")
    return best[1] if best else None


def _grouping_p2(ql: str, schema: Schema) -> Optional[str]:
    """'X and how many Y' / 'X along with the number of Y' -> the X column."""
    m = re.search(
        r"^(?:what|show|return|list|give|find)\b(.+?)\b(?:and|along with|,)\b"
        r".+?\b(?:number of|how many|total)\b", ql)
    if not m:
        return None
    phrase = m.group(1)
    # resolve from the trailing content words of the X clause
    content = [t for t in phrase.split()
               if t not in ("the", "a", "an", "all", "are", "is", "of", "for")]
    tail = " ".join(content[-4:])
    if not tail:
        return None
    ranked = link_tables(phrase, schema)
    prefer = ranked[0][1] if ranked else None
    return _resolve_column(tail.split(), phrase, schema, prefer, strict=False)


def synthesize(question: str, schema: Schema) -> list[Assertion]:
    """Derive checkable assertions from (question, schema). Deterministic."""
    q = question.lower()
    out: list[Assertion] = []
    seen_kinds: set[str] = set()
    ranked = link_tables(question, schema)
    prefer_table = ranked[0][1] if ranked else None

    def add(a: Assertion):
        # one assertion per kind, keep the first (highest-priority) derivation
        if a.kind in seen_kinds:
            return
        seen_kinds.add(a.kind)
        out.append(a)

    # 1. top-N / first-N -> at most N rows
    m = re.search(r"\btop\s+(\d+|\w+)\b", q) or re.search(r"\bfirst\s+(\d+|\w+)\b", q)
    if m:
        n = _num_word(m.group(1))
        if n:
            add(Assertion("card_max", {"n": n},
                          f"result has at most {n} row(s)",
                          f"'{m.group(0).strip()}'"))

    # 2. superlatives -> ordering on the metric column; skipped for grouped
    #    queries ("maximum per group" is not a global maximum)
    gcol_early = grouping_column(question, schema)
    is_scalar_phrasing = bool(_AGG_WORDS.search(q) or _SCALAR_WORDS.search(q))

    # 1b. "most common(ly) <noun>" / "<noun> that is most common(ly)" ->
    #    the mode: exactly one row, one row per noun
    noun = None
    m = re.search(r"\bmost common(?:ly)?\s+([a-z]+)\b", q)
    if m and m.group(1) not in ("used", "across"):
        noun = m.group(1)
    else:
        m2 = re.search(r"([a-z]+)\s+that is most common(?:ly)?\b", q)
        if m2:
            noun = m2.group(1)
    if noun and not gcol_early:
            col = _resolve_column([noun], question, schema, prefer_table,
                                  strict=True)
            if col:
                add(Assertion("card_exact", {"n": 1}, "result has exactly 1 row",
                              "'most common' phrasing"))
                add(Assertion("unique", {"col": col},
                              f"column {col} has no duplicate values",
                              "'most common' phrasing"))

    for pat, hints, direction in _SUPERLATIVES:
        if gcol_early:
            break
        if re.search(pat, q):
            # noun following the adjective refines the hints ("greatest height");
            # a noun that is just the table name ("tallest poker player") is
            # uninformative and dropped so it cannot outvote the real hint
            m_noun = re.search(pat + r"\s+([a-z]+)", q)
            noun_hints = ([m_noun.group(1)] if m_noun
                          and len(m_noun.group(1)) > 2 else [])
            if prefer_table:
                tw = words(prefer_table)
                noun_hints = [h for h in noun_hints if h not in tw]
            # count-reading: "the most documents", "most number of templates"
            # is an argmax over counts, not an ordering of a column
            noun = m_noun.group(1) if m_noun else ""
            if noun in ("number",) or (
                    noun and noun.endswith("s") and not noun.endswith("ss")
                    and _is_table_word(noun, schema)):
                add(Assertion("card_exact", {"n": 1},
                              "result has exactly 1 row",
                              f"superlative '{pat.strip(chr(92)+'b')}' (count reading)"))
                break
            col = _resolve_column(noun_hints + hints, question, schema,
                                  prefer_table, strict=True)
            if col:
                m_num = re.search(r"(\d+|\w+)\s+" + pat.strip("\\b"), q)
                n = _num_word(m_num.group(1)) if m_num else None
                if n and n > 1:
                    add(Assertion("card_max", {"n": n},
                                  f"result has at most {n} row(s)",
                                  f"'{m_num.group(0).strip()}'"))
                else:
                    add(Assertion("card_exact", {"n": 1},
                                  "result has exactly 1 row",
                                  f"superlative '{pat.strip(chr(92)+'b')}'"))
                add(Assertion("ordered", {"col": col, "direction": direction},
                              f"result is ordered by {col} {direction}",
                              f"superlative '{pat.strip(chr(92)+'b')}'"))
            break

    # 3. explicit ordering phrases; the sort key comes from the phrase after
    #    "order of"/"sorted by" ("in ascending order of net worth"), not from
    #    stray question words. Without an "of X", the sort key is the most
    #    mentioned column of the subject table.
    m_ord = re.search(
        r"\b(?:in\s+(?:ascending|descending|increasing|decreasing)\s+"
        r"(?:alphabetical\s+)?order\s+of"
        r"|sorted\s+by|order\s+by|from\s+(?:highest to lowest|lowest to highest)\s+of)"
        r"\s+([a-z_ ]+?)(?:\s+and\b|\s+or\b|[,.?]|$)", q)
    if m_ord or re.search(r"\b(in (ascending|descending|increasing|decreasing) order|"
                           r"sorted by|order by|from (highest to lowest|lowest to highest))\b", q):
        direction = "DESC" if re.search(
            r"\b(descending|decreasing|highest to lowest)\b", q) else "ASC"
        if m_ord:
            hints = [t for t in m_ord.group(1).split() if len(t) > 2]
            col = _resolve_column(hints, question, schema, prefer_table,
                                  strict=True)
        else:
            # sort key = best-mentioned column of the subject table
            # (PKs excluded unless the question asks for ids)
            col = _mentioned_column(question, schema, prefer_table)
        if col:
            add(Assertion("ordered", {"col": col, "direction": direction},
                          f"result is ordered by {col} {direction}",
                          "ordering phrase"))

    # 5. per/each grouping -> unique grouping column. Only when the question
    #    aggregates per group ("average ... for each", "how many ... for each"):
    #    a bare "list X for each Y" is just a listing, not a GROUP BY.
    gcol = gcol_early
    has_agg = is_scalar_phrasing or bool(
        re.search(r"\b(maximum|minimum|average|avg|total|sum|count|number)\b", q))

    # 4. counting / scalar aggregation -> single scalar value
    if is_scalar_phrasing and not gcol:
        add(Assertion("single_value", {},
                      "result is a single scalar value",
                      "count/aggregation phrasing"))

    if gcol and has_agg:
        add(Assertion("unique", {"col": gcol},
                      f"column {gcol} has no duplicate values",
                      "'per/each' grouping phrasing"))

    # 6. explicit distinct/unique phrasing -> unique on the named column.
    #    The distinct applies to the noun right after the keyword
    #    ("different first names and ages" -> names, not ages). Skipped for
    #    count/aggregation phrasing ("number of distinct X" is a scalar);
    #    strict resolution avoids guessing the wrong column.
    if (re.search(r"\b(distinct|unique|different|no duplicates)\b", q)
            and not is_scalar_phrasing):
        m_d = re.search(
            r"\b(?:distinct|unique|different)\s+([a-z_ ]+?)(?:\s+and\b|\s+or\b|[,.?]|$)", q)
        dhints = ([t for t in m_d.group(1).split() if len(t) > 2] if m_d
                  else [t for t in re.findall(r"[a-z_]+", q) if len(t) > 2])
        col = _resolve_column(
            dhints, question, schema, prefer_table, strict=True)
        if col:
            add(Assertion("unique", {"col": col},
                          f"column {col} has no duplicate values",
                          "distinct/unique phrasing"))

    return out


# --------------------------------------------------------------------------
# Oracle: gold SQL -> structural assertions (upper bound + E2 ground truth)
# --------------------------------------------------------------------------

_LIMIT_RE = re.compile(r"\blimit\s+(\d+)", re.IGNORECASE)
_ORDER_RE = re.compile(
    r"\border\s+by\s+([`\"\w.]+)(?:\s+(asc|desc))?", re.IGNORECASE)
_DISTINCT_COL_RE = re.compile(
    r"select\s+distinct\s+([`\"\w.]+)", re.IGNORECASE)
_GROUP_RE = re.compile(r"\bgroup\s+by\s+([`\"\w.]+)", re.IGNORECASE)
_SCALAR_AGG_RE = re.compile(
    r"select\s+(?:count|sum|avg|min|max)\s*\(", re.IGNORECASE)
_ONLY_AGG_RE = re.compile(
    r"select\s+(?:count|sum|avg|min|max)\s*\([^)]*\)\s*(?:from\b|$)", re.IGNORECASE)


def assertions_from_gold(gold_sql: str) -> list[Assertion]:
    """Extract structural assertions mechanically from a gold query."""
    out: list[Assertion] = []
    m = _LIMIT_RE.search(gold_sql or "")
    if m:
        out.append(Assertion("card_max", {"n": int(m.group(1))},
                             f"result has at most {m.group(1)} row(s)", "gold"))
    m = _ORDER_RE.search(gold_sql or "")
    if m:
        out.append(Assertion(
            "ordered",
            {"col": m.group(1).strip('`"'), "direction": (m.group(2) or "ASC").upper()},
            f"result is ordered by {m.group(1)} {(m.group(2) or 'ASC').upper()}",
            "gold"))
    m = _DISTINCT_COL_RE.search(gold_sql or "")
    if m:
        out.append(Assertion("unique", {"col": m.group(1).strip('`"')},
                             f"column {m.group(1)} has no duplicate values", "gold"))
    else:
        m = _GROUP_RE.search(gold_sql or "")
        if m:
            out.append(Assertion("unique", {"col": m.group(1).strip('`"')},
                                 f"column {m.group(1)} has no duplicate values", "gold"))
    if _SCALAR_AGG_RE.search(gold_sql or "") and not _GROUP_RE.search(gold_sql or ""):
        if _ONLY_AGG_RE.search(gold_sql or ""):
            out.append(Assertion("single_value", {},
                                 "result is a single scalar value", "gold"))
        else:
            # SELECT min(x), y ... -> one row, several columns
            out.append(Assertion("card_exact", {"n": 1},
                                 "result has exactly 1 row", "gold"))
    return out
