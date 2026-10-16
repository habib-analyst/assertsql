"""Candidate generator: a rule-based text-to-SQL translator.

A *literal* translator, not a clever one: schema linking (with synonym
expansion), keyword intents, sample-value WHERE matching, and FK-graph join
planning. It is the "single-shot" baseline, and the repair loop's job is to
fix what it gets wrong. No LLM is involved at any point.

Design note: the generator and the assertion synthesizer are *separate*
rule sets. The generator never consults assertions; assertsql's gains come
only from the counterexample-guided repair loop.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Optional

from .assertions import grouping_column
from .ir import QueryIR, SelectItem, Condition, Join
from .schema import Schema, expand_text, link_tables, tokens, words


def link_columns(question: str, schema: Schema, table: str) -> list[tuple[float, str]]:
    """Score columns of one table by token overlap (synonym-expanded)."""
    qt = set(tokens(expand_text(question)))
    scored = []
    for col in schema.tables[table].columns:
        overlap = len(words(col.name) & qt)
        scored.append((overlap, f"{table}.{col.name}"))
    scored.sort(reverse=True)
    return scored


def _extract_numbers(question: str) -> list[float]:
    return [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", question)]


def _extract_quoted(question: str) -> list[tuple[str, str]]:
    return re.findall(r"'([^']+)'|\"([^\"]+)\"", question)


def _comparative(question: str) -> Optional[tuple[str, float]]:
    """'older than 30' -> ('>', 30); 'at least 5' -> ('>=', 5)."""
    q = question.lower()
    m = re.search(r"(older|greater|more|higher|larger|above|over|heavier|bigger)"
                  r"(?:\s+[a-z_]+){0,3}?\s+than\s+(\d+(?:\.\d+)?)", q)
    if m:
        return (">", float(m.group(2)))
    m = re.search(r"(younger|less|fewer|lower|smaller|below|under|lighter)"
                  r"(?:\s+[a-z_]+){0,3}?\s+than\s+(\d+(?:\.\d+)?)", q)
    if m:
        return ("<", float(m.group(2)))
    m = re.search(r"at least\s+(\d+(?:\.\d+)?)", q)
    if m:
        return (">=", float(m.group(1)))
    m = re.search(r"at most\s+(\d+(?:\.\d+)?)", q)
    if m:
        return ("<=", float(m.group(1)))
    return None


def _best_col_for_value(question: str, schema: Schema, table: str,
                        numeric: bool) -> Optional[str]:
    cols = link_columns(question, schema, table)
    for score, cname in cols:
        if score <= 0:
            continue
        col = schema.tables[table].col(cname.split(".")[-1])
        if col and col.numeric == numeric:
            return cname
    for col in schema.tables[table].columns:
        if col.numeric == numeric:
            return f"{table}.{col.name}"
    return None


def _metric_column(question: str, schema: Schema, table: str) -> Optional[str]:
    return _best_col_for_value(question, schema, table, numeric=True)


_NEG_RE = re.compile(r"\b(not|no|without|never|n't|none)\b")


def _value_variants(v: str) -> list[str]:
    """Singular/plural variants so 'cats' matches sample value 'cat'."""
    vl = v.lower()
    out = [vl]
    if vl.endswith("s") and len(vl) > 3:
        out.append(vl[:-1])
    else:
        out.append(vl + "s")
    return out


def _value_conditions(question: str, schema: Schema, table: str
                      ) -> tuple[list[Condition], list[tuple[str, str, list[str]]]]:
    """Match column sample values mentioned in the question.

    'students who have cat or dog pet' -> Pets.PetType IN ('cat','dog').
    Values under negation ('do not have a cat') are returned separately as
    (table, col, values) so the caller can build a NOT IN subquery.
    """
    q = question.lower()
    conds: list[Condition] = []
    negated: list[tuple[str, str, list[str]]] = []
    for tname, tab in schema.tables.items():
        for col in tab.columns:
            if not col.samples:
                continue
            hits = []
            for v in col.samples:
                vl = v.lower()
                if len(vl) < 2:
                    continue
                m = None
                for var in _value_variants(v):
                    m = re.search(r"\b" + re.escape(var) + r"\b", q)
                    if m:
                        break
                if not m:
                    continue
                before = q[max(0, m.start() - 30):m.start()]
                if _NEG_RE.search(before):
                    negated.append((tname, col.name, [v]))
                    continue
                hits.append(v)
            # longest-first dedupe (avoid 'cat' shadowing 'wild cat')
            hits = sorted(set(hits), key=len, reverse=True)
            picked: list[str] = []
            for h in hits:
                if not any(h.lower() in p.lower() and h != p for p in picked):
                    picked.append(h)
            if picked:
                cname = f"{tname}.{col.name}"
                if len(picked) == 1:
                    conds.append(Condition(col=cname, op="=", value=picked[0]))
                else:
                    conds.append(Condition(col=cname, op="IN", value=picked))
    return conds, negated


def _join_path(schema: Schema, src: str, dst: str) -> list[Join]:
    """BFS over the FK graph; returns Join steps from src to dst (cap 3 hops)."""
    adj: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for fk in schema.fks:
        adj[fk.from_table.lower()].append((fk.to_table, fk))
        adj[fk.to_table.lower()].append((fk.from_table, fk))
    prev: dict[str, tuple[str, object]] = {}
    seen = {src.lower()}
    dq = deque([src])
    while dq:
        cur = dq.popleft()
        if cur.lower() == dst.lower():
            break
        for nxt, fk in adj.get(cur.lower(), []):
            if nxt.lower() not in seen:
                seen.add(nxt.lower())
                prev[nxt.lower()] = (cur, fk)
                dq.append(nxt)
    if dst.lower() not in prev and dst.lower() != src.lower():
        return []
    # reconstruct
    hops: list[tuple[str, object]] = []
    cur = dst
    while cur.lower() != src.lower():
        pcur, fk = prev[cur.lower()]
        hops.append((cur, fk))
        cur = pcur
    hops.reverse()
    joins: list[Join] = []
    cur = src
    for nxt, fk in hops:
        if fk.from_table.lower() == cur.lower():
            left, right = f"{cur}.{fk.from_col}", f"{nxt}.{fk.to_col}"
        else:
            left, right = f"{cur}.{fk.to_col}", f"{nxt}.{fk.from_col}"
        joins.append(Join(table=nxt, left=left, right=right))
        cur = nxt
    return joins[:3]


def _projection_phrase(question: str) -> str:
    """The noun phrase right after the interrogative verb ('find X of ...').

    This usually names what is projected, so its column hits identify the
    subject table better than whole-question token soup.
    """
    m = re.search(
        r"\b(?:find|list|show|give|return|what|which|who|tell|display|name|"
        r"how many|how much)(?: me| us| is| are| does| do)?\s+"
        r"(?:the\s+|a\s+|an\s+)?(.+?)\s+"
        r"(?:of|who|which|that|where|whose|having|with|for|in|on|and|or|,|\?|$)",
        question.lower())
    return m.group(1) if m else ""


def _subject_table(question: str, schema: Schema) -> str:
    """Pick the question's subject table.

    Projection-phrase column hits count triple: in 'find the first name of
    students ...', Fname (Student) outvotes the bare 'pet' name mention.
    """
    qx = expand_text(question)
    phrase = _projection_phrase(qx)
    pqt = set(tokens(expand_text(phrase))) if phrase else set()
    ranked = link_tables(qx, schema)
    best, best_score = ranked[0][1], -1.0
    # "the X of <noun>": the noun after 'of' usually names the subject table
    of_bonus: dict[str, float] = {}
    m_of = re.search(r"\bof\s+(?:the\s+|a\s+|an\s+)?([a-z_]+)", qx.lower())
    if m_of:
        noun = m_of.group(1)
        noun_sing = noun[:-1] if noun.endswith("s") and len(noun) > 3 else noun
        for _, tname in ranked:
            if words(tname) & {noun, noun_sing}:
                of_bonus[tname] = 3.0
    for name_score, tname in ranked:
        proj_hits = sum(1 for s, _ in link_columns(phrase or question, schema,
                                                   tname) if s > 0) if phrase else 0
        total = 3.0 * proj_hits + name_score + of_bonus.get(tname, 0.0)
        if total > best_score:
            best, best_score = tname, total
    return best


def generate(question: str, schema: Schema, evidence: str = "") -> QueryIR:
    """Translate a question to a candidate QueryIR. Deterministic."""
    q = (question + " " + evidence).lower()
    ir = QueryIR()

    table = _subject_table(q, schema)
    ir.table = table

    cols = link_columns(q, schema, table)
    # projection discipline: only top-scoring columns; PKs excluded unless
    # the question asks for ids (they otherwise inflate column counts)
    qt = set(tokens(expand_text(q)))
    wants_id = bool({"id", "ids", "identifier"} & qt)
    pk_names = {c.name.lower() for c in schema.tables[table].columns if c.pk}
    scored = [(s, c) for s, c in cols if s > 0
              and (wants_id or c.split(".")[-1].lower() not in pk_names)]
    maxscore = max([s for s, _ in scored], default=0)
    mentioned = [c for s, c in scored if s == maxscore][:3]
    if not mentioned:
        mentioned = [c for s, c in cols[:1]]

    gcol = grouping_column(question, schema)

    is_count = bool(re.search(
        r"\bhow many\b|\bnumber of\b|\bcount of\b|\bcount the\b", q))
    # "how much" is a count only with an explicit plural ("how much money"
    # is a sum/value question, not a count)
    if re.search(r"\bhow much\b", q) and not is_count:
        pass  # handled as a value/aggregation question below
    is_avg = bool(re.search(r"\baverage\b|\bavg\b|\bmean\b", q))
    is_sum = bool(re.search(r"\btotal\b|\bsum of\b|\bsum the\b", q))
    is_max = bool(re.search(r"\bmaximum\b|\bmax\b", q))
    is_min = bool(re.search(r"\bminimum\b|\bmin\b", q))
    m_top = re.search(r"\btop\s+(\d+)\b", q) or re.search(r"\bfirst\s+(\d+)\b", q)
    is_super = bool(re.search(
        r"\byoungest\b|\boldest\b|\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b|"
        r"\bgreatest\b|\bleast\b|"
        r"\bmost expensive\b|\bcheapest\b|\bmaximum\b|\bminimum\b|"
        r"\bearliest\b|\blatest\b|\btallest\b|\bshortest\b", q))
    is_distinct = bool(re.search(r"\b(distinct|different|unique)\b", q))

    # ---- SELECT clause ------------------------------------------------
    if is_count:
        if is_distinct and mentioned:
            ir.select = [SelectItem(kind="agg", agg="COUNT", col=mentioned[0],
                                    distinct=True)]
        else:
            ir.select = [SelectItem(kind="count_star")]
    elif is_avg or is_sum or is_max or is_min:
        agg = "AVG" if is_avg else "SUM" if is_sum else "MIN" if is_min else "MAX"
        col = _best_col_for_value(q, schema, table, numeric=True)
        ir.select = ([SelectItem(kind="agg", agg=agg, col=col)] if col
                     else [SelectItem(kind="count_star")])
    else:
        proj = mentioned[:3] or [f"{table}.{schema.tables[table].columns[0].name}"]
        ir.select = [SelectItem(kind="col", col=c) for c in proj]

    # ---- WHERE clause -------------------------------------------------
    val_conds, negated = _value_conditions(question, schema, table)
    for cond in val_conds:
        ir.where.append(cond)
    # negated value hits -> NOT IN subquery over the FK path
    # ("students who do not have a cat pet")
    pk_col = next((c.name for c in schema.tables[table].columns if c.pk),
                  schema.tables[table].columns[0].name)
    for tname, colname, values in negated:
        if tname.lower() == table.lower():
            op = "IN" if len(values) > 1 else "="
            ir.where.append(Condition(col=f"{table}.{colname}", op="NOT " + op
                                      if op == "IN" else "!=",
                                      value=values if len(values) > 1 else values[0]))
            continue
        sub = QueryIR(
            select=[SelectItem(kind="col", col=f"{table}.{pk_col}")],
            table=table,
            joins=_join_path(schema, table, tname),
            where=[Condition(col=f"{tname}.{colname}",
                             op="IN" if len(values) > 1 else "=",
                             value=values if len(values) > 1 else values[0])])
        if sub.joins or tname.lower() == table.lower():
            ir.where.append(Condition(col=f"{table}.{pk_col}", op="NOT IN",
                                      subquery=sub))
    comp = _comparative(q)
    if comp:
        op, val = comp
        col = _best_col_for_value(q, schema, table, numeric=True)
        if col:
            ir.where.append(Condition(col=col, op=op, value=val))
    for q1, q2 in _extract_quoted(question):
        vals = q1 or q2
        if not vals:
            continue
        col = _best_col_for_value(q, schema, table, numeric=False)
        if col:
            ir.where.append(Condition(col=col, op="LIKE", value=f"%{vals}%"))
    for num in _extract_numbers(question):
        if 1900 <= num <= 2100 and not comp:
            for col in schema.tables[table].columns:
                if col.numeric and col.name.lower() in ("year", "yr", "season"):
                    ir.where.append(Condition(col=f"{table}.{col.name}", op="=",
                                              value=int(num)))
                    break
            break

    # ---- JOIN planning: pull in tables owning WHERE columns -----------
    needed = {c.col.split(".")[0].lower() for c in ir.where
              if "." in c.col}
    needed.discard(table.lower())
    joined = {table.lower()}
    for t in sorted(needed):
        if t in joined or t not in {x.lower() for x in schema.tables}:
            continue
        real = next(x for x in schema.tables if x.lower() == t)
        for j in _join_path(schema, table, real):
            if j.table.lower() not in joined:
                ir.joins.append(j)
                joined.add(j.table.lower())

    # plain ordering phrases ("in descending order", "sorted by X")
    m_ordphrase = re.search(
        r"\b(?:in\s+(ascending|descending|increasing|decreasing)\s+order"
        r"|sorted\s+by|order\s+by)\b", q)
    if gcol:
        gtable = gcol.split(".")[0]
        if gtable.lower() == table.lower():
            if gcol not in [s.col for s in ir.select]:
                ir.select.insert(0, SelectItem(kind="col", col=gcol))
            ir.group_by = [gcol]
        # grouped: global superlative ORDER BY / LIMIT does not apply
    elif m_top:
        n = int(m_top.group(1))
        metric = _metric_column(q, schema, table)
        if metric:
            ir.order_by = [(metric, "DESC")]
        ir.limit = n
    elif is_super:
        if re.search(r"\byoungest\b", q):
            c = _best_col_for_value("age", schema, table, numeric=True)
            ir.order_by = [(c, "ASC")] if c else []
        elif re.search(r"\boldest\b", q):
            c = _best_col_for_value("age", schema, table, numeric=True)
            ir.order_by = [(c, "DESC")] if c else []
        elif re.search(r"\bhighest\b|\blargest\b|\bmost expensive\b|\bmaximum\b|"
                       r"\blatest\b|\btallest\b|\bgreatest\b", q):
            c = _metric_column(q, schema, table)
            ir.order_by = [(c, "DESC")] if c else []
        elif re.search(r"\blowest\b|\bsmallest\b|\bcheapest\b|\bminimum\b|"
                       r"\bearliest\b|\bshortest\b|\bleast\b", q):
            c = _metric_column(q, schema, table)
            ir.order_by = [(c, "ASC")] if c else []
        if ir.order_by:
            m_n = re.search(r"\b(\d+)\s+(?:youngest|oldest|highest|lowest|"
                            r"largest|smallest|maximum|minimum|earliest|"
                            r"latest|tallest|shortest|cheapest)\b", q)
            ir.limit = int(m_n.group(1)) if m_n else 1
    elif m_ordphrase:
        direction = "DESC" if re.search(r"descending|decreasing", q) else "ASC"
        m_of = re.search(r"(?:order\s+of|sorted\s+by|order\s+by)\s+([a-z_ ]+?)$",
                         q)
        keyq = m_of.group(1) if m_of else q
        metric = _metric_column(keyq, schema, table)
        if metric:
            ir.order_by = [(metric, direction)]

    if is_distinct and not is_count:
        ir.distinct = True

    return ir
