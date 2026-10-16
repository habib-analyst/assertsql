"""Benchmark runner for assertsql: E1 (Spider subset), E2 (assertion quality),
E3 (repair efficiency), E4 (BIRD mini-dev).

Reads datasets prepared under:
  $ASSERTSQL_DATA  (default: hidden_files/assertsql_data/)
    spider_subset.json, bird_minidev.json, MANIFEST.json

Writes results JSONL + summary to:
  $ASSERTSQL_RESULTS (default: hidden_files/build/assertsql/results/)

Run:  python bench/run_bench.py [--e1] [--e2] [--e3] [--e4] [--limit N]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assertsql.eval import BenchItem, accuracy, assertion_quality, run_arm

GOAL = os.path.expanduser(
    "~/workspace/goals/github-portfolio-linkedin-presence-buildout")
DATA = os.environ.get("ASSERTSQL_DATA",
                      os.path.join(GOAL, "hidden_files", "assertsql_data"))
RESULTS = os.environ.get(
    "ASSERTSQL_RESULTS",
    os.path.join(GOAL, "hidden_files", "build", "assertsql", "results"))


def _manifest():
    with open(os.path.join(DATA, "MANIFEST.json")) as f:
        return json.load(f)


_MANIFEST = None


def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = _manifest()
    return _MANIFEST


def load_spider(limit=None):
    m = manifest()["spider"]
    items = json.load(open(m["questions_file"]))
    db_by_id = {d["db_id"]: d["path"] for d in m["dbs"]}
    out = []
    for i, q in enumerate(items):
        db_path = db_by_id.get(q["db_id"])
        if not db_path or not os.path.exists(db_path):
            continue
        out.append(BenchItem(qid=f"spider-{i}", question=q["question"],
                             db_id=q["db_id"], db_path=db_path,
                             gold_sql=q["query"]))
    return out[:limit] if limit else out


def load_bird(limit=None):
    m = manifest()["bird"]
    items = json.load(open(m["questions_file"]))
    db_by_id = {d["db_id"]: d["path"] for d in m["dbs"]}
    out = []
    for q in items:
        db_path = db_by_id.get(q["db_id"])
        if not db_path or not os.path.exists(db_path):
            continue
        out.append(BenchItem(qid=f"bird-{q['question_id']}",
                             question=q["question"], db_id=q["db_id"],
                             db_path=db_path, gold_sql=q["SQL"],
                             evidence=q.get("evidence", "")))
    return out[:limit] if limit else out


def run_arm_logged(items, arm, tag):
    done = [0]

    def cb(it, res):
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(items):
            print(f"  [{tag}/{arm}] {done[0]}/{len(items)}", flush=True)

    t0 = time.time()
    res = run_arm(items, arm, on_item=cb)
    dt = time.time() - t0
    acc = accuracy(res)
    print(f"  [{tag}/{arm}] accuracy={acc:.3f} ({sum(r.correct for r in res)}/"
          f"{len(res)}) in {dt:.1f}s", flush=True)
    return res, acc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--e3", action="store_true")
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or not (args.e1 or args.e2 or args.e3 or args.e4):
        args.e1 = args.e2 = args.e3 = args.e4 = True
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}

    if args.e1 or args.e2 or args.e3:
        spider = load_spider(args.limit)
        print(f"E1/E2/E3: {len(spider)} spider subset questions")
        summary["spider_n"] = len(spider)

    if args.e1:
        accs = {}
        traces = {}
        for arm in ("single", "assertsql", "oracle"):
            res, acc, dt = run_arm_logged(spider, arm, "spider")
            accs[arm] = acc
            summary[f"spider_{arm}_s"] = dt
            json.dump([{"qid": r.qid, "arm": r.arm, "correct": r.correct,
                        "note": r.note,
                        "trace": r.trace.to_dict() if r.trace else None}
                       for r in res],
                      open(os.path.join(RESULTS, f"spider_{arm}.json"), "w"),
                      default=str)
            if arm == "assertsql":
                traces = res
        summary["spider_acc_single"] = accs["single"]
        summary["spider_acc_assertsql"] = accs["assertsql"]
        summary["spider_acc_oracle"] = accs["oracle"]
        summary["spider_gain_pp"] = (accs["assertsql"] - accs["single"]) * 100

    if args.e2:
        q = assertion_quality(spider)
        summary["assert_precision"] = q["precision"]
        summary["assert_recall"] = q["recall"]
        summary["assert_tp_fp_fn"] = [q["tp"], q["fp"], q["fn"]]
        summary["assert_per_kind"] = q["per_kind_tp_fp"]
        print(f"E2 assertion quality: P={q['precision']:.3f} "
              f"R={q['recall']:.3f} (tp={q['tp']} fp={q['fp']} fn={q['fn']})")

    if args.e3:
        res = json.load(open(os.path.join(RESULTS, "spider_assertsql.json")))
        # E3 counts only questions where repair actually engaged
        # (>=1 violated assertion at round 0). "Fixed" = 0 violations;
        # "unfixable" counts honest failures, not vacuous passes.
        rounds, fixed, unfix, improved = [], 0, 0, 0
        engaged = 0
        for r in res:
            tr = r["trace"]
            if tr is None:
                continue
            if tr["n_violations_start"] == 0:
                continue
            engaged += 1
            rounds.append(tr["rounds_used"])
            if tr["status"] == "fixed":
                fixed += 1
            elif tr["status"] == "improved":
                improved += 1
            else:
                unfix += 1
        rounds.sort()
        med = rounds[len(rounds) // 2] if rounds else 0
        total = engaged or 1
        summary["repair_engaged"] = engaged
        summary["repair_median_rounds"] = med
        summary["repair_fixed"] = fixed
        summary["repair_improved"] = improved
        summary["repair_unfixable"] = unfix
        summary["repair_unfixable_rate"] = unfix / total
        print(f"E3 repair (engaged={engaged}/{len(res)}): median rounds={med}, "
              f"fixed={fixed}, improved={improved}, unfixable={unfix} "
              f"({unfix/total:.1%} of engaged)")

    if args.e4:
        bird = load_bird(args.limit)
        print(f"E4: {len(bird)} BIRD mini-dev questions")
        summary["bird_n"] = len(bird)
        for arm in ("single", "assertsql", "oracle"):
            res, acc, dt = run_arm_logged(bird, arm, "bird")
            summary[f"bird_acc_{arm}"] = acc
            summary[f"bird_{arm}_s"] = dt
            json.dump([{"qid": r.qid, "arm": r.arm, "correct": r.correct,
                        "note": r.note,
                        "trace": r.trace.to_dict() if r.trace else None}
                       for r in res],
                      open(os.path.join(RESULTS, f"bird_{arm}.json"), "w"),
                      default=str)
        summary["bird_gain_pp"] = (summary["bird_acc_assertsql"]
                                   - summary["bird_acc_single"]) * 100

    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"),
              indent=2)
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
