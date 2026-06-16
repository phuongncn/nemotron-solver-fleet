#!/usr/bin/env python3
"""Near-miss scan: find puzzles whose LLM prediction is a NEAR-MISS of the origin answer.

The solver fleet demands EXACT match, so a puzzle whose rule is right but whose origin
answer is buggy/oddly-formatted never "solves". The LLM (base bf16 / v3-llm-harvest) already
brute-forced a rule per puzzle; its `predicted` value, when it MISSES the origin answer only
by a classifiable near-miss (affix, permutation, reversed, 1-edit, ...), is strong evidence
the rule is right and the ORIGIN answer is wrong. Record those in `suspect_puzzles`.

Classify (predicted vs expected); category-agnostic string + numeric checks:
  affix · permutation · reversed · edit1 · case_sep · off_by_small · sign_flip · scale_shift · hamming1

Usage:
  .venv/bin/python tools/near_miss_scan.py                 # dry-run: summary by reason + samples
  .venv/bin/python tools/near_miss_scan.py --apply         # write suspects to suspect_puzzles
"""
import sqlite3, os, sys, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_metric import verify

DB = '/home/admin/nemotron/data/nemotron.db'

def _num(s):
    s = str(s).strip()
    try:
        f = float(s.replace(',', '')); return f
    except ValueError:
        return None

def _lev1(a, b):
    """True if Levenshtein(a,b)==1 (one insert/delete/substitute)."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:                      # substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    # insert/delete: shorter is longer with one char removed
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False

def classify(pred, exp):
    """Return a near-miss reason (or None). pred=LLM answer, exp=origin answer. Both differ & not verify()."""
    p, e = str(pred).strip(), str(exp).strip()
    if not p or not e or p == e:
        return None
    pl, el = p.lower(), e.lower()
    # ---- numeric ----
    np_, ne = _num(p), _num(e)
    if np_ is not None and ne is not None:
        if np_ == ne:
            return None                                   # equal value (e.g. 051 vs 51) -> handled as affix below if str differs
        if np_ == -ne:
            return 'sign_flip'
        if ne != 0 and any(abs(np_ * 10 ** k - ne) < 1e-6 or abs(np_ / 10 ** k - ne) < 1e-6 for k in (1, 2, 3)):
            return 'scale_shift'
        if float(np_).is_integer() and float(ne).is_integer() and abs(np_ - ne) <= 2:
            return 'off_by_small'
    # ---- string ----
    if sorted(pl) == sorted(el):
        return 'reversed' if p == e[::-1] else 'permutation'
    if pl == el[::-1]:
        return 'reversed'
    # affix: one is the other +/- stray leading/trailing chars (incl leading zero, sign, punctuation)
    if el.endswith(pl) or el.startswith(pl) or pl.endswith(el) or pl.startswith(el):
        return 'affix'
    if p.lstrip('0+-') == e.lstrip('0+-') and p.lstrip('0+-'):
        return 'affix'
    # hamming1 (same length, exactly one position differs) — e.g. bit 1-off
    if len(p) == len(e) and sum(x != y for x, y in zip(p, e)) == 1:
        return 'hamming1'
    # case / separator only
    if re.sub(r'[\s_,-]', '', pl) == re.sub(r'[\s_,-]', '', el):
        return 'case_sep'
    if _lev1(p, e):
        return 'edit1'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--models', nargs='*', default=['v3-llm-harvest', 'nemotron-nano-bf16'],
                    help='prediction sources, priority order')
    args = ap.parse_args()
    db = sqlite3.connect(DB, timeout=60); db.row_factory = sqlite3.Row
    # one prediction per puzzle (first model in priority that has predicted != expected)
    seen = {}
    for m in args.models:
        for r in db.execute(
            "SELECT b.puzzle_id, b.expected, b.predicted, c.category FROM base_answers b "
            "JOIN classifications c ON c.id=b.puzzle_id "
            "WHERE b.model=? AND b.predicted IS NOT NULL AND b.predicted!='' AND b.expected!=b.predicted", (m,)):
            if r['puzzle_id'] in seen:
                continue
            # only genuine MISSES (marked wrong by the official metric). verify=true cases
            # (e.g. float rounding 50.51 vs 50.50, leading-zero) are already correct / handled
            # by the integer value_mismatch scan -> skip the rounding noise here.
            if verify(str(r['expected']).strip(), str(r['predicted']).strip()):
                continue
            reason = classify(r['predicted'], r['expected'])
            if reason:
                seen[r['puzzle_id']] = (r['category'], r['expected'], r['predicted'], reason)
    from collections import Counter
    by = Counter((v[0], v[3]) for v in seen.values())
    print(f"=== near-miss scan: {len(seen)} puzzles flagged ===")
    print("by (category, reason):")
    for k, n in by.most_common():
        print(f"  {k[0]:14s} {k[1]:14s} {n}")
    # samples per reason
    print("\nsamples:")
    shown = set()
    for pid, (cat, e, p, reason) in seen.items():
        if reason not in shown:
            shown.add(reason); print(f"  [{reason}] {pid[:8]} {cat}: expected={e!r} pred={p!r}")
    if args.apply:
        n = 0
        for pid, (cat, e, p, reason) in seen.items():
            db.execute("INSERT OR REPLACE INTO suspect_puzzles VALUES(?,?,?,?,?,?,?,?,datetime('now'))",
                       (pid, cat, reason, e, p,
                        'LLM prediction is a near-miss of origin answer (rule likely right, origin suspect)',
                        'suspect', 'near-miss-scan'))
            n += 1
        db.commit()
        print(f"\napplied: {n} rows -> suspect_puzzles")


if __name__ == '__main__':
    main()
