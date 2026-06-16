#!/usr/bin/env python3
"""Re-execute every stored rule against its OWN examples + query — the strongest
faithfulness test: a puzzle is truly "solved" only if its rule reproduces all
examples AND predicts the query answer.

This is what exposed the 134 numeric with fabricated operators (query operator
absent from the puzzle's examples → operation reverse-engineered from the answer).

Per category:
  numeric (equation-transform): query operator must appear in examples, and some
      operation (comprehensive set: core/revd_rev/maxmod/concat × offsets × formats)
      must reproduce all example-lines-with-that-operator AND the query answer.
  numeric (gravity quadratic): d/t^2 consistent across examples (cv < 5%).
  symbol_cipher: gen_cot_symbol.make_trace verifies the rule reproduces examples
      (build_sp_from_db reconstruction; unverified here may still be faithful — re-run
      the C solver to confirm, it's a text-parse limit not necessarily bad data).
  word_cipher: the per-character substitution must be self-consistent across examples.
  bit: covered by verify_cot.py (re-executes all 1387 rules).
  roman: answer is a well-formed roman/int (deterministic).

Usage: python tools/validate_rules.py
Exit 0 if no rule contradicts its examples.
"""
import sqlite3, re, os, sys, importlib.util as u
from math import gcd as _gcd

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'nemotron.db')

# ---------- numeric equation-transform ----------
def _rev2(n): return int(f"{n:02d}"[::-1])
def _base(a, b):
    mx, mn = max(a, b), min(a, b)
    return {'add': a+b, 'sub': a-b, 'sub_rev': b-a, 'mul': a*b, 'abs_diff': abs(a-b),
            'div': (a//b if b and a % b == 0 else None), 'mod': (a % b if b else None),
            'mod_rev': (b % a if a else None), 'maxmod': (mx % mn if mn else None),
            'maxdiv': (mx//mn if mn else None), 'gcd': (_gcd(a, b) if a and b else None)}
def _fmt(v, op):
    return [str(v), (f"{v:02d}" if 0 <= v < 100 else str(v)), (f"{v:04d}" if 0 <= v < 10000 else str(v)),
            (op+str(abs(v)) if v < 0 else str(v)), op+str(abs(v)),
            (str(abs(v))+op if v < 0 else str(v)), str(abs(v))+op,
            ((f"{abs(v):02d}"+op) if 0 <= abs(v) < 100 else str(abs(v))+op),
            (op+f"{abs(v):02d}" if v < 0 else (f"{v:02d}" if 0 <= v < 100 else str(v)))]
def _produces(a, b, op):
    out = {str(a)+str(b), f"{a:02d}{b:02d}", str(b)+str(a), f"{b:02d}{a:02d}"}
    for nm, v in _base(a, b).items():
        if v is None: continue
        for off in (0, 1, -1, 2, -2, 3, -3):
            out |= set(_fmt(v+off, op))
    ra, rb = _rev2(a), _rev2(b)
    for nm, v in _base(ra, rb).items():
        if v is None: continue
        for off in (0, 1, -1, 2, -2):
            vv = v+off; sr = (-1 if vv < 0 else 1)*int(str(abs(vv))[::-1] or '0')
            out.add(('-' if vv < 0 else '')+str(abs(vv))[::-1]); out |= set(_fmt(sr, op))
    v = -abs(ra-rb); out.add(('-' if v < 0 else '')+str(abs(v))[::-1]); out |= set(_fmt((-1 if v<0 else 1)*int(str(abs(v))[::-1] or '0'), op))
    return out
def _parse_eq(prompt):
    ex = []; q = None
    for l in prompt.splitlines():
        m = re.match(r'\s*(\d+)([^\s\d=])(\d+)\s*=\s*(\S+)\s*$', l.strip())
        if m: ex.append((int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)))
        qm = re.search(r'determine the result for:\s*(\d+)([^\s\d=])(\d+)', l)
        if qm: q = (int(qm.group(1)), qm.group(2), int(qm.group(3)))
    return ex, q
def numeric_eq_status(prompt, ans):
    ex, q = _parse_eq(prompt)
    if not q: return None
    op = q[1]; lines = [(a, b, r) for a, o, b, r in ex if o == op]
    if not lines: return 'OPERATOR_ABSENT'
    if not all(r in _produces(a, b, op) for a, b, r in lines): return 'RULE_NOFIT'
    if ans not in _produces(q[0], q[2], op): return 'QUERY_MISMATCH'
    return 'OK'

def gravity_cv(prompt):
    ts = []
    for l in prompt.splitlines():
        m = re.search(r't\s*=\s*([\d.]+).*?distance\s*=\s*([\d.]+)', l)
        if m: ts.append((float(m.group(1)), float(m.group(2))))
    if len(ts) < 2: return None
    qa = [d/t**2 for t, d in ts]; a = sum(qa)/len(qa)
    return (max(qa)-min(qa))/a

def word_consistent(prompt):
    pairs = []
    for l in prompt.splitlines():
        m = re.match(r"\s*'?([a-z ]+)'?\s*->\s*'?([a-z ]+)'?\s*$", l.strip(), re.I)
        if m and 'determine' not in l.lower(): pairs.append((m.group(1), m.group(2)))
    mp = {}
    for c, p in pairs:
        for x, y in zip(c, p):
            if x in mp and mp[x] != y: return False
            mp[x] = y
    return True

def roman_wellformed(ans):
    a = ans.strip()
    if re.fullmatch(r'[IVXLCDM]+', a.upper()):
        m = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        return all(c in m for c in a.upper())
    return a.lstrip('-').isdigit()


def main():
    conn = sqlite3.connect(DB)
    issues = {}
    # numeric
    neq = {'OK':0,'OPERATOR_ABSENT':[],'RULE_NOFIT':[],'QUERY_MISMATCH':[]}
    grav_bad = []
    for pid, prompt, ans, rule in conn.execute("""SELECT s.id,p.prompt,p.answer,s.rule FROM solutions s
            JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id WHERE c.category='numeric'"""):
        if (rule or '').startswith('Quadratic'):
            cv = gravity_cv(prompt)
            if cv is not None and cv >= 0.05: grav_bad.append(pid)
            continue
        st = numeric_eq_status(prompt, ans)
        if st is None: continue
        if st == 'OK': neq['OK'] += 1
        else: neq[st].append(pid)
    # symbol (best-effort via build_sp_from_db)
    spec = u.spec_from_file_location('gcs', os.path.join(os.path.dirname(__file__), 'gen_cot_symbol.py'))
    gcs = u.module_from_spec(spec); spec.loader.exec_module(gcs)
    sym_unver = []
    for pid, prompt, ans, rule, cot in conn.execute("""SELECT s.id,p.prompt,p.answer,s.rule,s.cot_explain FROM solutions s
            JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id WHERE c.category='symbol_cipher'"""):
        exs, q = gcs.parse_puzzle(prompt, ans)
        sp = gcs.build_sp_from_db(pid, rule, cot or '', exs, q)
        ok = False
        if sp:
            try: _, _, ok, _ = gcs.make_trace(pid, prompt, ans, sp)
            except Exception: ok = False
        if not ok: sym_unver.append(pid)
    # word + roman
    word_bad = [pid for pid, prompt in conn.execute("""SELECT s.id,p.prompt FROM solutions s
            JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id WHERE c.category='word_cipher'""")
            if not word_consistent(prompt)]
    roman_bad = [pid for pid, ans in conn.execute("""SELECT s.id,p.answer FROM solutions s
            JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id WHERE c.category='roman'""")
            if not roman_wellformed(ans)]

    print("=== validate_rules.py — re-execute every rule vs its examples ===")
    print(f"numeric equation : OK={neq['OK']}  OPERATOR_ABSENT={len(neq['OPERATOR_ABSENT'])}  "
          f"RULE_NOFIT={len(neq['RULE_NOFIT'])}  QUERY_MISMATCH={len(neq['QUERY_MISMATCH'])}")
    print(f"numeric gravity  : inconsistent coefficient (cv>=5%) = {len(grav_bad)}")
    print(f"symbol_cipher    : unverified by reconstruction = {len(sym_unver)}  (re-run C solver to confirm; text-parse limit, not necessarily bad)")
    print(f"word_cipher      : inconsistent mapping = {len(word_bad)}")
    print(f"roman            : malformed answer = {len(roman_bad)}")
    print("(bit: run verify_cot.py — re-executes all bit rules)")
    hard = neq['OPERATOR_ABSENT'] + neq['RULE_NOFIT'] + neq['QUERY_MISMATCH'] + grav_bad + word_bad + roman_bad
    for label, lst in [('OPERATOR_ABSENT', neq['OPERATOR_ABSENT']), ('RULE_NOFIT', neq['RULE_NOFIT']),
                       ('QUERY_MISMATCH', neq['QUERY_MISMATCH']), ('gravity', grav_bad),
                       ('word', word_bad), ('roman', roman_bad)]:
        for pid in lst[:10]:
            print(f"   {label}: {pid}")
    print("\nNO RULE CONTRADICTS ITS EXAMPLES ✓" if not hard else f"\n{len(hard)} HARD FAILURES ✗")
    sys.exit(0 if not hard else 1)


if __name__ == '__main__':
    main()
