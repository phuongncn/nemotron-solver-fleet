#!/usr/bin/env python3
"""Comprehensive CoT faithfulness verifier — run before every training run.

Checks (the standard that caught 422 gravity + 29 revsub + 7 bit import bugs):
  #1 boxed == answer (greedy regex, brace-answer safe)
  #2 every intermediate arithmetic "A op B = C" is correct
       - symbol: the op-step before "(opname)" (avoids BE/LE decode false positives)
       - numeric: explicit X op Y = Z equations
       - bit: every per-bit rule re-executed against ALL examples + query
  #3 no exploration phrases (Wait/Actually/I notice/seems/maybe/to complete/(?->)
  #4 every CoT has "Find:" and ends with "}"

Exit code 0 if clean, 1 if any issue. Usage: python tools/verify_cot.py [--show N]
"""
import sqlite3, re, sys, os
from math import gcd

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'nemotron.db')
SHOW = int(sys.argv[sys.argv.index('--show')+1]) if '--show' in sys.argv else 8
# --field FIELD : which CoT column to verify (default cot_explain, e.g. cot_c3)
FIELD = sys.argv[sys.argv.index('--field')+1] if '--field' in sys.argv else 'cot_explain'
FORBID = ['Wait', 'Actually', 'I notice', 'seems', 'maybe', 'to complete', '(?->']

# ---------- bit rule executor (LSB-indexed, mirrors solver semantics) ----------
def _ch(a, b, c): return (a & b) ^ ((1 - a) & c)
def _maj(a, b, c): return 1 if (a + b + c) >= 2 else 0

def _bit_term(term, IN):
    t = term.strip(); inv = False
    m = re.match(r'N(?:OT)?-(.+)$', t)
    if m: inv = True; t = m.group(1)
    mm = re.match(r'([A-Z\-]+?)(\d+)$', t)
    if not mm:
        if t in ('0', '1'): return int(t)
        raise ValueError(f"unparse:{term}")
    op, digs = mm.group(1), mm.group(2); d = [int(x) for x in digs]
    g = lambda i: IN[i]
    if op == 'I': r = g(d[0])
    elif op in ('N', 'NOT'): r = 1 - g(d[0])
    elif op == 'C': r = int(digs)
    elif op == 'XOR': r = g(d[0]) ^ g(d[1])
    elif op == 'XNOR': r = 1 - (g(d[0]) ^ g(d[1]))
    elif op == 'AND': r = g(d[0]) & g(d[1])
    elif op == 'OR': r = g(d[0]) | g(d[1])
    elif op == 'NAND': r = 1 - (g(d[0]) & g(d[1]))
    elif op == 'NOR': r = 1 - (g(d[0]) | g(d[1]))
    elif op == 'AND-NOT': r = g(d[0]) & (1 - g(d[1]))
    elif op == 'OR-NOT': r = g(d[0]) | (1 - g(d[1]))
    elif op == 'XOR-NOT': r = g(d[0]) ^ (1 - g(d[1]))
    elif op == 'CH': r = _ch(g(d[0]), g(d[1]), g(d[2]))
    elif op == 'MAJ': r = _maj(g(d[0]), g(d[1]), g(d[2]))
    elif op == 'OAI': r = (g(d[0]) | g(d[1])) & g(d[2])
    elif op == 'AOI': r = (g(d[0]) & g(d[1])) | g(d[2])
    elif op == 'TAND': r = g(d[0]) & g(d[1]) & g(d[2])          # 3-input AND
    elif op == 'TOR': r = g(d[0]) | g(d[1]) | g(d[2])           # 3-input OR
    elif op == 'TNAND': r = 1 - (g(d[0]) & g(d[1]) & g(d[2]))   # 3-input NAND
    elif op == 'TNOR': r = 1 - (g(d[0]) | g(d[1]) | g(d[2]))    # 3-input NOR
    else: raise ValueError(f"unknownop:{op}")
    return 1 - r if inv else r

def _bit_apply(rule, in_str):
    IN = [int(c) for c in in_str]
    out = ['?'] * 8
    for term in [x for x in rule.split(';') if x.strip()]:
        m = re.match(r'\s*bit(\d)=(.+)$', term)
        if not m: raise ValueError(f"badterm:{term}")
        out[int(m.group(1))] = str(_bit_term(m.group(2), IN))
    return ''.join(out)

def _bit_examples(prompt):
    exs = []; q = None
    for ln in prompt.splitlines():
        m = re.match(r'\s*([01]{8})\s*->\s*([01]{8})', ln)
        if m: exs.append((m.group(1), m.group(2)))
        qm = re.search(r'determine the output for:\s*([01]{8})', ln)
        if qm: q = qm.group(1)
    return exs, q

# ---------- symbol op-step check ----------
OPN = 'add|subtract|multiply|reverse-subtract|XOR|AND|OR|GCD|max%min|max//min|mod \\(a%b\\)|divide \\(a//b\\)|absolute difference'
def _sym_ok(seg):
    seg = seg.split(';')[-1].strip()
    m = re.search(r'(-?\d+)\s*(\+|-|×|//|mod|XOR|AND|OR)\s*(-?\d+)\s*=\s*(-?\d+)\s*$', seg)
    if not m:
        g = re.search(r'gcd\((-?\d+),(-?\d+)\)=(-?\d+)$', seg)
        if g: a, b, c = map(int, g.groups()); return gcd(abs(a), abs(b)) == c
        ab = re.search(r'\|(-?\d+)-(-?\d+)\|=(-?\d+)$', seg)
        if ab: a, b, c = map(int, ab.groups()); return abs(a - b) == c
        return None
    a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    return {'+': a+b, '-': a-b, '×': a*b, '//': (a//b if b else None),
            'mod': (a % b if b else None), 'XOR': a ^ b, 'AND': a & b, 'OR': a | b}[op] == c


def main():
    conn = sqlite3.connect(DB)
    # verify an arbitrary CoT column (default cot_explain). When verifying a
    # non-default field (e.g. cot_c3), only consider rows where it is non-null,
    # and verify that column AS the explain text; cot_short is still checked.
    if FIELD == 'cot_explain':
        rows = conn.execute("""SELECT s.id,c.category,s.cot_explain,s.cot_short,s.rule,p.prompt,p.answer
            FROM solutions s JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id""").fetchall()
    else:
        rows = conn.execute(f"""SELECT s.id,c.category,s.{FIELD},s.cot_short,s.rule,p.prompt,p.answer
            FROM solutions s JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id
            WHERE s.{FIELD} IS NOT NULL""").fetchall()
    print(f"(verifying field: {FIELD})")
    box_bad = []; phrase_bad = []; struct_bad = []
    arith_bad = []; bitfit_bad = []
    sym_ck = num_ck = bit_ck = 0
    for pid, cat, e, sh, rule, prompt, ans in rows:
        e = e or ''; sh = sh or ''
        # #1
        m = re.search(r'\\boxed\{(.+)\}', e)
        if (not m or m.group(1) != ans) and '{' not in ans and '}' not in ans:
            box_bad.append((pid, cat))
        # #3
        for ph in FORBID:
            if ph in e or ph in sh: phrase_bad.append((pid, cat, ph)); break
        # #4 — opening reasoning anchor: 'Find:' (v1/v2) OR 'Type:' (v3-C3 router)
        if ('Find:' not in e and 'Type:' not in e) or not e.rstrip().endswith('}'):
            struct_bad.append((pid, cat))
        # #2
        if cat == 'symbol_cipher':
            mm = re.search(r';\s*([^;]+?)\s*\((?:' + OPN + r')\)', e.split('Apply:')[-1])
            if mm:
                sym_ck += 1
                if _sym_ok(mm.group(1)) is False: arith_bad.append((pid, cat, mm.group(1)[:50]))
        elif cat == 'numeric':
            for mm in re.finditer(r'(\d+(?:\.\d+)?)\s*(×|\*|//| mod |\+)\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', e):
                a, op, b, c = mm.group(1), mm.group(2).strip(), mm.group(3), mm.group(4)
                dec = len(c.split('.')[-1]) if '.' in c else 0
                fa, fb = float(a), float(b)
                if op in ('×', '*'): r = fa * fb
                elif op == '+': r = fa + fb
                elif op == '//': r = (int(fa) // int(fb)) if fb else None
                elif op == 'mod': r = (int(fa) % int(fb)) if fb else None
                else: r = None
                if r is None: continue
                num_ck += 1
                if f"{round(r, dec):.{dec}f}" != c: arith_bad.append((pid, cat, mm.group(0)))
        elif cat == 'bit' and rule and rule.startswith('bit'):
            exs, q = _bit_examples(prompt)
            try:
                bad = any(_bit_apply(rule, i) != o for i, o in exs)
                if q and _bit_apply(rule, q) != ans: bad = True
                bit_ck += 1
                if bad: bitfit_bad.append((pid, cat))
            except Exception as ex:
                bitfit_bad.append((pid, f"{cat}:unparse:{ex}"))

    print(f"=== verify_cot.py — {len(rows)} solutions ===")
    print(f"#1 boxed≠answer (non-brace) : {len(box_bad)}" + (" ✓" if not box_bad else " ✗"))
    print(f"#2 arithmetic wrong         : {len(arith_bad)}  (symbol {sym_ck}, numeric {num_ck} checked)" + (" ✓" if not arith_bad else " ✗"))
    print(f"   bit rules NOT fitting ex  : {len(bitfit_bad)}  ({bit_ck} bit rules re-executed)" + (" ✓" if not bitfit_bad else " ✗"))
    print(f"#3 forbidden phrase         : {len(phrase_bad)}" + (" ✓" if not phrase_bad else " ✗"))
    print(f"#4 missing Find:/boxed-end  : {len(struct_bad)}" + (" ✓" if not struct_bad else " ✗"))
    for label, lst in [('box', box_bad), ('arith', arith_bad), ('bitfit', bitfit_bad), ('phrase', phrase_bad), ('struct', struct_bad)]:
        for item in lst[:SHOW]:
            print(f"   {label}: {item}")
    clean = not any([box_bad, arith_bad, bitfit_bad, phrase_bad, struct_bad])
    print("\nALL CLEAN ✓" if clean else "\nISSUES FOUND ✗")
    sys.exit(0 if clean else 1)


if __name__ == '__main__':
    main()
