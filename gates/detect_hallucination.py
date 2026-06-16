#!/usr/bin/env python3
"""Hallucination detector — catches fabricated values/operators that pass the
arithmetic+boxed checks but aren't derivable from the problem.

Complements verify_cot.py. The class of bug it targets:
  - 2f7e0e78: CoT did "-1+2=1" (arithmetically correct) but the constant 2 came
    from nowhere — not in the prompt, not a prior result.
  - 134 numeric: the QUERY operator never appears in the examples, so its operation
    is unknowable from a single prompt; the CoT reverse-engineered it from the answer.
    (Verified: no single op fits examples+query, and no GLOBAL operator dictionary
    exists across the dataset, so these are genuinely underdetermined -> answer-only.)

Checks (numeric is the high-risk category for fabricated constants/operators):
  C. query operator absent from the puzzle's own examples  (STRONG hallucination signal)
  A. orphan operand: a number used as an operand in "A op B = C" that is not in the
     prompt, not the answer, and not a previously-stated result  (provenance)

Number matching normalizes leading/trailing zeros (03 == 3, 8.90 == 8.9) and a small
set of structural constants is whitelisted. The puzzle's operator SYMBOL (often '?',
'%', '"', etc.) is domain content, NOT garbage — do not flag it.

Usage: python tools/detect_hallucination.py [--show N]
Exit 0 if clean, 1 if any strong (C) signal found.
"""
import sqlite3, re, sys, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'nemotron.db')
SHOW = int(sys.argv[sys.argv.index('--show')+1]) if '--show' in sys.argv else 12
FIELD = sys.argv[sys.argv.index('--field')+1] if '--field' in sys.argv else 'cot_explain'
STRUCT = {'256', '0', '1', '2', '8', '9', '10', '11', '12', '13', '14', '15', '16', '100'}

def norm(x):
    if '.' in x:
        x = x.rstrip('0').rstrip('.')
        return x or '0'
    return str(int(x)) if x else '0'

def nums(s):
    return set(norm(t) for t in re.findall(r'\d+\.?\d*', s))

def trusted(prompt, ans, cot):
    t = nums(prompt) | nums(ans)
    for m in re.finditer(r'(?:=|≈|->|→|:)\s*-?(\d+\.?\d*)', cot):
        t.add(norm(m.group(1)))
    return t | STRUCT

def query_op_absent(prompt):
    """For multi-operator 'equation' puzzles: is the query's operator symbol
    missing from every example line? Returns (absent_bool, op, example_ops)."""
    lines = [l.strip() for l in prompt.splitlines()]
    ql = [l for l in lines if l.lower().startswith('now')]
    if not ql:
        return False, None, set()
    qm = re.search(r'(\d+)\s*([^\s\d=])\s*(\d+)', ql[0].replace('determine the result for:', ''))
    if not qm:
        return False, None, set()
    op = qm.group(2); ex = set()
    for l in lines:
        if l.lower().startswith('now'):
            continue
        em = re.match(r'\d+\s*([^\s\d=])\s*\d+\s*=', l)
        if em:
            ex.add(em.group(1))
    return (bool(ex) and op not in ex), op, ex


def main():
    conn = sqlite3.connect(DB)
    rows = conn.execute("""SELECT s.id,c.category,s.cot_explain,p.prompt,p.answer,s.rule
        FROM solutions s JOIN classifications c ON s.id=c.id JOIN puzzles p ON s.id=p.id
        WHERE c.category='numeric'""").fetchall()
    op_absent = []; orphan = []
    for pid, cat, cot, prompt, ans, rule in rows:
        cot = cot or ''
        ab, op, ex = query_op_absent(prompt)
        if ab:
            op_absent.append((pid, op, sorted(ex), ans))
        # Regression puzzles (Multiply/Quadratic/Linear) legitimately introduce
        # a DERIVED coefficient (factor ≈ out/in) and t² (=t·t) — not present in
        # the prompt by construction. verify_cot already proves every regression
        # arithmetic line is exact, so the orphan-operand provenance check (built
        # for eq-type fabricated operands) does not apply. Skip it for them.
        if rule and rule.split(':')[0].split()[0] in ('Multiply', 'Quadratic', 'Linear'):
            continue
        t = trusted(prompt, ans, cot)
        for m in re.finditer(r'(?<![\d.])(\d+\.?\d*)\s*[+\-×*/]\s*(\d+\.?\d*)\s*=', cot):
            if norm(m.group(1)) not in t or norm(m.group(2)) not in t:
                orphan.append((pid, m.group(0)[:30])); break

    print(f"=== detect_hallucination.py — {len(rows)} numeric solutions ===")
    print(f"C. query operator ABSENT from examples (fabricated op): {len(op_absent)}"
          + (" ✓" if not op_absent else " ✗ -> these should be answer-only"))
    for x in op_absent[:SHOW]:
        print(f"   {x}")
    print(f"A. orphan operand constants (provenance, normalized): {len(orphan)}"
          + (" ✓" if not orphan else "  (review; leading-zero/operator-symbol cases are false positives)"))
    for x in orphan[:SHOW]:
        print(f"   {x}")
    sys.exit(0 if not op_absent else 1)


if __name__ == '__main__':
    main()
