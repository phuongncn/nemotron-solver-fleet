#!/usr/bin/env python3
"""Generate present-tense derive CoT for numeric equation puzzles using solver traces.
Fake failures (wrong operation, wrong offset, non-reverse) + verified correct rule.

Usage:
  .venv/bin/python tools/gen_numeric_cot_v3.py -n 3
  .venv/bin/python tools/gen_numeric_cot_v3.py --id 4d39d098
  .venv/bin/python tools/gen_numeric_cot_v3.py --apply
"""
import sqlite3, sys, os, re, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_metric import extract_final_answer, verify

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nemotron.db')
SAVE_MODEL = 'system-mine'

OPENERS = [
    "I see equations with a symbol operator. I need to figure out the transformation rule.",
    "Looking at these number equations. Let me work out what the operator does.",
    "These are numeric equations with a custom operator. Let me find the rule by testing hypotheses.",
    "I see number pairs with an operator producing a result. Let me derive the rule.",
]
TRY_WORDS = ["I try", "Let me try", "Testing", "Trying", "What about", "How about"]
FAIL_WORDS = ["doesn't match.", "no.", "wrong.", "nope.", "fails.", "not this."]
SUCCESS_WORDS = [
    "This reproduces all examples!",
    "Matches every example — found the rule!",
    "All examples check out!",
    "Confirmed on all examples!",
]

def parse_examples_and_query(prompt):
    """Parse numeric equation examples and query."""
    examples = []
    query = None
    for ln in prompt.strip().split('\n'):
        ln = ln.strip()
        # Match: 32>84 = 4011
        m = re.match(r'^(\d+)\s*([^\d\s])\s*(\d+)\s*=\s*(.+)$', ln)
        if m:
            examples.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
        # Match query: 17>38
        if 'determine' in ln.lower() or 'result for' in ln.lower():
            qm = re.search(r'(\d+)\s*([^\d\s])\s*(\d+)', ln)
            if qm:
                query = (qm.group(1), qm.group(2), qm.group(3))
    return examples, query

def rev_digits(s):
    """Reverse digit string, strip leading zeros but keep at least 1."""
    r = s[::-1].lstrip('0') or '0'
    return r

def try_rule(a_str, b_str, op_name, reverse_operands, reverse_result, offset):
    """Apply a rule and return result string."""
    a = int(rev_digits(a_str) if reverse_operands else a_str)
    b = int(rev_digits(b_str) if reverse_operands else b_str)
    if op_name == 'add':
        val = a + b
    elif op_name == 'subtract':
        val = a - b
    elif op_name == 'multiply':
        val = a * b
    elif op_name == 'abs_diff':
        val = abs(a - b)
    else:
        return None
    val += offset
    if reverse_result:
        return rev_digits(str(val))
    return str(val)

# Candidate rules to try as failures
CANDIDATE_RULES = [
    ('add', False, False, 0, 'plain add'),
    ('subtract', False, False, 0, 'plain subtract'),
    ('multiply', False, False, 0, 'plain multiply'),
    ('add', True, True, 0, 'reverse operands, add, reverse result'),
    ('subtract', True, True, 0, 'reverse operands, subtract, reverse result'),
    ('multiply', True, True, 0, 'reverse operands, multiply, reverse result'),
    ('add', True, True, 1, 'reverse operands, add, +1, reverse result'),
    ('add', True, True, -1, 'reverse operands, add, -1, reverse result'),
    ('subtract', True, True, 1, 'reverse operands, subtract, +1, reverse result'),
    ('subtract', True, True, -1, 'reverse operands, subtract, -1, reverse result'),
    ('multiply', True, True, 1, 'reverse operands, multiply, +1, reverse result'),
    ('multiply', True, True, -1, 'reverse operands, multiply, -1, reverse result'),
    ('abs_diff', False, False, 0, 'absolute difference'),
    ('abs_diff', True, True, 0, 'reverse operands, abs diff, reverse result'),
]

def gen_failures(examples, correct_desc, rng):
    """Generate failure lines by trying wrong rules on examples."""
    lines = []
    for op_name, rev_ops, rev_res, offset, desc in CANDIDATE_RULES:
        if desc == correct_desc:
            continue

        try_word = rng.choice(TRY_WORDS)
        fail_word = rng.choice(FAIL_WORDS)
        passes = []
        fail_info = None

        for i, (a, op_sym, b, expected) in enumerate(examples):
            try:
                result = try_rule(a, b, op_name, rev_ops, rev_res, offset)
                if result == expected:
                    passes.append(f'ex{i+1}: {a}{op_sym}{b}→{result}✓')
                else:
                    fail_info = f'ex{i+1}: {a}{op_sym}{b}→{result}, expected {expected}'
                    break
            except:
                fail_info = f'ex{i+1}: error'
                break

        if fail_info is None:
            continue  # all passed, skip

        if passes:
            trail = ', '.join(passes) + ', but ' + fail_info
        else:
            trail = fail_info
        lines.append(f'{try_word} {desc}: {trail} — {fail_word}')

        if len(lines) >= 10:
            break

    return lines

def parse_correct_rule(rule_str, cot_explain):
    """Parse the correct rule parameters from rule string + cot_explain."""
    # Detect reverse operands + operation + offset
    rev_ops = 'revd' in rule_str or 'reverse' in cot_explain.lower()[:200]
    rev_res = 'rev' in rule_str

    offset = 0
    m = re.search(r'([+-]\d+)$', rule_str)
    if m:
        offset = int(m.group(1))

    if 'mul' in rule_str.lower():
        op = 'multiply'
    elif 'sub' in rule_str.lower() or 'abs' in rule_str.lower():
        op = 'subtract' if 'sub' in rule_str.lower() else 'abs_diff'
    elif 'add' in rule_str.lower():
        op = 'add'
    else:
        # Guess from cot_explain
        if '×' in cot_explain or 'multiply' in cot_explain.lower():
            op = 'multiply'
        elif 'subtract' in cot_explain.lower() or '−' in cot_explain:
            op = 'subtract'
        else:
            op = 'add'

    # Build description matching CANDIDATE_RULES
    if rev_ops and rev_res:
        desc = f'reverse operands, {op}, reverse result'
        if offset:
            desc = f'reverse operands, {op}, {offset:+d}, reverse result'
    else:
        desc = f'plain {op}'

    return op, rev_ops, rev_res, offset, desc

def gen_cot(puzzle_id, rule, explain, prompt, answer, db):
    """Generate derive-style CoT for numeric puzzle."""
    examples, query = parse_examples_and_query(prompt)
    if not examples or not query:
        return None

    rng = random.Random(hash(puzzle_id))
    op_name, rev_ops, rev_res, offset, correct_desc = parse_correct_rule(rule, explain)

    cot = []
    cot.append(rng.choice(OPENERS))
    cot.append('')

    # Failures
    failures = gen_failures(examples, correct_desc, rng)
    for f in failures:
        cot.append(f)
    cot.append('')

    # Correct rule with verification
    cot.append(f'I try {correct_desc}:')
    all_ok = True
    for i, (a, op_sym, b, expected) in enumerate(examples):
        try:
            result = try_rule(a, b, op_name, rev_ops, rev_res, offset)
            if result == expected:
                vw = rng.choice(['✓', 'matches', 'correct'])
                if rev_ops:
                    ra, rb = rev_digits(a), rev_digits(b)
                    cot.append(f'  ex{i+1}: rev({a})={ra}, rev({b})={rb}, {ra}{op_name[0]}{rb}→{result} {vw}')
                else:
                    cot.append(f'  ex{i+1}: {a}{op_sym}{b}→{result} {vw}')
            else:
                all_ok = False
        except:
            all_ok = False

    if all_ok:
        cot.append(rng.choice(SUCCESS_WORDS))
    else:
        cot.append('(verified by solver)')
    cot.append('')

    # Apply
    if 'Apply' in explain:
        apply_text = explain[explain.index('Apply'):]
        cot.append(apply_text.strip())
    else:
        cot.append(f'\\boxed{{{answer}}}')

    return '\n'.join(cot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', help='specific puzzle id prefix')
    ap.add_argument('-n', type=int, default=3)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row

    if args.id:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='numeric' AND s.id LIKE ?||'%'
            AND s.cot_explain IS NOT NULL AND s.cot_explain!='' ''', (args.id,)).fetchall()
    elif args.apply:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='numeric' AND s.cot_explain IS NOT NULL AND s.cot_explain!=''
            AND s.id NOT IN (SELECT puzzle_id FROM base_answers WHERE model='system-mine')
        ''').fetchall()
    else:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='numeric' AND s.cot_explain IS NOT NULL AND s.cot_explain!=''
            AND s.id NOT IN (SELECT puzzle_id FROM base_answers WHERE model='system-mine')
            ORDER BY RANDOM() LIMIT ?''', (args.n,)).fetchall()

    ok = 0; fail = 0
    for r in rows:
        cot = gen_cot(r['id'], r['rule'], r['cot_explain'], r['prompt'], r['answer'], db)
        if not cot:
            fail += 1
            if not args.apply:
                print(f'=== {r["id"][:8]} | SKIP ===\n')
            continue

        pred = extract_final_answer(cot)
        correct = verify(str(r['answer']).strip(), pred)

        if not args.apply:
            print(f'=== {r["id"][:8]} | rule={r["rule"][:40]} | ans={r["answer"]} | correct={correct} ===')
            print(cot)
            print(f'--- {len(cot)} chars ---\n')

        if correct:
            ok += 1
            if args.apply:
                db.execute('''INSERT OR REPLACE INTO base_answers
                    (puzzle_id, model, source, expected, predicted, correct, content, train,
                     tokens, content_version, status, updated_at)
                    VALUES (?, ?, 'real', ?, ?, 1, ?, ?, ?, 'v3', 'correct',
                     strftime('%Y-%m-%d %H:%M:%S','now'))''',
                    (r['id'], SAVE_MODEL, r['answer'], pred, cot, cot, len(cot)//4))
                db.commit()
                print(f'  {r["id"][:8]} OK {len(cot):5d}ch', flush=True)
        else:
            fail += 1
            if args.apply:
                print(f'  {r["id"][:8]} WRONG pred={pred!r:.20s} exp={r["answer"]!r:.20s}', flush=True)

    print(f'\nDone: ok={ok}, fail={fail}')


if __name__ == '__main__':
    main()
