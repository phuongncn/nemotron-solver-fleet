#!/usr/bin/env python3
"""Generate present-tense derive CoT for bit puzzles, system-mine style.

FAITHFUL by construction: the actual rule is RE-EXECUTED (verify_cot._bit_apply
for per-bit rules; native executors for neighbour/rotate/affine) on every
example AND the query — both the "verify" lines and the "Apply" step are
recomputed here, never copied from cot_explain (v2 template). A rule that does
not reproduce all examples + the answer is DROPPED, never papered over with a
"(verified by solver)" placeholder (the old v3 bug — audit 2026-06-01).

Failures are real: common transforms (rotate/reverse/NOT/shift) computed on the
examples; only ones that genuinely fail are shown.

Usage:
  .venv/bin/python tools/gen_bit_cot_v3.py -n 3
  .venv/bin/python tools/gen_bit_cot_v3.py --id 0a1326f4
  .venv/bin/python tools/gen_bit_cot_v3.py --apply   # regenerate ALL (overwrite)
"""
import sqlite3, sys, os, re, argparse, random, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_metric import extract_final_answer, verify
from verify_cot import _bit_apply, _bit_term            # faithful per-bit executor
from gen_cot_bit_v2 import _readable, _bit_rule_terms     # readable bit-term form

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nemotron.db')
SAVE_MODEL = 'system-mine'

# === generic transforms for FAILURES (real computations) ===
def rotate_right(b, n): return b[-n:] + b[:-n]
def rotate_left(b, n):  return b[n:] + b[:n]
def reverse_bits(b):    return b[::-1]
def not_bits(b):        return ''.join('1' if c == '0' else '0' for c in b)
def shift_right(b, n):  return '0' * n + b[:len(b)-n]
def shift_left(b, n):   return b[n:] + '0' * n

TRANSFORMS = [
    ('rotate right 1', lambda b: rotate_right(b, 1)),
    ('rotate left 1', lambda b: rotate_left(b, 1)),
    ('rotate right 2', lambda b: rotate_right(b, 2)),
    ('reverse all bits', reverse_bits),
    ('NOT (flip all bits)', not_bits),
    ('shift right 1 (zero-fill)', lambda b: shift_right(b, 1)),
    ('shift left 1 (zero-fill)', lambda b: shift_left(b, 1)),
    ('reverse then NOT', lambda b: not_bits(reverse_bits(b))),
]

OPENERS = [
    "I see 8-bit binary transformations. Let me find the rule mapping input to output.",
    "Looking at input/output bit patterns. Let me work out the transformation.",
    "These are 8-bit binary puzzles. I need to determine the bitwise rule.",
    "I see binary strings being transformed. Let me figure out the pattern.",
]
TRY_WORDS = ["I try", "Let me try", "Testing", "Trying", "What about", "How about"]
FAIL_WORDS = ["doesn't match.", "no.", "wrong.", "nope.", "fails.", "not this."]
PASS_WORDS = ["ok", "matches", "checks out"]
SUCCESS_WORDS = [
    "All examples reproduce — this is the rule!",
    "Every example checks out — found it!",
    "Confirmed on all examples!",
    "All examples verify. That's the rule!",
]

def parse_examples(prompt):
    examples = []; query = None
    for ln in prompt.strip().split('\n'):
        ln = ln.strip()
        m = re.match(r'^([01]{8})\s*->\s*([01]{8})$', ln)
        if m:
            examples.append((m.group(1), m.group(2)))
        qm = re.search(r'(?:determine the output for|find|result)[:\s].*?([01]{8})', ln, re.I)
        if qm:
            query = qm.group(1)
    if query is None:  # fallback: last lone 8-bit string not in an example
        for ln in prompt.strip().split('\n'):
            mm = re.findall(r'[01]{8}', ln)
            if 'determine' in ln.lower() and mm:
                query = mm[-1]
    return examples, query

# ── faithful rule executor: returns (out_str, steps:list) or None ────────────

# perbit3 tuple op -> verify_cot bit-term token (indices are single digits 0-7)
_P3 = {'i': lambda d: f"I{d[0]}", 'not': lambda d: f"NOT{d[0]}",
       'and': lambda d: f"AND{d[0]}{d[1]}", 'or': lambda d: f"OR{d[0]}{d[1]}",
       'xor': lambda d: f"XOR{d[0]}{d[1]}", 'nand': lambda d: f"NAND{d[0]}{d[1]}",
       'nor': lambda d: f"NOR{d[0]}{d[1]}", 'andn': lambda d: f"AND-NOT{d[0]}{d[1]}",
       'orn': lambda d: f"OR-NOT{d[0]}{d[1]}", 'xnor': lambda d: f"XNOR{d[0]}{d[1]}",
       'ch': lambda d: f"CH{d[0]}{d[1]}{d[2]}", 'maj': lambda d: f"MAJ{d[0]}{d[1]}{d[2]}"}

def _perbit3_to_rule(rule):
    """Convert 'perbit3: bit0=('I',5,None,None); ...' to standard 'bit0=...; ...'.

    perbit3 is LSB-indexed (index i ↔ string position 7-i) for BOTH the output
    bit and the input references; verify_cot._bit_term is position-indexed, so
    remap i → 7-i on both sides. Returns the normalised rule or None."""
    body = rule.split(':', 1)[1]
    parts = re.findall(r'bit(\d)=\(([^)]*)\)', body)
    if len(parts) != 8:
        return None
    terms = {}
    for j, tup in parts:
        try:
            op, *args = ast.literal_eval('(' + tup + ')')
        except Exception:
            return None
        d = [7 - a for a in args if a is not None]
        fn = _P3.get(str(op).lower())
        if fn is None:
            return None
        terms[7 - int(j)] = fn(d)
    if len(terms) != 8:
        return None
    return '; '.join(f"bit{k}={terms[k]}" for k in range(8))

def exec_rule(rule, bits):
    """Re-execute `rule` on 8-bit string `bits`. (out, steps) or None."""
    rule = (rule or '').strip()

    # perbit3 tuple form -> normalise to standard per-bit, then fall through
    if rule.startswith('perbit3:'):
        norm = _perbit3_to_rule(rule)
        if norm is None:
            return None
        rule = norm

    # per-bit boolean functions: bit0=...; bit1=...
    if rule.startswith('bit'):
        terms = _bit_rule_terms(rule)
        if len(terms) != 8:
            return None
        try:
            out = _bit_apply(rule, bits)
        except Exception:
            return None
        if '?' in out:
            return None
        IN = [int(c) for c in bits]
        steps = [f"bit{j} = {_readable(terms[j])} = {_bit_term(terms[j], IN)}" for j in range(8)]
        return out, steps

    # 3-bit neighbour (cellular) function
    if rule.startswith('3-bit neighbor'):
        m = re.search(r'\{.*\}', rule)
        if not m:
            return None
        table = ast.literal_eval(m.group(0))
        wrap = 'no-wrap' not in rule
        b = [int(c) for c in bits]
        steps = []; out = []
        for i in range(8):
            l = b[(i - 1) % 8] if (wrap or i > 0) else 0
            r = b[(i + 1) % 8] if (wrap or i < 7) else 0
            v = table.get((l, b[i], r))
            if v is None:
                return None
            out.append(str(v)); steps.append(f"bit{i} = f(left={l}, self={b[i]}, right={r}) = {v}")
        return ''.join(out), steps

    # circular rotate
    m = re.search(r'Rotate (left|right)\s+(\d+)', rule)
    if m:
        d, k = m.group(1), int(m.group(2)) % 8
        out = (bits[k:] + bits[:k]) if d == 'left' else (bits[-k:] + bits[:-k] if k else bits)
        if d == 'left':
            steps = [f"rotate left {k}: move first {k} bits to the end — {bits[:k]}|{bits[k:]} = {out}"]
        else:
            steps = [f"rotate right {k}: move last {k} bits to the front — {bits[:8-k]}|{bits[8-k:]} = {out}"]
        return out, steps

    # affine: f(x) = (a*x + b) mod 256
    m = re.search(r'f\(x\)\s*=\s*\((\d+)\*x\s*\+\s*(\d+)\)\s*mod\s*256', rule)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        x = int(bits, 2); s = a * x + b; r = s % 256
        out = f"{r:08b}"
        steps = [f"read {bits} as x={x}", f"{a}×{x}={a*x}", f"+{b}={s}", f"{s} mod 256={r}", f"{r} as 8 bits = {out}"]
        return out, steps

    return None

def describe_rule(rule):
    rule = (rule or '').strip()
    if rule.startswith(('bit', 'perbit3')):
        return "each output bit is a fixed boolean function of the input bits"
    if rule.startswith('3-bit neighbor'):
        edge = "edges padded with 0" if 'no-wrap' in rule else "wrapping at the ends"
        return f"each output bit is a function of its (left, self, right) neighbours ({edge})"
    m = re.search(r'Rotate (left|right)\s+(\d+)', rule)
    if m:
        return f"circular rotate {m.group(1)} by {int(m.group(2)) % 8}"
    if rule.startswith('f(x)'):
        return "read the 8 bits as an integer x, output (a·x+b) mod 256"
    return rule

def gen_failures(examples, correct_out_first, rng):
    """Real failed transforms on examples (skip any that accidentally fit)."""
    lines = []
    for name, tf in TRANSFORMS:
        passes = []; fail = None
        for i, (inp, out) in enumerate(examples):
            res = tf(inp)
            if res == out:
                passes.append(f'ex{i+1}: {inp} = {res} {rng.choice(PASS_WORDS)}')
            else:
                fail = f'ex{i+1}: {inp} = {res}, expected {out}'; break
        if fail is None:
            continue  # fits all — not a failure, skip
        trail = (', '.join(passes) + ', but ' + fail) if passes else fail
        lines.append(f'{rng.choice(TRY_WORDS)} {name}: {trail} — {rng.choice(FAIL_WORDS)}')
        if len(lines) >= 5:
            break
    return lines

def gen_cot(puzzle_id, rule, explain, prompt, answer, db):
    examples, query = parse_examples(prompt)
    if not examples or not query or len(str(answer)) != 8:
        return None

    # GATE: rule must faithfully reproduce ALL examples + the query answer.
    ver = []
    for inp, out in examples:
        r = exec_rule(rule, inp)
        if r is None or r[0] != out:
            return None
        ver.append((inp, out, r))
    rq = exec_rule(rule, query)
    if rq is None or rq[0] != str(answer):
        return None

    rng = random.Random(hash(puzzle_id) & 0x7fffffff)
    cot = [rng.choice(OPENERS), '']

    fails = gen_failures(examples, examples[0][1], rng)
    cot += fails
    if fails:
        cot.append('')

    cot.append(f'The pattern: {describe_rule(rule)}. Verify by RECOMPUTING each example — reject the rule if any example fails:')
    # P1 fix (2026-06-02): verify must RECOMPUTE, never restate the expected output.
    #   per-bit / neighbour: full bit-by-bit derivation on ex1 (anchor the functions), then a
    #     compact computed-value line per remaining example (b0..b7 = the values we computed).
    #   affine / rotate: one-line arithmetic chain per example (real a·x+b mod 256 / rotation).
    perbit_like = rule.strip().startswith(('bit', '3-bit', 'perbit3'))
    for k, (inp, out, r) in enumerate(ver):
        if perbit_like:
            if k == 0:
                cot.append(f'  {inp}:')
                for st in r[1]:
                    cot.append(f'    {st}')
                cot.append(f'    -> {r[0]} (expected {out}) {rng.choice(PASS_WORDS)}')
            else:
                vals = ', '.join(f'b{j}={r[0][j]}' for j in range(8))
                cot.append(f'  {inp}: {vals} -> {r[0]} (expected {out}) {rng.choice(PASS_WORDS)}')
        else:
            cot.append(f'  {inp}: {"; ".join(r[1])} (expected {out}) {rng.choice(PASS_WORDS)}')
    cot.append(rng.choice(SUCCESS_WORDS))
    cot.append('')

    cot.append(f'Apply to {query}:')
    for st in rq[1]:
        cot.append(f'  {st}')
    cot.append(f'  result {rq[0]}')
    cot.append(f'\\boxed{{{answer}}}')
    return '\n'.join(cot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id')
    ap.add_argument('-n', type=int, default=3)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    if args.id:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='bit' AND s.id LIKE ?||'%'
            AND s.rule IS NOT NULL''', (args.id,)).fetchall()
    elif args.apply:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='bit' AND s.rule IS NOT NULL''').fetchall()
    else:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='bit' AND s.rule IS NOT NULL
            ORDER BY RANDOM() LIMIT ?''', (args.n,)).fetchall()

    ok = fail = drop = 0
    for r in rows:
        cot = gen_cot(r['id'], r['rule'], r['cot_explain'], r['prompt'], r['answer'], db)
        if not cot:
            drop += 1
            if not args.apply:
                print(f'=== {r["id"][:8]} | DROP | rule={r["rule"][:40]} ===\n')
            continue
        pred = extract_final_answer(cot)
        correct = verify(str(r['answer']).strip(), pred)
        if not correct:
            fail += 1
            print(f'  {r["id"][:8]} WRONG pred={pred!r} exp={r["answer"]!r}')
            continue
        ok += 1
        if not args.apply:
            print(f'=== {r["id"][:8]} | rule={r["rule"][:40]} | ans={r["answer"]} | OK ===')
            print(cot)
            print(f'--- {len(cot)} chars ---\n')
        else:
            db.execute('''INSERT OR REPLACE INTO base_answers
                (puzzle_id, model, source, expected, predicted, correct, content, train,
                 tokens, content_version, status, updated_at)
                VALUES (?, ?, 'real', ?, ?, 1, ?, ?, ?, 'v3', 'correct',
                 strftime('%Y-%m-%d %H:%M:%S','now'))''',
                (r['id'], SAVE_MODEL, r['answer'], pred, cot, cot, len(cot)//4))
            db.commit()
            print(f'  {r["id"][:8]} OK {len(cot):5d}ch', flush=True)
    print(f'\nDone: ok={ok}, fail={fail}, drop={drop}')


if __name__ == '__main__':
    main()
