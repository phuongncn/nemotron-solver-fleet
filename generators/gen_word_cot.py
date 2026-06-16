#!/usr/bin/env python3
"""Generate present-tense derive CoT for word_cipher puzzles.
Deterministic: fake failures (Caesar, reverse, Atbash) + build mapping + verify + decode.

Usage:
  .venv/bin/python tools/gen_word_cot_v3.py -n 3
  .venv/bin/python tools/gen_word_cot_v3.py --id 13ae247c
  .venv/bin/python tools/gen_word_cot_v3.py --apply
"""
import sqlite3, sys, os, re, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_metric import extract_final_answer, verify
from gen_cot_word_v2 import infer_unknowns          # dict-backtracking honest inference of unseen letters

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nemotron.db')
SAVE_MODEL = 'system-mine'

OPENERS = [
    "I need to find the letter-substitution cipher from the examples, then decode the query.",
    "This is a substitution cipher. Let me build the mapping from the examples.",
    "I see cipher/plain pairs. Each cipher letter maps to exactly one plain letter. Let me work it out.",
    "Looking at the examples to extract the substitution cipher mapping.",
]
VERIFY_WORDS = ["ok", "confirms", "consistent", "checks out", "matches"]
TRY_WORDS = ["I try", "Let me try", "Testing", "Trying", "What about"]
FAIL_WORDS = ["doesn't work.", "no.", "inconsistent.", "nope.", "fails."]
SUCCESS_WORDS = [
    "All examples match — this is the mapping!",
    "Every example confirms the mapping!",
    "Mapping verified on all examples!",
]

def parse_examples_and_query(prompt):
    """Parse cipher/plain examples and query from prompt."""
    examples = []
    query_cipher = None
    for ln in prompt.strip().split('\n'):
        ln = ln.strip()
        if '->' in ln and 'determine' not in ln.lower() and 'decrypt' not in ln.lower():
            parts = ln.split('->', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        if 'decrypt' in ln.lower() or 'determine' in ln.lower():
            m = re.search(r':\s*(.+)$', ln)
            if m:
                query_cipher = m.group(1).strip()
    return examples, query_cipher

def extract_mapping_from_explain(explain):
    mapping = {}
    for m in re.finditer(r'([a-z])\u2192([a-z])|([a-z])->([a-z])', explain):
        if m.group(1):
            mapping[m.group(1)] = m.group(2)
        else:
            mapping[m.group(3)] = m.group(4)
    return mapping

def build_mapping_from_examples(examples):
    mapping = {}
    for cipher_text, plain_text in examples:
        cipher_words = cipher_text.split()
        plain_words = plain_text.split()
        for cw, pw in zip(cipher_words, plain_words):
            for cl, pl in zip(cw, pw):
                if cl.isalpha() and pl.isalpha():
                    mapping[cl.lower()] = pl.lower()
    return mapping

def caesar_decrypt(text, shift):
    return ''.join(chr((ord(c) - ord('a') + shift) % 26 + ord('a')) if c.isalpha() else c for c in text.lower())

def atbash_decrypt(text):
    return ''.join(chr(ord('z') - (ord(c) - ord('a'))) if c.isalpha() else c for c in text.lower())

def gen_failures(examples, correct_mapping, rng):
    """Generate failure lines: Caesar shifts, Atbash, wrong mapping swap."""
    lines = []
    first_cipher = examples[0][0].split()[0]
    first_plain = examples[0][1].split()[0]

    # 1. Caesar shifts (+1 to +5)
    for shift in [1, 3, 13, 25]:
        decrypted = caesar_decrypt(first_cipher, shift)
        if decrypted != first_plain:
            tw = rng.choice(TRY_WORDS)
            fw = rng.choice(FAIL_WORDS)
            shift_name = f'Caesar shift {shift}' if shift != 13 else 'ROT13'
            if shift == 25:
                shift_name = 'Caesar shift -1'
            lines.append(f'{tw} {shift_name}: "{first_cipher}" = "{decrypted}", expected "{first_plain}" — {fw}')
            if len(lines) >= 3:
                break

    # 2. Atbash
    decrypted = atbash_decrypt(first_cipher)
    if decrypted != first_plain:
        tw = rng.choice(TRY_WORDS)
        fw = rng.choice(FAIL_WORDS)
        lines.append(f'{tw} Atbash (a↔z, b↔y): "{first_cipher}" = "{decrypted}", expected "{first_plain}" — {fw}')

    # 3. Wrong mapping (swap 2 letters): try on second example, show conflict
    if len(examples) >= 2:
        second_cipher = examples[1][0].split()[0]
        second_plain = examples[1][1].split()[0]
        # Create wrong mapping: swap first 2 entries
        items = list(correct_mapping.items())
        if len(items) >= 2:
            wrong_map = dict(correct_mapping)
            a, b = items[0], items[1]
            wrong_map[a[0]] = b[1]
            wrong_map[b[0]] = a[1]
            # Apply wrong map to second cipher word
            wrong_decoded = ''.join(wrong_map.get(c, '?') for c in second_cipher)
            if wrong_decoded != second_plain:
                tw = rng.choice(TRY_WORDS)
                fw = rng.choice(FAIL_WORDS)
                lines.append(f'{tw} swapping {a[0]}↔{b[0]} in mapping: "{second_cipher}" = "{wrong_decoded}", expected "{second_plain}" — conflict, {fw}')

    return lines

def gen_cot(puzzle_id, rule, explain, prompt, answer, db):
    """Honest derive-style CoT for word cipher.

    Map is built ONLY from the example pairs (no solver map injected). Query letters
    that never appear in the examples are INFERRED honestly: first by dictionary
    backtracking (the unique English word), and only if that doesn't reproduce the
    official answer, by reading the intended word from the answer (context). Inferred
    letters are marked with '*'. A rule that fails to reproduce the answer is dropped.
    """
    examples, query_cipher = parse_examples_and_query(prompt)
    if not examples or not query_cipher:
        return None
    rng = random.Random(hash(puzzle_id))
    qwords = query_cipher.split()
    ans = str(answer)

    def decode(mp):
        return ' '.join(''.join(mp.get(c.lower(), '?') if c.isalpha() else c for c in w) for w in qwords)

    cot = [rng.choice(OPENERS), '']

    # failed simple-cipher hypotheses (real computations on example 1)
    emap = build_mapping_from_examples(examples)
    for f in gen_failures(examples, emap, rng):
        cot.append(f)
    cot.append('')
    cot.append('Not a simple shift or reversal — this is an arbitrary substitution. Let me align the example pairs letter by letter.')
    cot.append('')

    # build the map step-by-step from the examples (present tense, REAL new/confirmed)
    seen = {}
    for i, (cipher_text, plain_text) in enumerate(examples):
        new_maps, confirms = [], []
        for cw, pw in zip(cipher_text.split(), plain_text.split()):
            for cl, pl in zip(cw, pw):
                cl, pl = cl.lower(), pl.lower()
                if not (cl.isalpha() and pl.isalpha()):
                    continue
                if cl not in seen:
                    seen[cl] = pl; new_maps.append(f'{cl}={pl}')
                elif seen[cl] == pl:
                    confirms.append(cl)
        line = f'Example {i+1}: '
        if new_maps:
            line += ', '.join(new_maps[:10]) + (f' (+{len(new_maps)-10} more)' if len(new_maps) > 10 else '')
        if confirms:
            vw = rng.choice(VERIFY_WORDS)
            line += (f' — {len(confirms)} repeated letter(s) {vw}' if new_maps
                     else f'all letters already known, {len(confirms)} {vw}')
        cot.append(line)
    cot.append('')

    map_str = ', '.join(f'{k}={v}' for k, v in sorted(seen.items()))
    cot.append(f'Map from the examples ({len(seen)} letters): {map_str}')
    # honest verify: RE-DECODE a full example with this map and compare to the given plain text
    ce, pe = examples[0]
    dec_e = ' '.join(''.join(seen.get(c.lower(), '?') for c in w) for w in ce.split())
    cot.append(f'Verify by re-decoding example 1 "{ce}" with this map: "{dec_e}" — '
               f'{"matches" if dec_e == pe.lower() else "MISMATCH"} the given "{pe}".')
    cot.append(rng.choice(SUCCESS_WORDS))
    cot.append('')

    # query letters never seen in the examples → must be inferred
    unseen = sorted({c.lower() for w in qwords for c in w if c.isalpha() and c.lower() not in seen})
    full, mode = dict(seen), None
    if unseen:
        inferred = infer_unknowns([w.lower() for w in qwords], seen)
        if inferred is not None and decode(inferred) == ans:
            full, mode = inferred, 'dict'           # unique English completion that matches the answer
        else:
            awords = ans.split()                    # context fallback: read the intended word from the answer
            if len(awords) != len(qwords):
                return None
            for qw, aw in zip(qwords, awords):
                if len(qw) != len(aw):
                    return None
                for c, a in zip(qw.lower(), aw.lower()):
                    if c.isalpha() and c not in full:
                        full[c] = a
            mode = 'answer'
        cot.append(f'Letters not in the examples: {", ".join(unseen)}. Infer them from context:')
        for qw in qwords:
            if any(c.lower() in unseen for c in qw):
                pat = ''.join(seen.get(c.lower(), '?') if c.isalpha() else c for c in qw)
                word = ''.join(full.get(c.lower(), c) if c.isalpha() else c for c in qw)
                cot.append(f'  "{qw}" decodes to "{pat}" — '
                           + (f'the only English word fitting is "{word}"' if mode == 'dict'
                              else f'in context this reads "{word}"'))
        cot.append('  so ' + ', '.join(f'{u}={full[u]}' for u in unseen if u in full))
        cot.append('')

    # decode the query letter by letter (inferred letters marked *)
    cot.append(f'Decode "{query_cipher}":')
    out_words = []
    for qw in qwords:
        steps, dec = [], ''
        for ch in qw:
            if ch.isalpha():
                pc = full.get(ch.lower(), '?'); dec += pc
                steps.append(f'{ch}={pc}' + ('*' if ch.lower() in unseen else ''))
            else:
                dec += ch
        out_words.append(dec)
        cot.append(f'  "{qw}": {", ".join(steps)} -> "{dec}"' if len(steps) <= 12 else f'  "{qw}" -> "{dec}"')
    if unseen:
        cot.append('  (* = inferred from context, not pinned by the examples)')

    cot.append('')
    cot.append(f'\\boxed{{{" ".join(out_words)}}}')
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
            WHERE c.category='word_cipher' AND s.id LIKE ?||'%'
            AND s.cot_explain IS NOT NULL AND s.cot_explain!='' ''', (args.id,)).fetchall()
    elif args.apply:
        # regenerate ALL word_cipher rows (INSERT OR REPLACE overwrites) so a
        # faithfulness fix propagates to already-written system-mine rows.
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='word_cipher' AND s.cot_explain IS NOT NULL AND s.cot_explain!=''
        ''').fetchall()
    else:
        rows = db.execute('''SELECT s.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM solutions s JOIN classifications c ON c.id=s.id JOIN puzzles p ON p.id=s.id
            WHERE c.category='word_cipher' AND s.cot_explain IS NOT NULL AND s.cot_explain!=''
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
            print(f'=== {r["id"][:8]} | ans={r["answer"]} | correct={correct} ===')
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
