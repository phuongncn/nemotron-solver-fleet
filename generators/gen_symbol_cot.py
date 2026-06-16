#!/usr/bin/env python3
"""Present-tense system-mine CoT for symbol_cipher — SINGLE SOURCE OF TRUTH.

Both the verify lines AND the apply step are computed from ONE authoritative
cipher-parameter set `sp` (base, endian, output-reverse, offset, symbol->digit
mapping, per-operator ops), obtained via the C solver / rule+cot fallback /
py-solve and GATED by gen_cot_symbol.make_trace (verify_examples + query
reproduce exactly). This fixes the "verify rule X, apply rule Y" antipattern
(train-team audit 2026-06-01, discuss/check-data-symbol.md):
  - the op shown verifying an operator is EXACTLY the op used to apply it,
  - every shown computation is real (gen_cot_symbol.compute), no empty "All match",
  - failures are real wrong (base/endian/op) configs computed on the examples,
  - a puzzle with no gated sp is DROPPED, never fabricated.

Usage:
  .venv/bin/python tools/gen_symbol_cot_v3.py -n 3
  .venv/bin/python tools/gen_symbol_cot_v3.py --id 0d90736f
  .venv/bin/python tools/gen_symbol_cot_v3.py --apply        # overwrite all system-mine symbol
"""
import sqlite3, sys, os, re, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_cot_symbol as G
from kaggle_metric import extract_final_answer, verify

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nemotron.db')
SAVE_MODEL = 'system-mine'
SOLVER_OUT = '/tmp/sym_out.txt'

ARITH = ['+', '-', 'r', '*', '%', 'M', '/', 'D', 'A', 'X', '&', 'O', 'G']

OPENERS = [
    "I see symbol equations with operator '{op}'. The symbols must be digits in some base — I need the base, endianness, and the operation.",
    "These are symbol equations using '{op}'. Let me find the symbol-digit map, the base, the endianness, and what '{op}' does.",
    "Symbol cipher with operator '{op}'. I work out the digit mapping, base, endianness, then the arithmetic.",
]
TRY = ["I try", "Let me try", "Testing", "Trying", "What about", "How about"]
FAIL = ["doesn't match.", "no.", "wrong.", "nope.", "fails.", "not this."]
PASS = ["ok", "matches", "checks out"]
SUCCESS = ["Every example reproduces — that's the cipher!", "All examples check out — found it!",
           "Confirmed on all examples!", "All examples verify. That's the rule!"]


def load_solver():
    sols = {}
    if os.path.exists(SOLVER_OUT):
        for line in open(SOLVER_OUT):
            p = G.parse_solver_line(line)
            if p:
                sols[p['id']] = p
    return sols


def complete_ops(sp, exs, query, answer):
    """Fill sp['ops'] for EVERY operator from a single source (this sp's
    base/endian/mapping). For each operator-symbol pick the op that reproduces
    ALL its examples; for the QUERY operator additionally require it reproduces
    the answer (disambiguates coincidences like subtract==mod — fix #3). Returns
    sp with complete ops, or None if any operator can't be resolved."""
    mapping, base, is_le, rev, offset = (sp['mapping'], sp['base'], sp['is_le'], sp['rev'], sp['offset'])
    inv = G.build_inv(mapping)
    ops = dict(sp.get('ops') or {})
    qop = query[2]
    op_chars = {lhs[2] for lhs, _ in exs} | {qop}
    for oc in op_chars:
        existing = ops.get(oc)
        if existing in ('C', 'R'):
            continue  # concat handled by make_trace path
        ex_oc = [(lhs, out) for lhs, out in exs if lhs[2] == oc]
        cand_order = ([existing] if existing in ARITH else []) + [o for o in ARITH if o != existing]
        chosen = None
        for cand in cand_order:
            ok = True
            for lhs, out in ex_oc:
                l0, l1, _o, r0, r1 = G.split_eq(lhs)
                if any(c not in mapping or mapping[c] < 0 for c in (l0, l1, r0, r1)):
                    ok = False; break
                lv = G.to_val(l0, l1, mapping, base, is_le); rv = G.to_val(r0, r1, mapping, base, is_le)
                if G.encode_symbols(G.compute(lv, rv, cand), base, is_le, len(out), rev, offset, inv, out) != out:
                    ok = False; break
            if not ok:
                continue
            if oc == qop:
                l0, l1, _o, r0, r1 = G.split_eq(query)
                if any(c not in mapping or mapping[c] < 0 for c in (l0, l1, r0, r1)):
                    continue
                lv = G.to_val(l0, l1, mapping, base, is_le); rv = G.to_val(r0, r1, mapping, base, is_le)
                if G.encode_symbols(G.compute(lv, rv, cand), base, is_le, len(answer), rev, offset, inv, answer) != answer:
                    continue
            chosen = cand; break
        if chosen is None:
            if oc == qop:
                return None          # query op MUST resolve (it drives the answer)
            ops.pop(oc, None)        # other op unresolvable under this sp -> don't claim it
            continue
        ops[oc] = chosen
    sp = dict(sp); sp['ops'] = ops
    return sp


def get_sp(pid, rule, cot_explain, prompt, answer, sols, provided=None):
    """Authoritative sp gated by make_trace (verify_examples + query reproduce).

    `provided` (e.g. the sp a synthetic puzzle was ENCODED with, stored in rule_params) is
    tried FIRST — instant, no solving — and still gated by make_trace so a bad sp can't slip in.
    """
    exs, query = G.parse_puzzle(prompt, answer)
    if query is None:
        return None, exs, query
    # provided sp first (free), then cheap tiers (solver-out, rule+cot), py_solve only if all miss
    def cand_gen():
        if provided:
            yield provided
        yield sols.get(pid)
        yield G.build_sp_from_db(pid, rule, cot_explain, exs, query)
        if not (rule and rule.startswith('positional_perm')):
            try:
                yield G.py_solve_auto(exs, query, answer, rule)   # slow — last resort
            except Exception:
                yield None
    for sp in cand_gen():
        if not sp:
            continue
        sp = dict(sp); sp['id'] = pid
        try:
            _, _, ok, _ = G.make_trace(pid, prompt, answer, sp)
        except Exception:
            ok = False
        if not ok:
            continue
        if sp.get('q_concat') or sp.get('ops', {}).get(query[2]) in ('C', 'R'):
            return sp, exs, query
        full = complete_ops(sp, exs, query, answer)
        if full is not None:
            full['id'] = pid
            return full, exs, query
    return None, exs, query


def _dec(c0, c1, mapping, base, is_le):
    """(text, value) for decoding a 2-symbol operand — real."""
    d0, d1 = mapping[c0], mapping[c1]
    order = f"{d0}+{d1}×{base}" if is_le else f"{d0}×{base}+{d1}"
    val = G.to_val(c0, c1, mapping, base, is_le)
    return f"{c0}{c1}={d0},{d1} ({order}={val})", val


def encode_explain(val, base, is_le, olen, rev, offset, inv, out):
    """HC2 fix: show the ENCODE step (value -> digits -> symbols), not just assert the result.
    Mirrors G.encode_digits/encode_symbols exactly so the shown work == the real encoding."""
    v = val + offset
    pre = f"{val}{offset:+d}={v}; " if offset else ""
    neg = v < 0 and olen >= 2
    mag = -v if neg else v
    body_len = olen - 1 if neg else olen
    # LSB-first digit extraction (real % / // steps) of the MAGNITUDE
    if mag == 0:
        lsb, steps = [0], [f"0%{base}=0"]
    else:
        lsb, steps, t = [], [], mag
        while t > 0:
            steps.append(f"{t}%{base}={t % base}"); lsb.append(t % base); t //= base
    d = (lsb + [0] * (body_len - len(lsb))) if is_le else ([0] * (body_len - len(lsb[::-1])) + lsb[::-1])
    order = "units first" if is_le else "MSB first"
    revtxt = ""
    if rev:
        d = d[::-1]; revtxt = ", then reverse digits"
    body_syms = "".join(inv[x] for x in d)
    if neg:
        # sign placement: SUFFIX-neg (operator char appended) vs PREFIX-sign, matched against target ends
        if out[:-1] == body_syms:
            pre += f"negative -> magnitude {mag}, then sign symbol '{out[-1]}' appended as suffix; "
            syms = body_syms + out[-1]
        else:
            pre += f"negative -> sign '{out[0]}' + magnitude {mag}; "
            syms = out[0] + body_syms
    else:
        syms = body_syms
    return f"{pre}{', '.join(steps)} -> digits {','.join(map(str, d))} ({order}{revtxt}) -> {syms}"


def gen_failures(exs, sp, query_op, rng):
    """Real failed attempts: wrong op (same base/endian) + endian flip, on an
    example that uses the query operator (else the first arithmetic example)."""
    mapping, base, is_le, rev, offset, ops = (sp['mapping'], sp['base'], sp['is_le'],
                                              sp['rev'], sp['offset'], sp['ops'])
    inv = G.build_inv(mapping)
    correct = ops.get(query_op)
    # pick a reference example for the query operator (or any arithmetic one)
    ref = None
    for lhs, out in exs:
        l0, l1, o, r0, r1 = G.split_eq(lhs)
        if all(c in mapping and mapping[c] >= 0 for c in (l0, l1, r0, r1)) and ops.get(o) not in ('C', 'R', None, '?'):
            if o == query_op:
                ref = (lhs, out, o); break
            if ref is None:
                ref = (lhs, out, o)
    if ref is None:
        return []
    lhs, out, o = ref
    l0, l1, _o, r0, r1 = G.split_eq(lhs)
    end = 'little-endian' if is_le else 'big-endian'
    lines = []

    def attempt(b, le, optag, desc):
        try:
            lv = G.to_val(l0, l1, mapping, b, le); rv = G.to_val(r0, r1, mapping, b, le)
            val = G.compute(lv, rv, optag)
            got = G.encode_symbols(val, b, le, len(out), rev, offset, G.build_inv(mapping), out)
        except Exception:
            got = None
        shown = got if got is not None else "can't encode"
        if got == out:
            return None  # accidentally fits — skip (keep honest)
        return f"{rng.choice(TRY)} base-{b} {desc}: {lhs} = {shown}, expected {out} — {rng.choice(FAIL)}"

    # wrong operations, same base/endian
    wrong_ops = [op for op in ['+', '-', '*', 'X', '%', 'A', 'r'] if op != ops.get(o)][:3]
    for wo in wrong_ops:
        ln = attempt(base, is_le, wo, f"{end} {G.OPCODE2HUMAN.get(wo, wo)}")
        if ln:
            lines.append(ln)
    # wrong endian, correct op
    ln = attempt(base, not is_le, ops.get(o), f"{'big-endian' if is_le else 'little-endian'} {G.OPCODE2HUMAN.get(ops.get(o), '?')}")
    if ln:
        lines.append(ln)
    # wrong base
    for wb in (base + 1, base - 1, 16):
        if wb == base or wb < 2:
            continue
        ln = attempt(wb, is_le, ops.get(o), f"{end} {G.OPCODE2HUMAN.get(ops.get(o), '?')}")
        if ln:
            lines.append(ln); break
    return lines[:5]


def gen_cot(pid, prompt, answer, sp, exs, query, rng):
    """Build present-tense CoT entirely from sp. Returns str or None (drop)."""
    mapping, base, is_le, rev, offset, ops = (sp['mapping'], sp['base'], sp['is_le'],
                                              sp['rev'], sp['offset'], sp['ops'])
    inv = G.build_inv(mapping)
    l0, l1, op, r0, r1 = G.split_eq(query)
    ot = ops.get(op)
    endian = 'little-endian' if is_le else 'big-endian'

    # concat / permutation path: make_trace is already faithful & single-source;
    # strip arrows (-> / →) for token efficiency.
    if sp.get('q_concat') or ot in ('C', 'R'):
        e, _, ok, _ = G.make_trace(pid, prompt, answer, sp)
        if not ok:
            return None
        return e.replace(' -> ', ' = ').replace('->', '=').replace('→', '=')

    if ot in (None, '?'):
        return None

    # Is the query operator itself demonstrated by an example? (drives framing)
    qop_in_examples = any(
        lhs[2] == op and ops.get(lhs[2]) not in ('C', 'R', None, '?')
        and all(c in mapping and mapping[c] >= 0 for c in (lhs[0], lhs[1], lhs[3], lhs[4]))
        for lhs, out in exs)

    cot = [rng.choice(OPENERS).format(op=op), '']
    map_items = sorted(((d, s) for s, d in mapping.items() if d >= 0))
    cot.append("From the examples each symbol is a digit: " + ", ".join(f"{s}={d}" for d, s in map_items) + ".")
    cot.append('')

    fails = gen_failures(exs, sp, op, rng)
    cot += fails
    if fails:
        cot.append('')

    extra = (f", then reverse the output digits" if rev else "") + (f", offset {offset:+d}" if offset else "")
    cot.append(f"The cipher is base-{base} {endian}{extra}; each operator has its own arithmetic. Reproducing the examples:")
    shown = 0
    for elhs, eout in exs:
        el0, el1, eop, er0, er1 = G.split_eq(elhs)
        eot = ops.get(eop)
        if eot in ('C', 'R', None, '?'):
            continue
        if any(c not in mapping or mapping[c] < 0 for c in (el0, el1, er0, er1)):
            continue
        elv = G.to_val(el0, el1, mapping, base, is_le)
        erv = G.to_val(er0, er1, mapping, base, is_le)
        evv = G.compute(elv, erv, eot)
        egot = G.encode_symbols(evv, base, is_le, len(eout), rev, offset, inv, eout)
        if egot != eout:
            return None  # gate: make_trace passed, so this should hold
        t0, _ = _dec(el0, el1, mapping, base, is_le)
        t1, _ = _dec(er0, er1, mapping, base, is_le)
        enc_ex = encode_explain(evv, base, is_le, len(eout), rev, offset, inv, eout)
        cot.append(f"  {elhs} = {eout}: {t0}, {t1}, {G.arith_expr(elv, erv, eot, evv)} ({G.OPCODE2HUMAN[eot]}); encode {enc_ex} {rng.choice(PASS)}")
        shown += 1
    if shown == 0:
        return None  # nothing verifiable shown — never emit an empty verify block

    for c in (l0, l1, r0, r1):
        if c not in mapping or mapping[c] < 0:
            return None
    lv = G.to_val(l0, l1, mapping, base, is_le)
    rv = G.to_val(r0, r1, mapping, base, is_le)
    val = G.compute(lv, rv, ot)
    got = G.encode_symbols(val, base, is_le, len(answer), rev, offset, inv, answer)
    if got != answer:
        return None
    t0, _ = _dec(l0, l1, mapping, base, is_le)
    t1, _ = _dec(r0, r1, mapping, base, is_le)
    raw = val + offset
    pad = 'trailing zeros' if is_le else 'leading zeros'
    note = (f", offset {offset:+d}={raw}" if offset else "") + (", reverse digits" if rev else "")
    olen = len(answer)

    if qop_in_examples:
        cot.append(rng.choice(SUCCESS))
        cot.append('')
        enc_q = encode_explain(val, base, is_le, olen, rev, offset, inv, answer)
        cot.append(f"Apply to {query}: {t0}, {t1}, {G.arith_expr(lv, rv, ot, val)} ({G.OPCODE2HUMAN[ot]}); "
                   f"encode to {olen} symbol(s) base-{base} {endian}: {enc_q} = {got}.")
    else:
        # query operator never appears in the examples — INFER it (best-guess):
        # the cipher (base/endian/mapping) is confirmed above; try the cipher's
        # operations on the query and keep the one that yields a valid encoding.
        cot.append(f"The base, endianness and digit mapping check out. But operator '{op}' "
                   f"appears only in the query, so I infer its operation by trying the "
                   f"cipher's operations on {query} ({t0}, {t1}):")
        rejected = 0
        for cand in ARITH:
            if cand == ot or rejected >= 2:
                continue
            cv = G.compute(lv, rv, cand)
            enc = G.encode_symbols(cv, base, is_le, olen, rev, offset, inv, answer)
            if enc is None:  # only show clearly-invalid candidates (no answer peeking)
                cot.append(f"  {G.OPCODE2HUMAN[cand]}: {G.arith_expr(lv, rv, cand, cv)} — can't encode a valid {olen}-symbol result, reject.")
                rejected += 1
        cot.append(f"  {G.OPCODE2HUMAN[ot]}: {G.arith_expr(lv, rv, ot, val)} — encode {encode_explain(val, base, is_le, olen, rev, offset, inv, answer)} = {got}, a valid {olen}-symbol result. So '{op}' = {G.OPCODE2HUMAN[ot]}.")
    cot.append(f"\\boxed{{{answer}}}")
    return '\n'.join(cot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id')
    ap.add_argument('-n', type=int, default=3)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--from-synth', metavar='SOURCE',
                    help="generate CoT for synthetic_puzzles WHERE source=SOURCE (e.g. 'trained'); "
                         "writes base_answers with source='synthetic', harvest_src='synth-symbol-<SOURCE>'")
    args = ap.parse_args()

    db = sqlite3.connect(DB, timeout=60); db.row_factory = sqlite3.Row
    sols = load_solver()
    synth = bool(args.from_synth)
    ba_source = 'synthetic' if synth else 'real'
    ba_hsrc = f'synth-symbol-{args.from_synth}' if synth else None
    if synth:
        # synthetic puzzles: re-derive sp from the puzzle's OWN examples (get_sp/py_solve_auto);
        # no real rule/cot needed. The same honest gen_cot path + boxed==answer gate applies.
        q = ("SELECT id, '' AS rule, '' AS cot_explain, prompt, answer, rule_params FROM synthetic_puzzles "
             "WHERE category='symbol_cipher' AND source=?")
        rows = (db.execute(q, (args.from_synth,)).fetchall() if args.apply
                else db.execute(q + " ORDER BY RANDOM() LIMIT ?", (args.from_synth, args.n)).fetchall())
    elif args.id:
        rows = db.execute("""SELECT p.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM puzzles p JOIN solutions s ON p.id=s.id JOIN classifications c ON p.id=c.id
            WHERE c.category='symbol_cipher' AND p.id LIKE ?||'%'""", (args.id,)).fetchall()
    elif args.apply:
        rows = db.execute("""SELECT p.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM puzzles p JOIN solutions s ON p.id=s.id JOIN classifications c ON p.id=c.id
            WHERE c.category='symbol_cipher'
            AND p.id IN (SELECT puzzle_id FROM base_answers WHERE model='system-mine')""").fetchall()
    else:
        rows = db.execute("""SELECT p.id, s.rule, s.cot_explain, p.prompt, p.answer
            FROM puzzles p JOIN solutions s ON p.id=s.id JOIN classifications c ON p.id=c.id
            WHERE c.category='symbol_cipher' ORDER BY RANDOM() LIMIT ?""", (args.n,)).fetchall()

    ok = fail = drop = 0
    dropped = []
    import json as _json
    for r in rows:
        provided = None
        if synth and r['rule_params']:
            try:
                d = _json.loads(r['rule_params'])
                if d.get('mapping') and d.get('ops'):       # full sp stored at gen time
                    provided = {'mapping': d['mapping'], 'base': d['base'], 'is_le': d['is_le'],
                                'rev': d['rev'], 'offset': d['offset'], 'ops': d['ops']}
            except Exception:
                provided = None
        if provided is None and not synth:                  # real: try the stored solved-map (e.g. dc-resolve
            m = db.execute("SELECT base,is_le,rev,offset,mapping,ops,q_concat "  # missing-op sps the rule-parser misses)
                           "FROM symbol_solved_map WHERE puzzle_id=?", (r['id'],)).fetchone()
            if m:
                try:
                    sp_m = {'base': m['base'], 'is_le': m['is_le'], 'rev': m['rev'], 'offset': m['offset'],
                            'mapping': _json.loads(m['mapping']), 'ops': _json.loads(m['ops'])}
                    if m['q_concat'] is not None:
                        sp_m['q_concat'] = m['q_concat']
                    provided = sp_m                          # get_sp still gates it via make_trace
                except Exception:
                    provided = None
        sp, exs, query = get_sp(r['id'], r['rule'], r['cot_explain'], r['prompt'], r['answer'], sols, provided=provided)
        cot = gen_cot(r['id'], r['prompt'], r['answer'], sp, exs, query, random.Random(hash(r['id']) & 0x7fffffff)) if sp else None
        if not cot:
            drop += 1; dropped.append(r['id'][:8])
            if not args.apply:
                print(f"=== {r['id'][:8]} | DROP (no gated sp) | rule={r['rule']} ===")
            continue
        pred = extract_final_answer(cot)
        if not verify(str(r['answer']).strip(), pred):
            fail += 1; print(f"  {r['id'][:8]} REJECT box!=ans"); continue
        ok += 1
        if not args.apply:
            print(f"=== {r['id'][:8]} | ans={r['answer']!r} | OK ===")
            print(cot); print(f"--- {len(cot)} chars ---\n")
        else:
            db.execute("""INSERT OR REPLACE INTO base_answers
                (puzzle_id, model, source, expected, predicted, correct, content, train,
                 tokens, content_version, status, updated_at, harvest_src)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, 'v3', 'correct',
                 strftime('%Y-%m-%d %H:%M:%S','now'), ?)""",
                (r['id'], SAVE_MODEL, ba_source, r['answer'], pred, cot, cot, len(cot)//4, ba_hsrc))
            db.commit()
            print(f"  {r['id'][:8]} OK {len(cot):5d}ch", flush=True)
    print(f"\nDone: ok={ok}, fail={fail}, drop={drop}")
    if dropped:
        print("dropped:", dropped[:30])


if __name__ == '__main__':
    main()
