"""bit_perbit — each output bit is a function of input bits. Three sub-forms, all in one skeleton:
  - per-bit (G3 family-first column derivation: single-source -> constant -> 2-input boolean)
  - 3-bit neighbor (elementary cellular automaton)
  - compound boolean / cay-bit / perbit3-tuple (functions stored in rule / cot_c3)
Correctness invariants G1 (no rubber-stamp) via Skeleton.confirm; G2 (no blanket fix line);
G3/G4 (minimal-param column vocabulary) via bitlib.derive_column.
"""
import ast
import itertools
import os
import re

from skeleton import Skeleton
import bitlib as B

# v5-improve §2 structural rotation/shift tie-break. OFF by default (v5-perfect renders the plain cfgA
# per-column form). When ON (env V5_BIT_TIEBREAK=1, used for the v5-perfect-bittie tag), a near-fit
# whole-word transform is kept as the BACKBONE: columns it already reproduces stay as the transform,
# only the deviating columns are fixed (anti-bleed: keep rotation structure, NOT a fresh per-column
# search — the v5fix3 mistake). chosen[] functions are IDENTICAL either way -> boxed unchanged; only the
# THEORY-2 narration differs (what the model is taught).
TIEBREAK = os.environ.get("V5_BIT_TIEBREAK") == "1"


def _bits(s):
    return [int(c) for c in s]


def matchcol_j(fn, examples, j):
    """True iff the whole-word transform fn already reproduces output column j on EVERY example."""
    return all(_bits(fn(inp))[j] == int(out[j]) for inp, out in examples)


def _transform_col_expr(fn, j):
    """The exact per-column expression the shift/rotation fn places at column j, by probing fn on the
    zero word and single-bit words: a copy `in{k}`, an inverted `NOT in{k}`, or a constant `0`/`1`.
    Returns None if column j is not single-source under fn (so it can't be honestly called 'the
    transform')."""
    zj = _bits(fn("0" * 8))[j]
    srcs = []
    for k in range(8):
        s = ["0"] * 8
        s[k] = "1"
        if _bits(fn("".join(s)))[j] != zj:
            srcs.append(k)
    if not srcs:
        return str(zj)                                      # constant column (e.g. shift zero-fill)
    if len(srcs) == 1:
        k = srcs[0]
        return f"in{k}" if zj == 0 else f"NOT in{k}"        # zj is the value when in_k=0
    return None


def _col_candidates(j):
    """Per-column boolean candidates in PRIOR order: constants, single-source copy/NOT, the positional
    XOR-offset family (inj XOR in(j+off)), then arbitrary 2-input gates last. Returns [(expr, fn), ...]."""
    cs = [("0", lambda x: 0), ("1", lambda x: 1)]
    for k in range(8):
        cs.append((f"in{k}", (lambda x, k=k: x[k])))
        cs.append((f"NOT in{k}", (lambda x, k=k: 1 - x[k])))
    for off in range(1, 8):
        cs.append((f"in{j} XOR in{(j + off) % 8}", (lambda x, j=j, off=off: x[j] ^ x[(j + off) % 8])))
    for a, b in itertools.combinations(range(8), 2):
        for nm, f in (("AND", lambda u, v: u & v), ("OR", lambda u, v: u | v), ("XOR", lambda u, v: u ^ v)):
            cs.append((f"in{a} {nm} in{b}", (lambda x, a=a, b=b, f=f: f(x[a], x[b]))))
    return cs


def _is_positional(expr, j):
    """A column function is 'positional' iff it is single-source (inK / NOT inK) or the offset-XOR
    family for THIS column (inj XOR in(j+off)). General 2-input gates are not positional."""
    return bool(re.fullmatch(r"(NOT )?in\d", expr)) or expr.startswith(f"in{j} XOR in")


def _col_derive(j, examples, query, ans_bit, stored_expr):
    """Derive output column j. Returns (chosen_expr, lines). chosen_expr reproduces every example AND
    gives ans_bit on the query (so the composed output is correct). Narration is GENUINE:
      - determined (one candidate fits the examples): show a simpler candidate FAIL a real example, then
        the unique survivor (real elimination);
      - underdetermined (several fit): list them, then pick the positional/minimal one BY CONVENTION
        (the prior the drill installs, here taught in-context);
      - if no enumerated candidate gives the answer bit: fall back to the stored function (honest: it
        fits every example)."""
    exrows = [(_bits(inp), int(out[j])) for inp, out in examples]
    qb = _bits(query)
    cands = _col_candidates(j)
    fits = [(e, f) for e, f in cands if all(f(x) == o for x, o in exrows)]
    chosen = next(((e, f) for e, f in fits if f(qb) == ans_bit), None)
    if chosen is None:                                       # complex (MAJ/CH/...) -> keep stored function
        return stored_expr, [f"  o{j} = {stored_expr} (this fits every example)."]
    ce, cf = chosen
    if len(fits) == 1:
        # genuine elimination: show the NEAREST near-miss (fits the most examples but still fails one),
        # preferring a non-trivial candidate over a bare constant -> a more teachable rejection.
        ranked = sorted(((e, f) for e, f in cands if (e, f) != chosen),
                        key=lambda ef: (sum(ef[1](x) == o for x, o in exrows), ef[0] not in ("0", "1")),
                        reverse=True)
        alt = next((ef for ef in ranked if any(ef[1](x) != o for x, o in exrows)), None)
        if alt:
            ae, af = alt
            bx, bo = next((x, o) for x, o in exrows if af(x) != o)
            ex_s = "".join(map(str, bx))
            return ce, [f"  o{j}: try {ae} -> {af(bx)} on {ex_s}, but the output bit is {bo}. X "
                        f"-> only o{j} = {ce} fits every example."]
        return ce, [f"  o{j} = {ce} (fits every example)."]
    # underdetermined: several fit. Be HONEST about which case this is:
    others = [e for e, f in fits if e != ce][:2]
    if (ce, cf) == fits[0]:
        # chosen IS the simplest fit. Only call it "positional" when it really is (single-source or the
        # inj XOR in(j+off) family for this column); a general 2-input gate is just "simplest".
        kind = "simplest positional" if _is_positional(ce, j) else "simplest"
        return ce, [f"  o{j}: several functions fit the examples ({', '.join([ce] + others)}); they agree "
                    f"on the examples but differ elsewhere, so by convention I take the {kind} function: "
                    f"o{j} = {ce}."]
    # a SIMPLER candidate also fits the examples but the column is not that one -> examples don't decide
    # it; state honestly (no false "simplest" claim). These are the genuinely underdetermined columns.
    return ce, [f"  o{j}: the examples are consistent with several functions ({', '.join(others + [ce])}) "
                f"and do not pin this column down on their own; the rule uses o{j} = {ce}."]


# ── v6: executable / un-imitable CONFIRM (kills rubber-stamp; docs/discuss/v6-bitperbit-fix.md) ──
# V6_BIT_CONFIRM selects how the per-bit verification is RENDERED — the rule/derivation are UNCHANGED
# (rule is honest), only the CONFIRM/ANSWER render differs so the model cannot ECHO the example output:
#   "execA"  -> derive-all, then CONFIRM by substituting the input bits INTO each column and computing;
#               the 'ok' stamp is emitted only AFTER the computed bit (cannot copy the expected output).
#   "atpick" -> no separate CONFIRM block; each column is verified AT THE MOMENT it is picked (propose a
#               function -> test it on every example by substitution -> keep only if all pass).
# CONTAINED to bit_perbit (every other cat copies v5-perfect-bittie verbatim).
V6 = os.environ.get("V6_BIT_CONFIRM")


def _col_eval(expr, term, IN):
    """Value of a column function on input bits IN. beval handles formula / MAJ / CH; complex stored
    terms (3-input NAND etc.) that beval can't parse fall back to the stored-term evaluator."""
    v = B.beval(expr, IN)
    if v is None:
        v = B._bit_term(term, IN)
    return v


def _subst(expr, IN):
    """expr with each inK replaced by its bit value (the executable substitution shown to the model)."""
    return re.sub(r"in(\d)", lambda m: str(IN[int(m.group(1))]), expr)


def _exec_col_line(j, expr, term, IN, expbit):
    """One executable CONFIRM line, e.g. 'o3 = in2 XOR in5 = 1 XOR 0 = 1  | exp 1 ok'. The ok/X stamp
    follows the COMPUTED bit, so it can't be produced by copying the example output. Returns (line, ok, val)."""
    val = _col_eval(expr, term, IN)
    ok = (val == expbit)
    if expr in ("0", "1"):
        body = f"o{j} = {expr}"                              # constant column: literal is its own value
    else:
        sub = _subst(expr, IN)
        body = f"o{j} = {expr} = {val}" if sub == str(val) else f"o{j} = {expr} = {sub} = {val}"
    return f"    {body}  | exp {expbit} {'ok' if ok else 'X'}", ok, val


def _emit_exec_confirm(sk, examples, exprs, terms):
    """Variant A CONFIRM 2: execute every column of every example by substitution (no echo possible)."""
    sk.line("CONFIRM 2 -- execute the rule on every example (substitute the input bits into each column "
            "and compute; 'ok' is written only after the computed bit):")
    for inp, out in examples:
        IN = _bits(inp)
        sk.line(f"  in={inp}:")
        comp = ""
        for j in range(8):
            line, ok, val = _exec_col_line(j, exprs[j], terms[j], IN, int(out[j]))
            sk.line(line)
            comp += str(val)
        sk.line(f"    columns assembled -> {comp}")
        sk._all_ok = sk._all_ok and (comp == out)


def _atpick_seg(n, expr, term, x, o):
    """One inline 'exN <subst>=<val>(exp<o> ok)' token for the verify-at-pick line."""
    v = _col_eval(expr, term, x)
    sub = _subst(expr, x)
    seg = f"{v}" if (expr in ("0", "1") or sub == str(v)) else f"{sub}={v}"
    return f"ex{n} {seg}(exp{o} {'ok' if v == o else 'X'})", (v == o)


def _atpick_col(sk, j, examples, expr, term, search):
    """Variant B: render pick-and-verify for one column. search=True (per-bit derivation) first shows a
    near-miss candidate FAILING, then the chosen function passing every example; search=False (functions
    given by the stored rule) just tests the given function on every example. Verification is fused into
    the pick, so there is no separate 'ok' block to rubber-stamp."""
    exrows = [(_bits(inp), int(out[j])) for inp, out in examples]
    fits = []
    sk.line(f"  o{j}:")
    if search:
        cands = _col_candidates(j)
        fits = [(e, f) for e, f in cands if all(f(x) == o for x, o in exrows)]
        ranked = sorted(((e, f) for e, f in cands if e != expr),
                        key=lambda ef: (sum(ef[1](x) == o for x, o in exrows), ef[0] not in ("0", "1")),
                        reverse=True)
        alt = next((ef for ef in ranked if any(ef[1](x) != o for x, o in exrows)), None)
        if alt:
            ae = alt[0]
            segs = []
            for nn, (x, o) in enumerate(exrows, 1):
                seg, ok = _atpick_seg(nn, ae, ae, x, o)
                segs.append(seg)
                if not ok:
                    break
            sk.line(f"    try {ae}: " + "; ".join(segs) + " -> fails, drop")
    segs = [_atpick_seg(nn, expr, term, x, o)[0] for nn, (x, o) in enumerate(exrows, 1)]
    others = [e for e, f in fits if e != expr][:2]
    tail = ""
    if len(fits) > 1 and others:
        kind = "simplest positional" if _is_positional(expr, j) else "simplest"
        tail = f" [others also fit ({', '.join(others)}); take the {kind}]"
    sk.line(f"    try {expr}: " + "; ".join(segs) + f" -> all examples pass, fix o{j} = {expr}{tail}")


def _emit_answer(sk, query, qout, exprs, terms, executable):
    """ANSWER: apply the rule to the query. When executable, each column SHOWS the substitution
    (o3 = in2 XOR in5 = 1 XOR 0 = 1) rather than just stamping the bit."""
    qIN = _bits(query)
    sk.line(f"ANSWER: query {query} -> in0..7 = {','.join(map(str, qIN))}")
    for j in range(8):
        if executable and exprs[j] not in ("0", "1"):
            sub = _subst(exprs[j], qIN)
            v = _col_eval(exprs[j], terms[j], qIN)
            line = f"  o{j} = {exprs[j]} = {v}" if sub == str(v) else f"  o{j} = {exprs[j]} = {sub} = {v}"
        else:
            line = f"  o{j} = {exprs[j]} = {_bits(qout)[j]}"
        sk.line(line)
    sk.line(f"  -> {qout}")
    sk.line(f"\\boxed{{{qout}}}")
    sk._boxed = qout


# ── helpers shared by the compound-boolean form ─────────────────────────────────
_TUP_OP = {"or": "OR", "and": "AND", "xor": "XOR", "nand": "NAND", "nor": "NOR", "xnor": "XNOR",
           "andn": "ANDN", "orn": "ORN", "nandn": "NANDN", "norn": "NORN"}


def _tuple_to_expr(tup):
    op = tup[0]
    if op == "I":
        return f"in{tup[1]}"
    if op == "not":
        return f"NOT in{tup[1]}"
    if op in ("ch", "maj"):
        return f"{op.upper()}(in{tup[1]},in{tup[2]},in{tup[3]})"
    if op in _TUP_OP and tup[2] is not None:
        return f"in{tup[1]} {_TUP_OP[op]} in{tup[2]}"
    return None


def _extract_bit_terms(rule, cot_c3):
    """{0..7: expr-string} from rule field or cot_c3 (boolean-formula or perbit3-tuple)."""
    for src in (rule or "", cot_c3 or ""):
        if "bit0=" not in src and "bit0 =" not in src:
            continue
        body = src
        m = re.search(r"(?:input bits:|Functions[^:]*:)\s*(.+)", src, re.S)
        if m:
            body = m.group(1)
        terms = {}
        for mm in re.finditer(r"bit(\d)\s*=\s*(.+?)(?=;\s*bit\d|\.\s|\.$|$)", body, re.S):
            i = int(mm.group(1))
            rhs = mm.group(2).strip().rstrip(".").strip()
            if rhs.startswith("("):
                try:
                    e = _tuple_to_expr(ast.literal_eval(rhs))
                    if e:
                        rhs = e
                except Exception:
                    pass
            terms[i] = rhs
        if len(terms) == 8:
            return terms
    return None


# ── sub-renderer: 3-bit neighbor (cellular automaton) ───────────────────────────
def _render_neighbor(rule, prompt, answer):
    m = re.search(r"3-bit neighbor function \((wrapping|no-wrap)\)", rule or "")
    if not m:
        return None
    wrap = m.group(1) == "wrapping"
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None
    table, witness = {}, {}
    for inp, out in examples:
        if len(inp) != len(out):
            return None
        for i in range(len(inp)):
            w = B.neighbor_window(inp, i, wrap)
            o = int(out[i])
            if w in table and table[w] != o:
                return None
            if w not in table:
                witness[w] = (inp, i, out[i])
            table[w] = o

    def apply(s):
        if not all(B.neighbor_window(s, i, wrap) in table for i in range(len(s))):
            return None
        return "".join(str(table[B.neighbor_window(s, i, wrap)]) for i in range(len(s)))

    qout = apply(query)
    if qout is None:
        return None
    edge = ("the string wraps around (bit 0's left neighbour is bit 7, bit 7's right is bit 0)"
            if wrap else "bits off the ends are treated as 0")
    sk = Skeleton()
    sk.analyze("8-bit input -> 8-bit output. Each output bit depends on a 3-bit window of the input "
               f"(left neighbour, itself, right neighbour; {edge}); the SAME rule applies at every "
               "position. I read the 3-bit -> 1-bit table off the examples, then apply it.")
    sk.examples([f"{i} -> {o}" for i, o in examples])
    sk.theory(1, "each output bit just copies the input bit (identity).")
    bad = next(((inp, out, i) for inp, out in examples for i in range(len(inp)) if inp[i] != out[i]), None)
    if bad:
        sk.reject(1, f"position {bad[2]} of {bad[0]}", bad[0][bad[2]], bad[1][bad[2]], "depends on neighbours too")
    sk.line()
    sk.theory(2, "build the window -> output table by reading every position of every example:")
    for w in sorted(table):
        inp, i, ob = witness[w]
        sk.line(f"  ({w[0]},{w[1]},{w[2]}) -> {table[w]}   e.g. {inp} position {i} -> {ob}")
    sk.line("All positions agree across every example (no conflicts).")
    sk.confirm(2, [(inp, apply(inp), out) for inp, out in examples], "CONFIRM 2 (recompute with the table):")
    sk.all_reproduce()
    sk.rule(f"o[i] = table(in[i-1], in[i], in[i+1]) with {edge}; table = "
            + ", ".join(f"({w[0]}{w[1]}{w[2]})->{table[w]}" for w in sorted(table)) + ".")
    sk.answer(f"query {query} -> apply the table at each position -> {qout}.", qout)
    return sk


# ── sub-renderer: compound boolean / cay-bit / perbit3 ──────────────────────────
def _render_compound(terms, prompt, answer):
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None

    def apply(s):
        IN = [int(c) for c in s]
        out = ""
        for i in range(8):
            v = B.beval(terms[i], IN)
            if v is None:
                return None
            out += str(v)
        return out

    for inp, out in examples:
        if apply(inp) != out:
            return None
    qout = apply(query)
    if qout is None:
        return None
    sk = Skeleton()
    verify_txt = ("test each function on every example as I state it" if V6 == "atpick"
                  else "confirm by recomputing every example, then apply")
    sk.analyze("8-bit input -> 8-bit output. No shift or rotation lines up every column, so each output "
               "bit is its own fixed boolean function of the input bits (some combine two inputs with a "
               f"gate). I state each bit's function, {verify_txt}.")
    sk.examples([f"{i} -> {o}" for i, o in examples])
    sk.theory(1, "a single whole-word shift or rotation.")
    name, fn = B.best_simple_transform(examples)
    sk.reject(1, f"{name} on {examples[0][0]}", fn(examples[0][0]), examples[0][1], "bits are recombined")
    sk.line()
    if V6 == "atpick":                                       # functions given by rule -> test each (no search)
        sk.theory(2, "per-bit boolean functions (test each on every example):")
        for i in range(8):
            _atpick_col(sk, i, examples, terms[i], terms[i], search=False)
    else:
        sk.theory(2, "per-bit boolean functions:")
        for i in range(8):
            sk.line(f"  o{i} = {terms[i]}")
        if V6 == "execA":
            _emit_exec_confirm(sk, examples, terms, terms)
        else:
            sk.confirm(2, [(inp, apply(inp), out) for inp, out in examples],
                       "CONFIRM 2 (recompute every example bit-by-bit):")
    sk.all_reproduce()
    sk.rule("; ".join(f"o{i} = {terms[i]}" for i in range(8)) + ".")
    _emit_answer(sk, query, qout, terms, terms, executable=(V6 in ("execA", "atpick")))
    return sk


# ── sub-renderer: standard per-bit from the STORED rule (ground truth) ──────────
# The rule is underdetermined by examples alone (a different boolean can fit the examples but
# disagree on the query), so we present the SOLVED stored rule (validated by exec_rule), using
# example-reads only for the columns that differ from the closest whole-word transform.
def _render_perbit(rule, prompt, answer):
    terms = B.stored_terms(rule)
    if terms is None:
        return None
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None
    for inp, out in examples:                               # stored rule must reproduce every example
        r = B.exec_rule(rule, inp)
        if r is None or r[0] != out:
            return None
    qout = B.exec_rule(rule, query)[0]
    ans = answer if answer == qout else qout                # gold == query output (gate verifies)

    # Nit-3 gate: if a single whole-word transform reproduces EVERY example (and the query), this is not
    # really a per-bit puzzle -> commit the transform cleanly (avoids the "all-ok CONFIRM 1 then 'does not
    # reproduce every column'" self-contradiction). Route to the shift/rotate renderer.
    name0, fn0 = B.best_simple_transform(examples)
    if all(fn0(inp) == out for inp, out in examples) and fn0(query) == qout:
        sk = Skeleton()
        sk.analyze("8-bit input -> 8-bit output. A single whole-word transform lines up every column, so I "
                   "test it directly rather than deriving each bit separately.")
        sk.examples([f"{i} -> {o}" for i, o in examples])
        sk.theory(1, f"the whole word is a single transform: {name0}.")
        sk.confirm(1, [(inp, fn0(inp), out) for inp, out in examples], "CONFIRM 1 (recompute every example):")
        sk.all_reproduce()
        sk.rule(f"output = {name0} of the input.")
        sk.answer(f"query {query} -> {name0} -> {qout}.", qout)
        return sk

    # derive each output column genuinely (enumerate -> eliminate / positional-prior), giving the answer
    chosen, col_lines = {}, {}
    for j in range(8):
        ce, lines = _col_derive(j, examples, query, _bits(qout)[j], B.readable(terms[j]))
        chosen[j], col_lines[j] = ce, lines

    def apply(s):                                           # compose from the CHOSEN per-column functions
        IN = _bits(s)
        out = ""
        for j in range(8):
            v = B.beval(chosen[j], IN)
            if v is None:                                  # stored complex term -> eval via _bit_term
                v = B._bit_term(terms[j], IN)
            out += str(v)
        return out
    for inp, out in examples:                               # chosen functions must reproduce everything
        if apply(inp) != out:
            return None
    if apply(query) != qout:
        return None

    name, fn = B.best_simple_transform(examples)
    # §2 structural backbone: a column is part of the transform ONLY when chosen[j] is LITERALLY the bit
    # the transform puts there (a copy in_k / NOT in_k / constant) — not a complex gate that merely
    # agrees on the examples. _transform_col_expr returns that exact expr (or None). Honest "matches the
    # transform" requires chosen[j] == that expr.
    backbone = [j for j in range(8)
                if matchcol_j(fn, examples, j) and chosen[j] == _transform_col_expr(fn, j)]
    sk = Skeleton()
    # near-fit only when the transform genuinely carries MOST columns (>=4) -> the "keep backbone, fix the
    # rest" framing is truthful; otherwise it is really a per-column puzzle -> plain narration (cfgA).
    if V6 == "atpick":
        # v6 atpick-v2 fix: THEORY-1 (whole-word transform) is tested on the EXAMPLES only. When it
        # reproduces every example it must NOT be dismissed with the false "does not reproduce every
        # column" line (the v6-atpick-v1 dead-branch bug, e.g. fee5976e). Distinguish honestly:
        #   - transform fails >=1 example  -> genuinely not whole-word -> derive all 8 by test-at-pick.
        #   - transform fits every example but FAILS THE QUERY (near-fit, e.g. rotate + one fixed bit) ->
        #     keep the columns it already gets right, derive ONLY the deviating column(s) -> no contradiction
        #     and the structural lesson (rotate + small fix) is taught.
        transform_all_ex = all(fn(inp) == out for inp, out in examples)
        if transform_all_ex and len(backbone) >= 4:           # near-fit: matches every example, fails the query
            deviating = [j for j in range(8) if j not in backbone]
            dcols = ", ".join(f"o{j}" for j in deviating)
            bcols = ", ".join(f"o{j}" for j in backbone)
            sk.analyze("8-bit input -> 8-bit output. A whole-word shift/rotation reproduces every example but "
                       "not the query, so the rule is that transform with a few columns overridden. I keep the "
                       "columns the transform already gets right and derive only the deviating ones by "
                       "PROPOSING a function and TESTING it on every example (substituting the input bits), "
                       "keeping it only if all examples pass -- so each derived column is verified as it is chosen.")
            sk.examples([f"{i} -> {o}" for i, o in examples])
            sk.theory(1, f"the whole word is a single transform: {name}.")
            sk.confirm(1, [(inp, fn(inp), out) for inp, out in examples], "CONFIRM 1 (recompute every example):", gating=False)
            sk.line(f"It reproduces every example, but on the query {query} it gives {fn(query)} while the "
                    f"answer is {qout} -- they differ at {dcols}. So the rule is {name} with {dcols} "
                    f"overridden: I keep the columns the transform already gets right and derive only {dcols}.")
            sk.line()
            sk.theory(2, f"keep {bcols} as the {name}; derive only {dcols} by test-at-pick:")
            for j in backbone:
                sk.line(f"  o{j} = {chosen[j]} (the {name} puts this bit here).")
            for j in deviating:
                _atpick_col(sk, j, examples, chosen[j], terms[j], search=True)
            sk.line()
        else:                                                 # genuine per-bit (transform fails >=1 example)
            sk.analyze("8-bit input -> 8-bit output. I first try several whole-word transforms (shift, rotate, "
                       "XOR, NOT, reverse, etc.); if none fits every column, each output bit must be its own "
                       "function of the input bits. I derive each column by PROPOSING a function and immediately "
                       "TESTING it on every example (substituting the input bits and computing); I keep a "
                       "function only if it matches the output bit on all examples, otherwise I try the next.")
            sk.examples([f"{i} -> {o}" for i, o in examples])
            sk.theory(1, f"the whole word is a single transform: {name}.")
            sk.confirm(1, [(inp, fn(inp), out) for inp, out in examples], "CONFIRM 1 (recompute every example):", gating=False)
            # v7 THEORY 1.5: enumerate more global transforms before giving up → per-bit
            ranked = B.global_transforms_ranked(examples)
            ex0_in, ex0_out = examples[0]
            shown = {name}                                   # already showed the best shift/rotate
            theory15 = []
            for tname, tfn, tscore in ranked:
                if tname in shown:
                    continue
                res = tfn(ex0_in)
                if res == ex0_out:
                    continue                                 # skip transforms that happen to pass ex0 (weak reject)
                theory15.append((tname, res))
                shown.add(tname)
                if len(theory15) >= 3:
                    break
            if theory15:
                sk.line()
                sk.line("THEORY 1.5 -- try more global transforms:")
                for tname, tres in theory15:
                    sk.line(f"  {tname} on {ex0_in}: {tres}, expected {ex0_out}  X")
                sk.line("None of the global transforms reproduce every column.")
            else:
                sk.line("No other global transform matches; each bit must be derived separately.")
            sk.line()
            sk.theory(2, "derive each output column (propose -> test on every example -> keep only if all pass):")
            for j in range(8):
                _atpick_col(sk, j, examples, chosen[j], terms[j], search=True)
            sk.line()
    elif TIEBREAK and len(backbone) >= 4 and len(backbone) < 8:
        deviating = [j for j in range(8) if j not in backbone]
        mlist = ", ".join(f"o{j}" for j in backbone)
        dlist = ", ".join(f"o{j}" for j in deviating)
        sk.analyze("8-bit input -> 8-bit output. A whole-word shift/rotation lines up MOST columns but not "
                   "every one. Rather than re-derive all 8 bits from scratch, I keep the columns the "
                   "transform already gets right and fix ONLY the few that deviate (each by the simplest "
                   "function that fits); then I confirm the whole rule on every example.")
        sk.examples([f"{i} -> {o}" for i, o in examples])
        sk.theory(1, f"the whole word is a single transform: {name}.")
        sk.confirm(1, [(inp, fn(inp), out) for inp, out in examples], "CONFIRM 1 (recompute every example):", gating=False)
        sk.line(f"It already gets columns {mlist} right on every example (those are exactly the {name}); "
                f"only {dlist} deviate. I keep the backbone columns and fix only the deviating ones.")
        sk.line()
        sk.theory(2, f"keep {mlist} as the {name}; fix only {dlist} (simplest function that fits each):")
        for j in backbone:
            sk.line(f"  o{j} = {chosen[j]} (the {name} puts this bit here).")
        for j in deviating:
            for ln in col_lines[j]:
                sk.line(ln)
        sk.line()
    else:
        sk.analyze("8-bit input -> 8-bit output. No single whole-word shift or rotation lines up every column, "
                   "so each output bit is its own function of the input bits. For each column I list the "
                   "functions that fit the examples and, where the examples leave a choice, take the simplest "
                   "positional one; then I confirm the whole rule on every example.")
        sk.examples([f"{i} -> {o}" for i, o in examples])
        sk.theory(1, f"the whole word is a single transform: {name}.")
        sk.confirm(1, [(inp, fn(inp), out) for inp, out in examples], "CONFIRM 1 (recompute every example):", gating=False)
        sk.line("It does not reproduce every column, so I derive the output bits one column at a time.")
        sk.line()
        sk.theory(2, "derive each output column from the examples (simplest positional function that fits):")
        for j in range(8):
            for ln in col_lines[j]:
                sk.line(ln)
        sk.line()
    if V6 == "execA":
        _emit_exec_confirm(sk, examples, chosen, terms)
    elif V6 != "atpick":
        sk.confirm(2, [(inp, apply(inp), out) for inp, out in examples],
                   "CONFIRM 2 (recompute every example with the full rule):")
    sk.all_reproduce()
    sk.rule("; ".join(f"o{j} = {chosen[j]}" for j in range(8)) + ".")
    _emit_answer(sk, query, qout, chosen, terms, executable=(V6 in ("execA", "atpick")))
    return sk


def render(pid, rule, prompt, answer, ctx):
    if rule and rule.startswith("BitProg:"):               # whole-word program (some land under bit_perbit)
        import cat_bit_nonlinear
        return cat_bit_nonlinear.render(pid, rule, prompt, answer, ctx)
    if rule and "neighbor" in rule:
        sk = _render_neighbor(rule, prompt, answer)
    else:
        sk = _render_perbit(rule, prompt, answer)           # from stored rule (ground truth)
        if sk is None:                                      # compound/cay-bit functions in rule/cot_c3
            terms = _extract_bit_terms(rule, ctx.cot_c3(pid))
            sk = _render_compound(terms, prompt, answer) if terms else None
    from common import gate
    if not gate(sk, answer):
        return None
    return sk.build(), dict(form="perbit")

