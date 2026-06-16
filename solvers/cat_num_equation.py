"""num_equation — each operator glyph hides an operation (endian/op/offset/reverse), decoded per-glyph.
Two forms: (a) query glyph present in examples -> decode its rows (G4: stored spec, re-brute with the
ANSWER constraint when the stored spec drops the result format); (b) query glyph ABSENT -> missing-op
family (query op = the one of {add,sub,mul} the examples leave out)."""
import ast
import os
import re
import sys

from skeleton import Skeleton
from common import gate, verify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
import gen_numeric_eq_cot as EQ


def _parse(prompt):
    exs, q = [], None
    for ln in prompt.splitlines():
        m = re.match(r"\s*(\d+)\s*(\D)\s*(\d+)\s*=\s*(\S+)\s*$", ln)
        if m:
            exs.append((m.group(1), m.group(2), m.group(3), m.group(4)))
        qm = re.search(r"result for:\s*(\d+)\s*(\D)\s*(\d+)", ln)
        if qm:
            q = (qm.group(1), qm.group(2), qm.group(3))
    return exs, q


def _brute_glyph(rows, g):
    def fits(sp):
        return all((c := EQ.compute(a, b, g, sp)) and c[0] == r for a, b, r in rows)
    for mode in EQ._BRUTE_CONCAT:
        if fits({"concat": mode}):
            return {"concat": mode}
    for rev_ops in (False, True):
        for rev_res in (False, True):
            for op in EQ._BRUTE_OPS:
                for off in EQ._BRUTE_OFFS:
                    for fmt in EQ._BRUTE_FMTS:
                        sp = {"op": op, "rev_ops": rev_ops, "rev_res": rev_res, "off": off, "fmt": fmt}
                        if fits(sp):
                            return sp
    return None


def _brute_glyph_exact(rows, g, qa, qb, gold):
    """Like _brute_glyph but also requires the QUERY (qa g qb) to reproduce the gold STRING exactly.
    Picks a spec that both fits every example row AND gives the exact official answer for the query
    (so leading zeros / sign formatting are preserved). Returns spec or None."""
    def fits_rows(sp):
        return all((c := EQ.compute(a, b, g, sp)) and c[0] == r for a, b, r in rows)

    def hits_query(sp):
        c = EQ.compute(qa, qb, g, sp)
        return c and str(c[0]) == gold
    for mode in EQ._BRUTE_CONCAT:
        sp = {"concat": mode}
        if fits_rows(sp) and hits_query(sp):
            return sp
    for rev_ops in (False, True):
        for rev_res in (False, True):
            for op in EQ._BRUTE_OPS:
                for off in EQ._BRUTE_OFFS:
                    for fmt in EQ._BRUTE_FMTS:
                        sp = {"op": op, "rev_ops": rev_ops, "rev_res": rev_res, "off": off, "fmt": fmt}
                        if fits_rows(sp) and hits_query(sp):
                            return sp
    return None


# the standard arithmetic meaning a bare operator glyph suggests -- the FIRST guess a reader makes, and
# (v5fix2 Việc 2, v7: removed anti-concat bias) absent-op: take the simplest operation that reproduces the query.
_STD_GLYPH = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "X": "mul",
              "×": "mul", "·": "mul", "/": "mod", "%": "mod"}


# arithmetic ops in Occam (simplest-first) order; concat is ranked AFTER all arithmetic
_SIMPLE_ARITH = ["add", "sub", "sub_rev", "mul", "abs_diff", "mod", "mod_rev", "maxmod", "maxdiv", "gcd"]
_OFF_GROUPS = [[0], [1, -1], [2, -2], [3, -3]]


def _simplest_glyph(rows, g):
    """Pick the SIMPLEST spec that fits EVERY row of glyph g (Occam): minimal |offset|, then basic op,
    then fewest reversals, then plainest format. Prevents an under-determined glyph (often a single
    non-query example) from being taught an exotic over-fit -- e.g. 'subtract, add 2, pad to 2 digits'
    when 'absolute difference, pad' already fits 77(78=01 with no fabricated offset. Returns spec|None."""
    def fits(sp):
        return all((c := EQ.compute(a, b, g, sp)) and c[0] == r for a, b, r in rows)
    FMTS = [None, "lz2", "lz4", "suffix_neg", "prefix_neg", "prefix_always",
            "suffix_always", "lz2_suffix_neg", "lz2_prefix_neg"]
    REVS = [(False, False), (True, True), (True, False), (False, True)]
    best = None
    for mode in EQ._BRUTE_CONCAT:                            # concat ranked after every arithmetic op
        if fits({"concat": mode}):
            best = ((0, len(_SIMPLE_ARITH), 0, 0), {"concat": mode})
            break
    for grp in _OFF_GROUPS:                                  # smallest |offset| group with any fit wins
        for off in grp:
            for oi, op in enumerate(_SIMPLE_ARITH):
                for ri, (ro, rr) in enumerate(REVS):
                    for fi, fmt in enumerate(FMTS):
                        sp = {"op": op, "rev_ops": ro, "rev_res": rr, "off": off, "fmt": fmt}
                        if fits(sp):
                            key = (abs(off), oi, ri, fi)
                            if best is None or key < best[0]:
                                best = (key, sp)
        if best is not None:
            break
    # fall back to the full brute (wider op/format catalog) so coverage never drops vs _brute_glyph
    return best[1] if best else _brute_glyph(rows, g)


def _pass1_then_fail(g, rows, sp):
    """v5fix3 BUG 3: find a plausible WRONG candidate op that reproduces the FIRST example row but FAILS a
    LATER row -> teaches 'one matching row isn't proof, verify EVERY row before committing' (the real
    backtrack, not the always-fail-row-1 ritual). Returns (cand_spec, fail_row_index) or None."""
    if len(rows) < 2:
        return None
    pool, seen = [], set()
    std = _STD_GLYPH.get(g)
    for op in ([std] if std else []) + ["add", "sub", "sub_rev", "mul", "abs_diff"] + (
            [sp["op"]] if "op" in sp else []):
        if op and op in EQ.BASE_OPS:
            pool.append({"op": op, "rev_ops": False, "rev_res": False, "off": 0, "fmt": None})
    for cand in pool:
        key = tuple(sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        c0 = EQ.compute(rows[0][0], rows[0][1], g, cand)
        if not c0 or c0[0] != rows[0][2]:                    # must reproduce the FIRST row
            continue
        for i in range(1, len(rows)):
            ci = EQ.compute(rows[i][0], rows[i][1], g, cand)
            got = ci[0] if ci else "undefined"
            if got != rows[i][2]:                            # ...but fail a later one
                return cand, i
    return None


def _glyph_derive_lines(g, rows, sp, pinned=True):
    """Genuine fail->backtrack derivation for one glyph (v5-perfect-fix2 Việc 1: teach a CONFIRM with
    TEETH -- one that actually FAILS a recompute before it passes, so the model learns verification is
    real, not a rubber stamp). The first candidate is the glyph's STANDARD arithmetic meaning (what a
    reader assumes `+`/`-`/`*` mean); if that differs from the true op, compute it on the rows -- it
    really mismatches, SHOW the X, then refine. If the standard guess equals the true base op but the
    spec adds a reverse/offset/format, fall back to plain-op -> refine. Every failure shown is a real
    computed mismatch (got != expected), never a scripted decoy.

    DETERMINACY GATE (pinned=False): a glyph with <2 examples is NOT pinned by them -- asserting an exotic
    rule off one row is the round-1 fabricate-from-1-example anti-pattern. For such a glyph we state
    honestly that one example can't fix the rule and give the SIMPLEST consistent operation, no fake
    fail->refine confidence."""
    if not pinned:
        L = [f"  glyph `{g}` appears in only one example, so it is not fully pinned by the examples; the "
             f"simplest operation consistent with it is {EQ.describe(sp)}:"]
        for a, b, r in rows:
            L.append(f"    {a}{g}{b}: " + "; ".join(EQ.compute(a, b, g, sp)[1]) + f"  -> {r} ok")
        return L
    # BUG 3 preferred backtrack: a wrong op that PASSES row 1 then FAILS a later row (teaches verify-all)
    p1f = _pass1_then_fail(g, rows, sp)
    if p1f is not None:
        cand, fi = p1f
        a0, b0, r0 = rows[0]
        ai, bi, ri = rows[fi]
        gc0, gci = EQ.compute(a0, b0, g, cand), EQ.compute(ai, bi, g, cand)
        L = [f"  glyph `{g}`: try {EQ.describe(cand)} -- {a0}{g}{b0}: " + "; ".join(gc0[1]) + f" -> {r0} ok, "
             f"but {ai}{g}{bi}: " + "; ".join(gci[1]) + f" -> {gci[0]}, expected {ri}. X -- one matching row "
             f"is not enough; it must hold for EVERY row. Refine to {EQ.describe(sp)}:"]
        for a, b, r in rows:
            L.append(f"    {a}{g}{b}: " + "; ".join(EQ.compute(a, b, g, sp)[1]) + f"  -> {r} ok")
        return L
    L = []
    true_op = sp.get("op")
    guess = None
    std_op = _STD_GLYPH.get(g)
    if std_op and true_op and std_op != true_op and std_op in EQ.BASE_OPS:
        guess = {"op": std_op, "rev_ops": False, "rev_res": False, "off": 0, "fmt": None}
    elif "op" in sp and (sp.get("rev_ops") or sp.get("rev_res") or sp.get("off", 0) or sp.get("fmt")):
        guess = {"op": true_op, "rev_ops": False, "rev_res": False, "off": 0, "fmt": None}
    fail = None
    if guess is not None:
        for a, b, r in rows:
            c = EQ.compute(a, b, g, guess)
            got = c[0] if c else "undefined"
            if got != r:
                fail = (a, b, r, got)
                break
        if fail is None:
            guess = None                                     # the guess fits every row -> no fake fail
    if fail is not None:
        a, b, r, got = fail
        gc = EQ.compute(a, b, g, guess)
        steps = "; ".join(gc[1]) if gc else "undefined"
        L.append(f"  glyph `{g}`: first guess {EQ.describe(guess)} -- {a}{g}{b}: {steps} -> {got}, "
                 f"but the output is {r}. X -> {EQ.describe(guess)} is wrong; refine to {EQ.describe(sp)}:")
    else:
        L.append(f"  glyph `{g}` = {EQ.describe(sp)}:")
    for a, b, r in rows:
        L.append(f"    {a}{g}{b}: " + "; ".join(EQ.compute(a, b, g, sp)[1]) + f"  -> {r} ok")
    return L


def _opname3(bop, off):
    s = {"add": "add", "sub": "subtract", "mul": "multiply"}.get(bop, bop)
    if off > 0:
        s += f" then +{off}"
    elif off < 0:
        s += f" then {off}"
    return s


# ── form (b): query op ABSENT from examples -> "the remaining transformation" ────
# Classic case: shown ops are 2 of {add,sub,mul}, query = the 3rd. DC's missing-op-recover extends
# this to a wider catalog (concat/abs_diff/maxmod/...). The rule names the query op but its (and the
# shown ops') exact offset is lossy, so we BRUTE each shown glyph from its rows (genuine derive) and
# build the query spec from the rule's stated base/off/ro/rr/fmt, then verify it reproduces the ANSWER.
def _qspec(qbase, qoff, ro, rr, fmt):
    sp = EQ.spec_from_opname(qbase, fmt)
    if sp is None:
        return None
    if "concat" in sp:
        return sp                                            # concat ignores rev/off
    sp = dict(sp)
    sp["rev_ops"], sp["rev_res"], sp["off"], sp["fmt"] = ro, rr, qoff, fmt
    return sp


def _parse_missing_rule(rule):
    """Two stored formats for the query op of an Eq-missing-op rule -> (qbase, qoff, ro, rr, fmt) or None.
      A) query X=base(offN) ro=B rr=B fmt=F
      B) query X=base | T(rev_ops,rev_res,off,fmt)=(B, B, N, F)"""
    rule = rule or ""
    m = re.search(r"query\s*\S=(\w+)\(off([+-]?\d+)\)\s*ro=(\w+)\s*rr=(\w+)\s*fmt=(\w+)", rule)
    if m:
        return m.group(1), int(m.group(2)), m.group(3) == "True", m.group(4) == "True", \
            (None if m.group(5) == "None" else m.group(5))
    m = re.search(r"query\s*\S=(\w+)\s*\|\s*T(?:\([^)]*\))?\s*=\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*([+-]?\d+)\s*,\s*('?\w+'?)\s*\)",
                  rule)
    if m:
        f = m.group(5).strip("'")
        return m.group(1), int(m.group(4)), m.group(2) == "True", m.group(3) == "True", (None if f == "None" else f)
    return None


# missing-op: the query glyph is absent from the examples. We ALWAYS decode every SHOWN operator
# genuinely from its own rows (those CONFIRMs can fail), then settle the query operator HONESTLY:
#   - FORCED: the family is exactly {add,sub,mul}, the examples show the other two (>=2 rows each,
#     one shared endian, no fabricated offset) -> the absent op is the remaining one (genuine forcing).
#   - otherwise the absent op is NOT determined by the examples -> we say so plainly and state the
#     operation that reproduces the query (no fake "the examples force it" claim, no stamped CONFIRM
#     on a fabrication). Either way boxed==answer; the difference is honest wording, not coverage.
def _arith_class(op):
    if op in ("sub", "sub_rev"):
        return "sub"
    if op in ("add", "mul"):
        return op
    return None                                              # outside the closed 3-op family


# complexity-ordered op list for the simplest-rule search: basic arithmetic first, exotic last
_SIMPLE_OPS = ["add", "sub", "sub_rev", "mul", "abs_diff", "concat", "concat_rev", "mod", "mod_rev",
               "maxmod", "maxdiv", "gcd"]


def _simplest_qspec(qa, qb, qg, answer, gspec):
    """Among all specs that reproduce the answer for the ABSENT query op, return the simplest by
    (|offset|, op-rank, fmt-rank). The reverse-operand/reverse-result settings are inherited from the
    shown ops when they agree (the puzzle's convention); offsets are tried 0 first. Returns a spec or
    None. This makes the honest CoT state the minimal answer-consistent rule, not a back-fit construction.
    EXACT match required (not Kaggle ±1%): the simpler rule must reproduce the gold STRING exactly, else
    it would print a boxed value differing from the official answer (e.g. add->109 vs gold 108)."""
    gold = str(answer).strip()
    def _hit(c):
        return c and str(c[0]) == gold
    ros = {sp.get("rev_ops", False) for sp in gspec.values() if "op" in sp}
    rrs = {sp.get("rev_res", False) for sp in gspec.values() if "op" in sp}
    ro_opts = [ros.pop()] if len(ros) == 1 else [False, True]
    rr_opts = [rrs.pop()] if len(rrs) == 1 else [False, True]
    best = None
    for off in [0, 1, -1, 2, -2]:
        for oi, op in enumerate(_SIMPLE_OPS):
            for fi, fmt in enumerate([None, "suffix_neg", "prefix_neg", "lz2", "lz2_suffix_neg"]):
                if op in EQ._BRUTE_CONCAT:
                    sp = {"concat": op}
                    if off != 0 or fi:           # concat ignores offset/fmt -> only consider once
                        continue
                    if _hit(EQ.compute(qa, qb, qg, sp)):
                        key = (abs(off), oi, fi)
                        if best is None or key < best[0]:
                            best = (key, sp)
                    continue
                for ro_ in ro_opts:
                    for rr_ in rr_opts:
                        cand = {"op": op, "rev_ops": ro_, "rev_res": rr_, "off": off, "fmt": fmt}
                        if _hit(EQ.compute(qa, qb, qg, cand)):
                            key = (abs(off), oi, fi)
                            if best is None or key < best[0]:
                                best = (key, cand)
        if best is not None and best[0][0] == 0:
            break                                            # found an offset-0 (simplest) rule; stop
    return best[1] if best else None


def _missing_op(qa, qg, qb, exs, rule, answer):
    parsed = _parse_missing_rule(rule)
    if parsed is None:
        return None
    qbase, qoff, ro, rr, fmt = parsed
    by_g = {g: [(a, b, r) for a, gg, b, r in exs if gg == g] for g in {g for _, g, _, _ in exs}}
    if qg in by_g:                                            # query op must genuinely be absent
        return None
    # decode every SHOWN glyph from its own rows (genuine; a wrong shown op would fail its CONFIRM)
    gspec = {}
    for g, rows in by_g.items():
        sp = _simplest_glyph(rows, g)                        # Occam: sparse shown glyphs get the simplest fit
        if sp is None:
            return None
        gspec[g] = sp

    # is the query op genuinely FORCED? closed-3 family, the other two shown (>=2 rows each), one
    # shared endian, no fabricated offset, and the forced spec reproduces the answer.
    qclass = _arith_class(qbase)
    qsp = None
    forced = False
    if qclass is not None and qoff == 0 and all(len(r) >= 2 for r in by_g.values()):
        classes, ok = set(), True
        for sp in gspec.values():
            c = _arith_class(sp.get("op")) if "op" in sp else None
            if c is None:
                ok = False
                break
            classes.add(c)
        ros = {sp.get("rev_ops", False) for sp in gspec.values()}
        rrs = {sp.get("rev_res", False) for sp in gspec.values()}
        if ok and classes == ({"add", "sub", "mul"} - {qclass}) and len(ros) == 1 and len(rrs) == 1:
            fmts = {sp.get("fmt") for sp in gspec.values()}
            qfmt = fmts.pop() if len(fmts) == 1 else fmt
            cand = {"op": qbase, "rev_ops": ros.pop(), "rev_res": rrs.pop(), "off": 0, "fmt": qfmt}
            cc = EQ.compute(qa, qb, qg, cand)
            if cc and verify(answer, cc[0]):
                qsp, forced = cand, True
    if qsp is None:                                          # honest path: state the query rule
        # GUARD 1: the absent query op is answer-fit either way, so among the rules that reproduce the
        # answer pick the SIMPLEST one (Occam) -- minimal |offset|, basic op, reverse-settings matching
        # the shown ops' convention. Avoids over-engineered rules like "subtract 2 + suffix" (899e7ce8)
        # when "subtract (b-a)" already reproduces gold. Falls back to the stored spec if nothing simpler.
        qsp = _simplest_qspec(qa, qb, qg, answer, gspec) or _qspec(qbase, qoff, ro, rr, fmt)
        if qsp is None:
            return None
    qc = EQ.compute(qa, qb, qg, qsp)
    if not qc or not verify(answer, qc[0]):
        return None

    sk = Skeleton()
    if forced:
        sk.analyze("each operator symbol hides one arithmetic operation, all read at the same digit-endian. "
                   f"The query operator `{qg}` never appears in the examples, so I decode every SHOWN operator "
                   "from its rows; the query operator is then forced to be the one remaining arithmetic op.")
    else:
        sk.analyze("each operator symbol hides its own operation. I decode and verify every SHOWN operator "
                   f"from its rows. The query operator `{qg}` never appears in the examples, so it is not "
                   "pinned down by them alone. I take the simplest operation that reproduces the query "
                   "(arithmetic or concatenation) and check it.")
    sk.examples([f"{a}{g}{b} = {r}" for a, g, b, r in exs], header="Examples (equation = output):")
    a0, g0, b0, r0 = exs[0]
    sk.theory(1, "each operator keeps its standard meaning.")
    sk.reject(1, f"{a0}{g0}{b0} standard", f"{a0}{g0}{b0}", r0, "decode each operator from its rows")
    sk.line()
    sk.theory(2, "identify each SHOWN operator from its rows (try the plain op first, fix where it fails), "
                 "then settle the absent query operator.")
    for gchar in sorted(gspec, key=lambda c: [a for a, g, b, r in exs].count(c)):
        pinned = len(by_g[gchar]) >= 2                       # shown glyph with <2 examples = not pinned
        for ln in _glyph_derive_lines(gchar, by_g[gchar], gspec[gchar], pinned=pinned):
            sk.line(ln)
    if forced:
        shown_names = " and ".join(sorted({"add": "add", "sub": "subtract", "mul": "multiply"}[
            _arith_class(gspec[g].get("op"))] for g in gspec))
        sk.line(f"  The operators shown are {shown_names}; the family is add/subtract/multiply, so the absent "
                f"operator `{qg}` is forced to be the remaining one: {EQ.describe(qsp)}.")
    elif "concat" in qsp:
        sk.line(f"  The query operator `{qg}` is absent from the examples, so it is not determined by them; "
                f"no standard arithmetic operation reproduces the query, so the rule is to {EQ.describe(qsp)}.")
    else:
        sk.line(f"  The query operator `{qg}` is absent from the examples, so it is not determined by them. "
                f"The simplest operation that reproduces the query is {EQ.describe(qsp)}.")
    qtail = (" (the remaining " + {"add": "add", "sub": "subtract", "mul": "multiply"}[qclass] + ")") if forced else ""
    sk.line()
    sk.confirm(2, [(f"{a}{g}{b}: computed={EQ.compute(a, b, g, gspec[g])[0]}",
                     EQ.compute(a, b, g, gspec[g])[0], r) for a, g, b, r in exs])
    sk.line("Every shown example reproduces exactly.")
    sk.rule("; ".join(f"`{g}` = {EQ.describe(gspec[g])}" for g in gspec)
            + f"; `{qg}` = {EQ.describe(qsp)}{qtail}.")
    sk.answer(f"{qa}{qg}{qb} -> " + "; ".join(qc[1]) + f" = {qc[0]}.", qc[0])
    return sk


def render(pid, rule, prompt, answer, ctx):
    exs, q = _parse(prompt)
    if not exs or q is None:
        return None
    qa, qg, qb = q
    if not any(g == qg for _, g, _, _ in exs):
        sk = _missing_op(qa, qg, qb, exs, rule, answer)
        if not gate(sk, answer):
            return None
        return sk.build(), dict(form="missing-op")

    glyphs = sorted({g for _, g, _, _ in exs})
    by_g = {g: [(a, b, r) for a, gg, b, r in exs if gg == g] for g in glyphs}
    stored = EQ.parse_rule(rule or "")
    gspec = {}
    for g in glyphs:
        sp = stored.get(g) or stored.get(None)
        if not (sp and all((c := EQ.compute(a, b, g, sp)) and c[0] == r for a, b, r in by_g[g])):
            sp = _simplest_glyph(by_g[g], g)                 # Occam: avoid exotic over-fit on a sparse glyph
        if sp is None:
            return None
        gspec[g] = sp
    # G4: query glyph must reproduce the gold EXACTLY (not just Kaggle ±1%), so the boxed value matches the
    # official answer string -- e.g. concat of "08" must keep the leading zero ("0861", not "861").
    gold = str(answer).strip()
    qc = EQ.compute(qa, qb, qg, gspec[qg]) if qg in gspec else None
    if qc is None or str(qc[0]) != gold:
        sp = _brute_glyph_exact(by_g[qg], qg, qa, qb, gold)  # row-consistent AND exact gold (keeps format)
        if sp is None:
            sp = EQ.brute_spec(by_g[qg], qa, qb, qg, answer)  # fall back to answer-constrained (±1%)
        if sp is None:
            return None
        gspec[qg] = sp
        qc = EQ.compute(qa, qb, qg, sp)
        if qc is None or not verify(answer, qc[0]):
            return None

    sk = Skeleton()
    sk.analyze(f"the operator glyph(s) {' '.join(glyphs)} each carry a hidden operation. I decode each from "
               f"its own rows, then apply the rule for `{qg}` to the query {qa}{qg}{qb}.")
    sk.examples([f"{a}{g}{b} = {r}" for a, g, b, r in exs], header="Examples (equation = output):")
    STD = {"*": "mul", "+": "add", "-": "sub", "/": "mod", "%": "mod"}

    def std_of(a, b, g):
        c = EQ.compute(a, b, g, {"op": STD.get(g, "mul"), "rev_ops": False, "rev_res": False})
        return c[0] if c else "undefined"
    disproof = next(((a, g, b, r) for a, g, b, r in exs + [(qa, qg, qb, answer)] if std_of(a, b, g) != r), None)
    sk.theory(1, "each glyph keeps its standard meaning.")
    if disproof:
        a, g, b, r = disproof
        sk.reject(1, f"{a}{g}{b} standard", std_of(a, b, g), r, "standard meanings fail; identify the remapped operation per glyph")
    else:
        sk.line("CONFIRM 1: standard ops match the example rows only by coincidence but break on the query; "
                "I pin each glyph's true operation from its rows.")
    sk.line()
    sk.theory(2, "each glyph maps to a specific operation. Pin it from its rows (try the plain op first, "
                 "fix it where it fails):")
    for g in glyphs:
        pinned = len(by_g[g]) >= 2 or g == qg                # <2 examples (and not the answer glyph) = not pinned
        for ln in _glyph_derive_lines(g, by_g[g], gspec[g], pinned=pinned):
            sk.line(ln)
    sk.line()
    sk.confirm(2, [(f"{a}{g}{b}", EQ.compute(a, b, g, gspec[g])[0], r) for a, g, b, r in exs],
               "CONFIRM 2 (recompute every example end-to-end):")
    sk.line("Every example reproduces exactly.")
    sk.rule("; ".join(f"`{g}` = {EQ.describe(gspec[g])}" for g in glyphs) + ".")
    sk.answer(f"{qa}{qg}{qb} -> " + "; ".join(qc[1]) + f" = {qc[0]}.", qc[0])

    if not gate(sk, answer):
        return None
    return sk.build(), dict(glyphs=len(glyphs))
