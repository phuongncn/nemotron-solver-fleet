"""bit_nonlinear — a short whole-word bitwise PROGRAM (BitProg: shl/shr/rotl/rotr/xor/and/or/ch/maj).
Reuses the proven interpreter in tools/gen_bit_nonlinear_cot.py (no re-implementation)."""
import os
import sys

from skeleton import Skeleton
import bitlib as B
import bitcegis as CG

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
import gen_bit_nonlinear_cot as BNL


def _render_cegis(pid, rule, prompt, answer, ctx):
    """BitProg-cegis: a sequence of y := <expr> statements (recursive 8-bit program). Confirm on every
    example, apply to the query. Same skeleton as the BitProg path."""
    from common import gate
    prog = rule.split("BitProg-cegis:", 1)[1].strip()
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None
    try:
        for inp, out in examples:
            if format(CG.run(prog, int(inp, 2))[0], "08b") != out:
                return None
        qval, qtrace = CG.run(prog, int(query, 2))
    except Exception:
        return None
    qout = format(qval, "08b")

    sk = Skeleton()
    sk.analyze("8-bit input -> 8-bit output. No single shift, rotation or per-bit boolean lines up every "
               "example, so the rule is a short bitwise PROGRAM over the whole byte: a running value y "
               "(starting as the input x) rewritten by a few word operations. I state the program, confirm "
               "it on every example, then apply it to the query.")
    sk.examples([f"{i} -> {o}" for i, o in examples])
    x0, o0 = int(examples[0][0], 2), examples[0][1]
    sk.theory(1, "a single whole-word rotation or NOT.")
    sk.reject(1, f"rotate-left-1 / NOT on {examples[0][0]}",
              f"{format(((x0 << 1) | (x0 >> 7)) & 0xFF, '08b')} / {format(x0 ^ 0xFF, '08b')}",
              o0, "neither; the bits are recombined by a multi-step program")
    sk.line()
    sk.theory(2, f"the program (x = input byte, y starts as x): {CG.sentence(prog)}. "
                 "Here select(s,a,b) keeps bit a where s=1 else bit b; majority = the majority of three bits.")
    rows = []
    for inp, out in examples:
        _, tr = CG.run(prog, int(inp, 2))
        steps = "; ".join(f"y={v}" for _, v in tr)
        computed = tr[-1][1]
        bits = " ".join(f"b{j}:{computed[j]}={'ok' if computed[j]==out[j] else 'X'}" for j in range(8))
        rows.append((f"{inp}: {steps} [{bits}]", computed, out))
    sk.confirm(2, rows, "CONFIRM 2 (recompute every example step by step, verify each bit):")
    sk.all_reproduce()
    sk.rule(f"{CG.sentence(prog)}.")
    sk.line(f"ANSWER: query {query} ->")
    for rhs, v in qtrace:
        sk.line(f"  y := {CG.readable(rhs)} = {v}")
    sk.line(f"  -> {qout}")
    sk.line(f"\\boxed{{{qout}}}")
    sk._boxed = qout
    if not gate(sk, answer):
        return None
    return sk.build(), dict(out=qout)


def render(pid, rule, prompt, answer, ctx):
    if rule and rule.startswith("BitProg-cegis:"):
        return _render_cegis(pid, rule, prompt, answer, ctx)
    if not rule or not rule.startswith("BitProg:"):
        return None
    prog = rule.split("BitProg:", 1)[1].strip()
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None
    try:
        for inp, out in examples:
            if BNL.b8(BNL.run(prog, int(inp, 2))[0]) != out:
                return None
        qval, qtrace = BNL.run(prog, int(query, 2))
    except Exception:
        return None
    qout = BNL.b8(qval)

    sk = Skeleton()
    sk.analyze("8-bit input -> 8-bit output. No single shift, rotation or per-bit boolean lines up every "
               "example, so the rule is a short bitwise PROGRAM over the whole byte (an intermediate value y "
               "built from input x by a few word operations). I state the program, confirm on every example, apply.")
    sk.examples([f"{i} -> {o}" for i, o in examples])
    x0, o0 = int(examples[0][0], 2), examples[0][1]
    sk.theory(1, "a single whole-word rotation or NOT.")
    sk.reject(1, f"rotate-left-1 / NOT on {examples[0][0]}", f"{BNL.b8(BNL.rotl(x0,1))} / {BNL.b8(x0^BNL.MASK)}",
              o0, "neither; bits are recombined by a multi-step program")
    sk.line()
    sk.theory(2, f"the program (x = input byte): {BNL.rule_sentence(prog)}. "
                 "Here select(s,a,b) takes bit a where s=1 else bit b; maj = majority of the three.")
    rows = []
    for inp, out in examples:
        res, tr = BNL.run(prog, int(inp, 2))
        steps = "; ".join(f"{d}={v}" for d, v in tr)
        computed = BNL.b8(res)
        bits = " ".join(f"b{j}:{computed[j]}={'ok' if computed[j]==out[j] else 'X'}" for j in range(8))
        rows.append((f"{inp}: {steps} [{bits}]", computed, out))
    sk.confirm(2, rows, "CONFIRM 2 (recompute every example step by step, verify each bit):")
    sk.all_reproduce()
    sk.rule(f"{BNL.rule_sentence(prog)}.")
    sk.line(f"ANSWER: query {query} ->")
    for d, v in qtrace:
        sk.line(f"  {d} = {v}")
    sk.line(f"  -> {qout}")
    sk.line(f"\\boxed{{{qout}}}")
    sk._boxed = qout

    from common import gate
    if not gate(sk, answer):
        return None
    return sk.build(), dict(out=qout)
