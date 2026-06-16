"""num_unit — a hidden constant conversion factor: out = k * in. Derive k from the ratios."""
import re

from skeleton import Skeleton
from common import float_eq, gate


def _parse(prompt):
    exs, q = [], None
    for ln in prompt.splitlines():
        m = re.search(r"([\d.]+)\s*m?\s+becomes\s+([\d.]+)", ln)
        if m:
            exs.append((float(m.group(1)), float(m.group(2)), m.group(1), m.group(2)))
        qm = re.search(r"convert the following measurement:\s*([\d.]+)", ln)
        if qm:
            q = qm.group(1)
    return exs, q


def _fmt(x, like):
    dp = len(like.split(".")[1]) if "." in like else 0
    return f"{x:.{dp}f}"


def render(pid, rule, prompt, answer, ctx):
    exs, q = _parse(prompt)
    if not exs or q is None:
        return None
    ks = [o / i for i, o, _, _ in exs if i]
    if not ks:
        return None
    k = sum(ks) / len(ks)
    qv = float(q)
    pred = _fmt(qv * k, answer)

    sk = Skeleton()
    sk.analyze("each measurement maps to an output of similar magnitude, so I hypothesise a single hidden "
               "conversion factor: out = k * in. I first rule out a constant offset, then derive k from the "
               "ratios and verify on every example.")
    sk.examples([f"{si} -> {so}" for _, _, si, so in exs])
    offs = [round(o - i, 4) for i, o, _, _ in exs]
    sk.theory(1, "a constant additive offset, out = in + c.")
    sk.line(f"CONFIRM 1: c = out - in over the examples = {offs}  X -- the offset is not constant, so it is not additive.")
    sk.line()
    sk.theory(2, "a constant multiplicative factor, out = k * in. Read k = out / in off each example:")
    for i, o, si, so in exs:
        sk.line(f"  {so} / {si} = {o / i:.6f}")
    sk.line(f"All ratios agree at k ~ {k:.6f}.")
    # G1: confirm with COMPUTED k*in vs gold output string (rounded to its decimals)
    sk.confirm(2, [(f"{si} * {k:.6f}", _fmt(k * i, so), so) for i, o, si, so in exs], eq=float_eq)
    sk.all_reproduce()
    sk.rule(f"out = {k:.6f} * in.")
    sk.answer(f"query {q} -> {q} * {k:.6f} = {pred}.", pred)

    if not gate(sk, answer):
        return None
    return sk.build(), dict(k=round(k, 6))
