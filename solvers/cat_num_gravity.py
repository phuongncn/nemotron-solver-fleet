"""num_gravity — d = 0.5 * g * t^2 with a secret g. Derive g = 2d / t^2 from the examples."""
import re

from skeleton import Skeleton
from common import float_eq, gate


def _parse(prompt):
    exs, q = [], None
    for ln in prompt.splitlines():
        m = re.search(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)", ln)
        if m:
            exs.append((float(m.group(1)), float(m.group(2)), m.group(1), m.group(2)))
        qm = re.search(r"distance for\s*t\s*=\s*([\d.]+)\s*s", ln)
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
    gs = [2 * d / (t * t) for t, d, _, _ in exs if t]
    if not gs:
        return None
    g = sum(gs) / len(gs)
    qt = float(q)
    pred = _fmt(0.5 * g * qt * qt, answer)

    sk = Skeleton()
    sk.analyze("the prompt gives the law d = 0.5 * g * t^2, so the only unknown is the gravity g. I first "
               "rule out Earth gravity, then solve g = 2d / t^2 from each example and verify it is constant.")
    sk.examples([f"t = {st} s -> d = {sd} m" for t, d, st, sd in exs], header="Examples (t -> d):")
    t0, d0, st0, sd0 = exs[0]
    sk.theory(1, "standard Earth gravity g = 9.8.")
    sk.reject(1, f"0.5 * 9.8 * {st0}^2", _fmt(0.5 * 9.8 * t0 * t0, sd0), sd0, "gravity is not 9.8 here")
    sk.line()
    sk.theory(2, "a secret constant g. Solve g = 2d / t^2 from each example:")
    for t, d, st, sd in exs:
        sk.line(f"  2 * {sd} / {st}^2 = {2 * d / (t * t):.4f}")
    sk.line(f"All give g ~ {g:.4f}.")
    sk.confirm(2, [(f"0.5 * {g:.4f} * {st}^2", _fmt(0.5 * g * t * t, sd), sd) for t, d, st, sd in exs], eq=float_eq)
    sk.all_reproduce()
    sk.rule(f"d = 0.5 * {g:.4f} * t^2.")
    sk.answer(f"query t = {q} -> 0.5 * {g:.4f} * {q}^2 = {pred}.", pred)

    if not gate(sk, answer):
        return None
    return sk.build(), dict(g=round(g, 4))
