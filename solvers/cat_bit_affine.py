"""bit_affine — read the 8-bit string as an integer; output = (a*x + b) mod 256.
Derive a,b genuinely from two examples (mod-256 inverse), then verify on every example."""
import re

from skeleton import Skeleton
import bitlib as B


def _inv_mod256(a):
    a %= 256
    if a % 2 == 0:
        return None
    for x in range(1, 256, 2):
        if (a * x) % 256 == 1:
            return x
    return None


def render(pid, rule, prompt, answer, ctx):
    m = re.search(r"\(\s*(\d+)\s*\*\s*x\s*\+\s*(\d+)\s*\)\s*mod\s*256", rule or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None
    f = lambda x: (a * x + b) % 256
    for inp, out in examples:
        if format(f(int(inp, 2)), "08b") != out:
            return None
    rows = [(int(inp, 2), int(out, 2), inp, out) for inp, out in examples]

    # derive a,b from two examples whose x-difference is invertible (odd) mod 256
    pair = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if ((rows[i][0] - rows[j][0]) % 256) % 2 == 1:
                pair = (rows[i], rows[j])
                break
        if pair:
            break
    if pair:
        (x1, y1, _, _), (x2, y2, _, _) = pair
        dx = (x1 - x2) % 256
        dy = (y1 - y2) % 256
        inv = _inv_mod256(dx)
        a_d = (dy * inv) % 256
        b_d = (y1 - a_d * x1) % 256
        derive = (f"from x={x1}->y={y1} and x={x2}->y={y2}: a = (y1-y2)*(x1-x2)^-1 = {dy}*{inv} mod 256 "
                  f"= {a_d}; b = y1 - a*x1 = {y1} - {a_d}*{x1} mod 256 = {b_d}.")
    else:
        a_d, b_d = a, b
        derive = f"out = ({a}*x + {b}) mod 256."

    sk = Skeleton()
    sk.analyze("8-bit input -> 8-bit output. I first try a simple bitwise transform (shift, rotate, NOT); "
               "if that fails, I read each 8-bit string as an integer 0-255 and try an affine integer rule "
               "out = (a*x + b) mod 256, solving a,b from two examples and verifying on all.")
    sk.examples([f"{inp} -> {out}   (x={x} -> y={y})" for x, y, inp, out in rows],
                header="Examples (x -> y as integers):")
    name, fn = B.best_simple_transform(examples)
    sk.theory(1, f"a simple bitwise transform: {name}.")
    sk.reject(1, f"{name} on {examples[0][0]}", fn(examples[0][0]), examples[0][1],
              "bits are recombined by an integer arithmetic rule, not a bitwise transform")
    sk.line()
    sk.theory(2, f"affine integer: out = (a*x + b) mod 256. {derive}")
    sk.confirm(2, [(f"x={x}: ({a_d}*{x}+{b_d}) mod 256", (a_d * x + b_d) % 256, y) for x, y, _, _ in rows])
    sk.all_reproduce()
    sk.rule(f"out = ({a_d}*x + {b_d}) mod 256.")
    qx = int(query, 2)
    qy = (a_d * qx + b_d) % 256
    qout = format(qy, "08b")
    sk.answer(f"query {query} -> x={qx}. ({a_d}*{qx}+{b_d}) mod 256 = {qy} = {qout}.", qout)

    from common import gate
    if not gate(sk, answer):
        return None
    return sk.build(), dict(a=a_d, b=b_d)
