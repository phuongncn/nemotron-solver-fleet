"""bit_rotate — the output is a cyclic rotation of the input by a fixed amount/direction."""
import re

from skeleton import Skeleton
import bitlib as B


def render(pid, rule, prompt, answer, ctx):
    m = re.match(r"\s*Rotate\s+(left|right)\s+(\d)", rule or "")
    if not m:
        return None
    direction, k = m.group(1), int(m.group(2))
    examples, query = B.parse_examples(prompt)
    if not examples or query is None:
        return None

    def rot(s):
        kk = k % 8
        if kk == 0:
            return s
        return (s[kk:] + s[:kk]) if direction == "left" else (s[-kk:] + s[:-kk])

    for inp, out in examples:
        if rot(inp) != out:
            return None

    sk = Skeleton()
    sk.analyze("8-bit input -> 8-bit output. The output is a cyclic rearrangement of the input (same bits, "
               "slid around the ring), so I hypothesise a rotation, read its direction and amount off one "
               "example, then verify on every example.")
    sk.examples([f"{i} -> {o}" for i, o in examples])
    inp0, out0 = examples[0]
    sk.theory(1, f"align {inp0} with {out0}. Each output bit is the input bit {k} place(s) to the "
                 f"{direction} on the ring -> rotate {direction} by {k}.")
    sk.confirm(1, [(inp, rot(inp), out) for inp, out in examples])
    sk.all_reproduce()
    sign = "+" if direction == "left" else "-"
    sk.rule(f"rotate {direction} by {k}  (o[i] = in[(i{sign}{k}) mod 8]).")
    qout = rot(query)
    sk.answer(f"query {query} -> rotate {direction} by {k} -> {qout}.", qout)

    from common import gate
    if not gate(sk, answer):
        return None
    return sk.build(), dict(direction=direction, k=k)
