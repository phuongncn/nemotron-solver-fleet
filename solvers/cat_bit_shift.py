"""bit_shift — logical shift of the whole bit-string by a fixed distance, zeros filling the vacated
end. The shift (direction + amount k) is OVER-DETERMINED by the examples (one global k must map every
input to its output), so the derivation is genuine and single-mechanism: find k, reject the rotation
decoy (vacated end is 0, not wrapped), confirm all examples. Some of these puzzles carry an overfit
per-bit / neighbour rule in solutions.rule (a function that merely happens to fit the rows); we ignore
it and derive the minimal shift straight from the examples."""
import re

from skeleton import Skeleton
from common import gate


def _parse(prompt):
    exs, q = [], None
    for ln in prompt.splitlines():
        m = re.match(r"\s*([01]+)\s*->\s*([01]+)\s*$", ln)
        if m:
            exs.append((m.group(1), m.group(2)))
        qm = re.search(r"determine the output for:\s*([01]+)", ln) or re.search(r"for:\s*([01]+)\s*$", ln)
        if qm:
            q = qm.group(1)
    return exs, q


def _shr(s, k):
    return "0" * k + s[:len(s) - k]


def _shl(s, k):
    return s[k:] + "0" * k


def _ror(s, k):
    k %= len(s)
    return s[len(s) - k:] + s[:len(s) - k]


def _rol(s, k):
    k %= len(s)
    return s[k:] + s[:k]


def render(pid, rule, prompt, answer, ctx):
    exs, q = _parse(prompt)
    if not exs or q is None:
        return None
    n = len(exs[0][0])
    if any(len(a) != n or len(b) != n for a, b in exs) or len(q) != n:
        return None
    # derive the unique shift (direction, k) that maps EVERY input to its output (over-determined)
    found = None
    for dirn, fn in (("right", _shr), ("left", _shl)):
        for k in range(1, n):
            if all(fn(a, k) == b for a, b in exs):
                found = (dirn, k, fn)
                break
        if found:
            break
    if found is None:
        return None
    dirn, k, fn = found
    qout = fn(q, k)
    if qout != answer:
        return None
    rot = _ror if dirn == "right" else _rol

    sk = Skeleton()
    sk.analyze(f"every output is the input with all {n} bits slid the same way and the vacated end filled "
               f"in. I find how far the bits move, then check whether the vacated end wraps around or "
               f"fills with zeros.")
    sk.examples([f"{a} -> {b}" for a, b in exs])
    a0, b0 = exs[0]
    # establish the distance from example 1 by lining up a surviving run of bits
    sk.line(f"In {a0} -> {b0}, the bits keep their pattern but sit {k} place(s) to the {dirn}, and the "
            f"{k} bit(s) at the {'left' if dirn == 'right' else 'right'} end of the output are 0.")
    sk.line()
    # rotation decoy: only genuine on an example whose shifted-out bits are NOT all zero (else rotate
    # and shift coincide and the reject would be vacuous). Show it there; skip the decoy if none exists.
    rej = next(((a, b) for a, b in exs if rot(a, k) != b), None)
    tn = 1
    if rej is not None:
        ra, rb = rej
        sk.theory(tn, f"a rotation by {k} to the {dirn} (the bits that fall off wrap to the other end).")
        sk.reject(tn, f"rotate {ra} {dirn} by {k}", rot(ra, k), rb,
                  "the wrapped bits land non-zero, but the output's vacated end is 0 -> reject rotation")
        sk.line()
        tn = 2
    sk.theory(tn, f"a logical shift by {k} to the {dirn} (zeros shifted into the vacated end).")
    sk.line(f"  shift {a0} {dirn} by {k}: drop the {k} bit(s) off the {'right' if dirn == 'right' else 'left'} "
            f"end, prepend {k} zero(s) -> {b0} ok")
    sk.line()
    sk.confirm(tn, [(a, fn(a, k), b) for a, b in exs])
    sk.line("Every example reproduces exactly.")
    sk.rule(f"output = input shifted {k} place(s) to the {dirn} (logical, zero-filled).")
    sk.answer(f"shift the query {q} {dirn} by {k} -> {qout}.", qout)

    if not gate(sk, answer):
        return None
    return sk.build(), dict(dir=dirn, k=k)
