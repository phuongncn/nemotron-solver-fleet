"""Interpreter for the BitProg-cegis rule syntax (DC's CEGIS solver output): a sequence of `y = <expr>`
statements separated by `|`, where <expr> is a recursive 8-bit expression over x (input) and y (running
value): shl/shr/rotl/rotr(arg,k), NOT(arg), maj(a,b,c), ch(a,b,c), and infix XOR/AND/OR. The old
gen_bit_nonlinear_cot interpreter can't represent nested operands like NOT(rotl(x,3)), so this is a
small dedicated recursive evaluator. Pure functions; no DB."""
import re

MASK = 0xFF


def _split_args(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out]


def _split_infix(s):
    """Split top-level XOR/AND/OR (left-assoc) -> [operand, OP, operand, ...]."""
    s = s.strip()
    depth, toks, cur, i = 0, [], "", 0
    while i < len(s):
        if s[i] == "(":
            depth += 1; cur += s[i]; i += 1; continue
        if s[i] == ")":
            depth -= 1; cur += s[i]; i += 1; continue
        if depth == 0:
            m = re.match(r"\s+(XOR|AND|OR)\s+", s[i:])
            if m:
                toks.append(cur); toks.append(m.group(1)); cur = ""; i += m.end(); continue
        cur += s[i]; i += 1
    toks.append(cur)
    return toks


def _ev(e, x, y):
    e = e.strip()
    parts = _split_infix(e)
    if len(parts) > 1:
        val = _ev(parts[0], x, y)
        for k in range(1, len(parts), 2):
            rhs = _ev(parts[k + 1], x, y)
            val = {"XOR": val ^ rhs, "AND": val & rhs, "OR": val | rhs}[parts[k]]
        return val & MASK
    if e == "x":
        return x
    if e == "y":
        return y
    m = re.match(r"(\w+)\((.*)\)$", e)
    if not m:
        raise ValueError(e)
    fn, a = m.group(1).lower(), _split_args(m.group(2))
    if fn in ("shl", "shr", "rotl", "rotr"):
        v, k = _ev(a[0], x, y), int(a[1]) & 7
        if fn == "shl":
            return (v << k) & MASK
        if fn == "shr":
            return (v >> k) & MASK
        if fn == "rotl":
            return ((v << k) | (v >> (8 - k))) & MASK if k else v
        return ((v >> k) | (v << (8 - k))) & MASK if k else v          # rotr
    if fn == "not":
        return _ev(a[0], x, y) ^ MASK
    if fn == "maj":
        p = [_ev(z, x, y) for z in a]
        return (p[0] & p[1]) | (p[0] & p[2]) | (p[1] & p[2])
    if fn == "ch":
        p = [_ev(z, x, y) for z in a]
        return (p[0] & p[1]) | ((p[0] ^ MASK) & p[2])
    raise ValueError(fn)


def _stmts(prog):
    """Each `|`-separated statement -> its RHS expression (drop a leading `y =`)."""
    out = []
    for st in prog.split("|"):
        st = st.strip()
        out.append(st.split("=", 1)[1].strip() if "=" in st else st)
    return out


def run(prog, x):
    """Execute on 8-bit int x -> (out_int, [(rhs_expr, b8_value), ...] per statement)."""
    y, tr = x, []
    for rhs in _stmts(prog):
        y = _ev(rhs, x, y) & MASK
        tr.append((rhs, format(y, "08b")))
    return y & MASK, tr


_WORD = {"shl": "shift-left", "shr": "shift-right", "rotl": "rotate-left", "rotr": "rotate-right"}


def readable(e):
    """Plain-English-ish read of an expression (for the rule sentence)."""
    e = e.strip()
    parts = _split_infix(e)
    if len(parts) > 1:
        s = readable(parts[0])
        for k in range(1, len(parts), 2):
            s += f" {parts[k]} " + readable(parts[k + 1])
        return s
    if e in ("x", "y"):
        return e
    m = re.match(r"(\w+)\((.*)\)$", e)
    fn, a = m.group(1).lower(), _split_args(m.group(2))
    if fn in _WORD:
        return f"{_WORD[fn]}({readable(a[0])}, {a[1]})"
    if fn == "not":
        return f"NOT({readable(a[0])})"
    if fn == "maj":
        return "majority(" + ", ".join(readable(z) for z in a) + ")"
    if fn == "ch":
        return "select(" + ", ".join(readable(z) for z in a) + ")"
    return e


def sentence(prog):
    return "; then ".join(f"y := {readable(r)}" for r in _stmts(prog))
