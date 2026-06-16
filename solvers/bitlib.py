"""Shared bit-family helpers (used by cat_bit_*). Keeps the per-cat modules thin."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from gen_bit_cot_v3 import parse_examples, exec_rule       # (examples,query); faithful stored-rule executor
from verify_cot import _bit_term                            # evaluate a stored term (XOR12/CH203/MAJ235/I4/...)
from gen_cot_bit_v2 import _readable                        # human form of a stored term


def stored_terms(rule):
    """ 'bit0=XOR12; bit1=CH203; ...' -> ['XOR12','CH203',...] (index 0..7) or None."""
    t = {}
    for part in (rule or "").split(";"):
        m = re.match(r"\s*bit(\d)=(.+)$", part)
        if m:
            t[int(m.group(1))] = m.group(2).strip()
    return [t.get(i) for i in range(8)] if len(t) == 8 else None


def readable(term):
    """Readable stored term; bare digit for constants (matches gold style)."""
    m = re.fullmatch(r"C([01])", term)
    return m.group(1) if m else _readable(term)

# ── boolean op semantics (bit-level; ex import_cay_bit.py) ───────────────────────
BBIN = {"XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: 1 - (a ^ b), "AND": lambda a, b: a & b,
        "OR": lambda a, b: a | b, "NAND": lambda a, b: 1 - (a & b), "NOR": lambda a, b: 1 - (a | b),
        "NANDN": lambda a, b: 1 - (a & (1 - b)), "ORN": lambda a, b: a | (1 - b),
        "ANDN": lambda a, b: a & (1 - b), "NORN": lambda a, b: 1 - (a | (1 - b))}


def _btok(s):
    return re.findall(r"\(|\)|MAJ\(\s*in\d\s*,\s*in\d\s*,\s*in\d\s*\)|CH\(\s*in\d\s*,\s*in\d\s*,\s*in\d\s*\)"
                      r"|NOT|in\d|[01]|[A-Za-z]+", s)


def _term(t, i, IN):
    if i >= len(t):
        return None, i
    tok = t[i]
    if tok == "(":
        v, i = _expr(t, i + 1, IN)
        if v is None or i >= len(t) or t[i] != ")":
            return None, i
        return v, i + 1
    if tok == "NOT":
        v, i = _term(t, i + 1, IN)
        return (None if v is None else 1 - v), i
    m = re.fullmatch(r"(MAJ|CH)\(\s*in(\d)\s*,\s*in(\d)\s*,\s*in(\d)\s*\)", tok)
    if m:
        a, b, c = (IN[int(m.group(k))] for k in (2, 3, 4))
        return ((1 if a + b + c >= 2 else 0) if m.group(1) == "MAJ" else (b if a else c)), i + 1
    if tok in ("0", "1"):
        return int(tok), i + 1
    m = re.fullmatch(r"in(\d)", tok)
    return (IN[int(m.group(1))], i + 1) if m else (None, i)


def _expr(t, i, IN):
    v, i = _term(t, i, IN)
    while v is not None and i < len(t) and t[i] in BBIN:
        op = t[i]
        r, i = _term(t, i + 1, IN)
        if r is None:
            return None, i
        v = BBIN[op](v, r)
    return v, i


def beval(expr, IN):
    """Evaluate a per-bit boolean expression string on input bit list IN (in0..inN)."""
    toks = _btok(expr.strip())
    v, i = _expr(toks, 0, IN)
    return v if i == len(toks) else None


# ── G3/G4: derive one output column from examples, family-first (minimal param) ──
def derive_column(col_out, ins):
    """col_out[e] = output bit for example e; ins[e] = input bit list for example e.
    Try in order: single-source in[k] / NOT in[k] -> constant 0/1 -> 2-input boolean.
    Accept the FIRST that fits all examples (reject boolean when a simpler family fits).
    Returns (term_str, family) or None (needs >2-input -> caller falls back)."""
    n = len(ins[0])
    E = range(len(ins))
    for k in range(n):                                       # (1) single source
        if all(ins[e][k] == col_out[e] for e in E):
            return f"in{k}", "single"
    for k in range(n):                                       # (1b) single source, negated
        if all((1 - ins[e][k]) == col_out[e] for e in E):
            return f"NOT in{k}", "single"
    if all(c == 0 for c in col_out):                         # (2) constant
        return "0", "const"
    if all(c == 1 for c in col_out):
        return "1", "const"
    for a in range(n):                                       # (3) 2-input boolean
        for b in range(a + 1, n):
            for name, fn in BBIN.items():
                if all(fn(ins[e][a], ins[e][b]) == col_out[e] for e in E):
                    return f"in{a} {name} in{b}", "bool"
    return None


# ── whole-word shift/rotate decoy detection (THEORY-1) ──────────────────────────
def _shift_rotate_cands(n):
    """Basic shift/rotate candidates for n-bit strings."""
    cands = {"identity (output = input)": lambda s: s}
    for k in range(1, n):
        cands[f"left shift by {k}"] = (lambda k: lambda s: s[k:] + "0" * k)(k)
        cands[f"right shift by {k}"] = (lambda k: lambda s: "0" * k + s[:-k])(k)
        cands[f"rotate left by {k}"] = (lambda k: lambda s: s[k:] + s[:k])(k)
        cands[f"rotate right by {k}"] = (lambda k: lambda s: s[-k:] + s[:-k])(k)
    return cands


def best_simple_transform(examples):
    """Return (name, fn(str)->str) of the single whole-word transform that matches the most
    output columns — used as the THEORY-1 decoy (it is shown FAILING)."""
    n = len(examples[0][0])
    cands = _shift_rotate_cands(n)

    def score(fn):
        return sum(1 for inp, out in examples for j in range(n) if fn(inp)[j] == out[j])
    return max(cands.items(), key=lambda kv: score(kv[1]))


def all_global_transforms(n=8):
    """Extended catalog of whole-word transforms for THEORY 1.5: rotate, shift, XOR-with-constant,
    XOR-with-shifted-self, NOT, reverse-bits, swap-nibbles. Returns {name: fn(str)->str}."""
    cands = _shift_rotate_cands(n)

    def _xor(s, mask):
        return "".join(str(int(a) ^ int(b)) for a, b in zip(s, mask))

    cands["NOT (complement)"] = lambda s: "".join("1" if c == "0" else "0" for c in s)
    cands["reverse bits"] = lambda s: s[::-1]
    if n == 8:
        cands["swap nibbles"] = lambda s: s[4:] + s[:4]

    for k in range(1, n):
        cands[f"XOR with left-shift by {k}"] = (lambda k: lambda s: _xor(s, s[k:] + "0" * k))(k)
        cands[f"XOR with right-shift by {k}"] = (lambda k: lambda s: _xor(s, "0" * k + s[:-k]))(k)
        cands[f"XOR with rotate-left by {k}"] = (lambda k: lambda s: _xor(s, s[k:] + s[:k]))(k)

    return cands


def global_transforms_ranked(examples):
    """All global transforms ranked by how many (example, column) pairs they match. Returns
    [(name, fn, n_match)] sorted by n_match descending, then name for stability."""
    n = len(examples[0][0])
    cands = all_global_transforms(n)
    total = n * len(examples)

    def score(fn):
        return sum(1 for inp, out in examples for j in range(n) if fn(inp)[j] == out[j])
    ranked = [(nm, fn, score(fn)) for nm, fn in cands.items()]
    ranked.sort(key=lambda t: (-t[2], t[0]))
    return ranked


def neighbor_window(s, i, wrap):
    n = len(s)
    l = s[(i - 1) % n] if wrap else (s[i - 1] if i - 1 >= 0 else "0")
    r = s[(i + 1) % n] if wrap else (s[i + 1] if i + 1 < n else "0")
    return (int(l), int(s[i]), int(r))
