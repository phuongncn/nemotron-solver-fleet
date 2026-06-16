#!/usr/bin/env python3
"""Additive `fine_category` layer on classifications — FINAL 12-subtype scheme (USER DIRECTIVE 2026-06-05).
Consistent <family>_<subtype> prefix; 0 uncat (every puzzle typed, incl unsolved tail for routing).
Does NOT touch `category` (5-cat) or train data. Reproducible: maps from solver rule + bit_shift id-list.
--apply to write, default dry-run."""
import sqlite3, sys, os

DB = "data/nemotron.db"
APPLY = "--apply" in sys.argv
BIT_SHIFT_IDS = "docs/discuss/_bit_shift_ids.txt"  # 39 pure-shift puzzles mis-grouped as bit_perbit (validated)

bit_shift = set(l.strip() for l in open(BIT_SHIFT_IDS) if l.strip()) if os.path.exists(BIT_SHIFT_IDS) else set()

# 12 fine-cats grouped by family. Rollup to 5-cat must hold (num_*=numeric, bit_*=bit, word_*=word, sym_*=symbol, rom_*=roman).
FINE_CATS = ['num_unit','num_gravity','num_equation',
             'bit_perbit','bit_rotate','bit_affine','bit_shift','bit_nonlinear',
             'word_cipher','sym_arith','sym_perm','rom_numeral']

def fine_cat(pid, category, rule):
    r = (rule or "").strip()
    if category == "numeric":
        if rule is None:          return "num_equation"   # unsolved tail = equation-guess (query-op absent)
        if r.startswith("Multiply"):  return "num_unit"
        if r.startswith("Quadratic"): return "num_gravity"
        return "num_equation"
    if category == "bit":
        if pid in bit_shift:      return "bit_shift"       # pure SHR/SHL-k (validated brute-force)
        if rule is None:          return "bit_nonlinear"   # unsolved tail = nonlinear majority/choice/XOR-mix
        if r.startswith("Rotate"):                return "bit_rotate"
        if r.startswith("f(x)") and "mod" in r:   return "bit_affine"
        return "bit_perbit"
    if category == "symbol_cipher":
        if rule is None:          return "sym_arith"       # unsolved tail = cryptarithm-guess
        if r.startswith("positional_perm"): return "sym_perm"
        return "sym_arith"
    if category == "word_cipher":  return "word_cipher"
    if category == "roman":        return "rom_numeral"
    return None

con = sqlite3.connect(DB); cur = con.cursor()
cols = [c[1] for c in cur.execute("PRAGMA table_info(classifications)")]
if "fine_category" not in cols:
    if APPLY: cur.execute("ALTER TABLE classifications ADD COLUMN fine_category TEXT"); print("[+] added column")
    else: print("[dry] would ADD COLUMN fine_category")

rows = cur.execute("""SELECT c.id, c.category, s.rule
                      FROM classifications c LEFT JOIN solutions s ON c.id=s.id""").fetchall()
from collections import Counter
updates=[]; counts=Counter(); roll=Counter()
ROLL5={'num':'numeric','bit':'bit','word':'word_cipher','sym':'symbol_cipher','rom':'roman'}
for pid, cat, rule in rows:
    fc = fine_cat(pid, cat, rule)
    updates.append((fc, pid)); counts[fc]+=1
    if fc: roll[ROLL5[fc.split('_')[0]]] += 1

print(f"bit_shift id-list loaded: {len(bit_shift)}")
if APPLY:
    cur.executemany("UPDATE classifications SET fine_category=? WHERE id=?", updates); con.commit()
    print(f"[+] updated {len(updates)} rows")
else:
    print(f"[dry] would update {len(updates)} rows")

print("\n=== fine_category counts (12-cat) ===")
for fc in FINE_CATS: print(f"  {fc:16s} {counts.get(fc,0)}")
nulls = counts.get(None,0)
print(f"  {'(NULL)':16s} {nulls}   <- must be 0")

print("\n=== AC: rollup → 5-cat (must match classifications.category) ===")
real = {r[0]:r[1] for r in cur.execute("SELECT category, COUNT(*) FROM classifications GROUP BY category")}
allok=True
for c5 in ['numeric','bit','symbol_cipher','word_cipher','roman']:
    m = roll[c5]==real[c5]; allok &= m
    print(f"  {c5:14s} rollup={roll[c5]:5d}  category={real[c5]:5d}  {'OK' if m else 'MISMATCH'}")
print(f"\n  total={sum(counts.values())}  NULL={nulls}  rollup_ok={allok}")
con.close()
