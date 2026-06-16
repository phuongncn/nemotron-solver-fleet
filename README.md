# Nemotron Solver Fleet

When the forums said certain puzzles were unsolvable, I kept digging. The organizers promised every puzzle has a logic — and they were right.

This is the deterministic solver fleet that cracked **9,500 out of 9,500** training puzzles in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge). 100% coverage. Zero LLM involvement. Pure code.

The solvers found the hidden rule behind each puzzle. The generators turned those rules into verbose Chain-of-Thought that teaches a model *how to search*, not just *what to answer*. The gates verified every output before it touched the training set.

**Result: Private 0.86 · Public 0.85 · Rank 166/4,354 · Silver Medal**

---

## What's Inside

### `/solvers` — Rule Discovery

Each puzzle category has its own solver that reverse-engineers the hidden transformation:

| Solver | Category | Approach |
|--------|----------|----------|
| `numeric_solver.py` | Numeric (equation, gravity, unit) | Brute-force operation space — including missing-op recovery that cracked +35 "unsolvable" puzzles |
| `symbol_solver_dfs.c` | Symbol cipher (cryptarithm) | C-based DFS over base × endianness × operator combinations. Fast enough to sweep the full search space |
| `bit_cegis.py` + `cat_bit_*.py` | Bit operations | Per-bit function analysis (XOR, AND, OR, shifts, rotations) + Z3 CEGIS for non-linear combinations |
| `near_miss_scan.py` | All categories | Brute-forces a rule per unsolved puzzle; when the rule fits all examples but the query is a near-miss, classifies the puzzle as suspect-wrong-answer |
| `cat_num_equation.py` | Numeric equation | Structured equation solver with forward/backward operation recovery |
| `cat_num_gravity.py` | Gravity/physics | Pattern matching for gravity-style numeric puzzles |
| `cat_num_unit.py` | Unit conversion | Unit conversion pattern recognition |

### `/generators` — Chain-of-Thought Generation

Deterministic CoT generators — no LLM, no hallucination, 100% mechanically correct:

| Generator | What it produces |
|-----------|-----------------|
| `gen_symbol_cot.py` | Verbose derive-style: try ~10 wrong rules (real computation, stop at first failing example), then verify the correct rule on ALL examples |
| `gen_bit_cot.py` | Per-bit serial analysis with exhaustive operator enumeration |
| `gen_word_cot.py` | Cipher identification + key derivation with failure cases |
| `gen_numeric_cot.py` | Operation search with verify-all-examples guard |

The key insight: **failure cases first**. Each CoT shows ~10 wrong hypotheses with real computed values before deriving the correct rule. This teaches the model to reject wrong answers — not just assert right ones.

### `/gates` — Quality Verification

Every training example passes three gates before touching the model:

| Gate | What it catches |
|------|-----------------|
| `verify_cot.py` | Boxed answer ≠ gold answer, arithmetic errors, exploration without commitment, format violations |
| `detect_hallucination.py` | Fabricated values, invented operations, numbers not derivable from the puzzle |
| `validate_rules.py` | Re-executes every rule on all examples + query — if it doesn't reproduce, it doesn't train |

### `/utils` — Scoring & Classification

| Tool | Purpose |
|------|---------|
| `kaggle_metric.py` | Official Kaggle grading: numeric ±1% isclose, bit strict, roman/word/symbol string match, last-boxed extraction |
| `apply_fine_category.py` | Fine-grained classification (bit → perbit/rotate/shift/affine/nonlinear, symbol → arith/perm, etc.) |

---

## The Pipeline

```
Puzzle ──→ Solver (find the rule) ──→ Generator (verbose CoT from rule)
                                          │
                                     Gates (verify)
                                          │
                                     Training data
```

1. **Solve**: deterministic solver finds the exact rule for each puzzle
2. **Generate**: mechanical CoT with failure-case-first teaching (~3K–7K tokens)
3. **Verify**: three independent gates catch errors before training
4. **Train**: 9,500 verified examples + curriculum weighting → LoRA adapter

## The Lesson

I solved every puzzle. The score was 0.76. My clean, efficient CoT was teaching the model WHAT to answer, not HOW to search. Verbose exhaustive-search CoT — messy, long, full of wrong attempts — jumped it to 0.85.

*Solving the puzzle is not the same as teaching someone to solve it.*

---

## Related Resources

- **Kaggle Notebook:** [fususu-solution-v9](https://www.kaggle.com/code/phuongncn/fususu-solution-v9)
- **Adapter (LoRA weights):** [nemotron-v9b-curric-adapter](https://www.kaggle.com/datasets/phuongncn/nemotron-v9b-curric-adapter)
- **Full Solutions + Twins:** [nemotron-fususu-solutions](https://www.kaggle.com/datasets/phuongncn/nemotron-fususu-solutions)
- **Competition:** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

## License

MIT
