"""
OFFICIAL Kaggle grading logic — copied VERBATIM from the competition metric notebook
(https://www.kaggle.com/code/metric/nvidia-nemotron-metric, version 15).

Use these EVERYWHERE we score (mine_base, eval_compare, ...) so our local numbers match Kaggle.

KEY FACTS (from the real code):
- Extraction: LAST non-empty `\\boxed{...}` (handles answers containing '}'); else "final answer is:"
  patterns; else LAST numeric `-?\\d+(\\.\\d+)?`; else last non-empty line. (raw text = full generation
  incl. <think> reasoning — Kaggle uses NO reasoning parser.)
- verify():
    * binary string (^[01]+$)  -> STRICT case-insensitive string match (NO tolerance)  [bit puzzles]
    * float-parseable          -> math.isclose(rel_tol=1e-2, abs_tol=1e-5)               [numeric]
    * else                     -> case-insensitive string match                          [roman/word/symbol]
- Run params (from competition overview): max_tokens=7680, temperature=0.0, top_p=1.0,
  max_num_seqs=64, gpu_memory_utilization=0.85, max_model_len=8192, dtype=auto (BF16),
  enable_prefix_caching=True, enable_chunked_prefill=True.
  (NOTE: score() function DEFAULTS in the code differ — max_tokens=3584, temperature=1.0 — but the
   competition runs with the overview params above, which override the defaults.)
"""
import re
import math


def extract_final_answer(text):
    r"""VERBATIM from the official metric."""
    if text is None:
        return 'NOT_FOUND'
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        ms = re.findall(pattern, text, re.IGNORECASE)
        if ms:
            return ms[-1].strip()
    ms = re.findall(r'-?\d+(?:\.\d+)?', text)
    if ms:
        return ms[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer, predicted):
    """VERBATIM from the official metric. True if `predicted` is graded correct vs `stored_answer`."""
    stored_answer = (stored_answer or '').strip()
    predicted = (predicted or '').strip()
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
