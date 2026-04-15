import re
import string
from collections import Counter


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_score(solution_str: str, ground_truth: str) -> float:
    pred = _normalize_text(solution_str)
    ref = _normalize_text(ground_truth)

    if not pred or not ref:
        return 0.0
    if pred == ref:
        return 1.0

    pred_tokens = pred.split()
    ref_tokens = ref.split()
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
