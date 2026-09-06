"""Reward functions for Parser."""

import logging
import re

logger = logging.getLogger("Parser-Reward")


def normalize_answer(answer: str) -> str:
    """Normalize answer for exact-match comparison."""
    if not answer:
        return ""

    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = " ".join(answer.split())
    return answer.strip()


def exact_match_reward(prediction: str, ground_truth: str | list[str]) -> float:
    """Return 1.0 when prediction exactly matches any normalized ground truth."""
    if not prediction:
        return 0.0

    ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    normalized_pred = normalize_answer(prediction)

    for gt in ground_truths:
        if not gt:
            continue
        if normalized_pred == normalize_answer(gt):
            return 1.0

    return 0.0
