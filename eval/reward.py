"""Reward functions for LongMas-R1.

Implements exact match reward and optional LLM-as-Judge reward.
"""

import re
import string
import logging

logger = logging.getLogger("LongMasR1-Reward")


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_reward(prediction: str, ground_truth: str | list[str]) -> float:
    """Calculate exact match reward.
    
    Returns 1.0 if the prediction matches the ground truth exactly (after normalization),
    0.0 otherwise.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer(s)
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not prediction:
        return 0.0
    
    # Handle list of ground truth answers
    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]
    
    normalized_pred = normalize_answer(prediction)
    
    for gt in ground_truths:
        if not gt:
            continue
        normalized_gt = normalize_answer(gt)
        if normalized_pred == normalized_gt:
            return 1.0
    
    return 0.0


def sub_em_reward(prediction: str, ground_truth: str | list[str]) -> float:
    prediction = normalize_answer(prediction)
    for gt in ground_truth:
        gt = normalize_answer(gt)
        if (gt in prediction) or (prediction in gt):
            return 1.0
    return 0.0