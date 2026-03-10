"""
Evaluation metrics for CTI triple and entity extraction.

evaluate_triples / evaluate_entities accept any matcher from core.eval.matchers.
Aggregate functions compute macro and micro averages across samples.
"""

from typing import List


def _compute_prf(tp: int, predicted: int, gold: int) -> dict:
    precision = tp / predicted if predicted > 0 else 0.0
    recall    = tp / gold      if gold      > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp":        tp,
        "predicted": predicted,
        "gold":      gold,
    }


async def evaluate_triples(
    predicted: List[dict],
    gold: List[dict],
    matcher=None,
) -> dict:
    """
    Compute precision / recall / F1 for triple extraction.

    matcher: any matcher from core.eval.matchers (JaccardMatcher,
             EmbeddingMatcher, LLMMatcher). Defaults to JaccardMatcher(0.5).
    """
    if matcher is None:
        from core.eval.matchers import JaccardMatcher
        matcher = JaccardMatcher(threshold=0.5)
    tp = await matcher.match_triples(predicted, gold)
    return _compute_prf(tp, len(predicted), len(gold))


async def evaluate_entities(
    predicted: List[dict],
    gold: List[dict],
    matcher=None,
) -> dict:
    """Compute precision / recall / F1 for entity extraction."""
    if matcher is None:
        from core.eval.matchers import JaccardMatcher
        matcher = JaccardMatcher(threshold=0.5)
    tp = await matcher.match_entities(predicted, gold)
    return _compute_prf(tp, len(predicted), len(gold))


def aggregate_metrics(sample_metrics: List[dict]) -> dict:
    """Compute macro-averaged and micro-averaged metrics across samples."""
    if not sample_metrics:
        return {}

    result = {}
    for k in ["precision", "recall", "f1"]:
        vals = [m[k] for m in sample_metrics if k in m]
        result[f"macro_{k}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    tp   = sum(m.get("tp",        0) for m in sample_metrics)
    pred = sum(m.get("predicted", 0) for m in sample_metrics)
    gold = sum(m.get("gold",      0) for m in sample_metrics)
    micro = _compute_prf(tp, pred, gold)
    result["micro_precision"] = micro["precision"]
    result["micro_recall"]    = micro["recall"]
    result["micro_f1"]        = micro["f1"]

    return result
