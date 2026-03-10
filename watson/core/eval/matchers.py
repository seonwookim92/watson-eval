"""
Semantic matching strategies for triple and entity evaluation.

All matchers expose the same async interface:
    await matcher.match_triples(predicted, gold)  -> int (TP count)
    await matcher.match_entities(predicted, gold) -> int (TP count)

Available matchers:
    JaccardMatcher   – word-token Jaccard overlap (fast, no dependencies)
    EmbeddingMatcher – cosine similarity via sentence-transformers (no LLM)
    LLMMatcher       – LLM batch semantic judgment (most accurate, ~1 call/sample)
"""

import json
import re
from typing import List, Optional, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

def _triple_str(t: dict) -> str:
    return (
        f"{t.get('subject', '')} {t.get('relation', '')} {t.get('object', '')}"
    ).strip()


def _entity_str(e: dict) -> str:
    return e.get("name", "").strip()


def _tokenize(text: str) -> set:
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    union = len(ta | tb)
    return len(ta & tb) / union if union > 0 else 0.0


def _greedy_match(scores, threshold: float) -> int:
    """
    Given a dict {(pred_i, gold_j): sim}, greedily assign
    predicted→gold pairs (each gold used at most once).
    Returns TP count.
    """
    # Sort by similarity descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    used_gold = set()
    tp = 0
    for (pi, gj), sim in ranked:
        if sim >= threshold and gj not in used_gold:
            tp += 1
            used_gold.add(gj)
    return tp


# ── Jaccard ───────────────────────────────────────────────────────────────────

class JaccardMatcher:
    """
    Baseline: Jaccard similarity on word tokens.
    Threshold applies to both subject AND object independently.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    async def match_triples(self, predicted: List[dict], gold: List[dict]) -> int:
        matched_gold: set = set()
        tp = 0
        for pred in predicted:
            for i, g in enumerate(gold):
                if i in matched_gold:
                    continue
                subj_sim = _jaccard(pred.get("subject", ""), g.get("subject", ""))
                obj_sim  = _jaccard(pred.get("object",  ""), g.get("object",  ""))
                if subj_sim >= self.threshold and obj_sim >= self.threshold:
                    tp += 1
                    matched_gold.add(i)
                    break
        return tp

    async def match_entities(self, predicted: List[dict], gold: List[dict]) -> int:
        matched_gold: set = set()
        tp = 0
        for pred in predicted:
            for i, g in enumerate(gold):
                if i in matched_gold:
                    continue
                if _jaccard(_entity_str(pred), _entity_str(g)) >= self.threshold:
                    tp += 1
                    matched_gold.add(i)
                    break
        return tp


# ── Embedding ─────────────────────────────────────────────────────────────────

class EmbeddingMatcher:
    """
    Cosine similarity via sentence-transformers.
    Encodes all strings in a single batch call, then matches greedily.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.75,
    ):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self._np = np

    def _cosine(self, a, b) -> float:
        np = self._np
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

    def _batch_match(self, pred_strs: List[str], gold_strs: List[str]) -> int:
        if not pred_strs or not gold_strs:
            return 0
        all_embs = self.model.encode(pred_strs + gold_strs, show_progress_bar=False)
        pred_embs = all_embs[: len(pred_strs)]
        gold_embs = all_embs[len(pred_strs) :]

        scores = {
            (pi, gj): self._cosine(pe, ge)
            for pi, pe in enumerate(pred_embs)
            for gj, ge in enumerate(gold_embs)
        }
        return _greedy_match(scores, self.threshold)

    async def match_triples(self, predicted: List[dict], gold: List[dict]) -> int:
        return self._batch_match(
            [_triple_str(t) for t in predicted],
            [_triple_str(t) for t in gold],
        )

    async def match_entities(self, predicted: List[dict], gold: List[dict]) -> int:
        return self._batch_match(
            [_entity_str(e) for e in predicted],
            [_entity_str(e) for e in gold],
        )


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLMMatcher:
    """
    LLM-based semantic matching.

    Makes ONE LLM call per (predicted_list, gold_list) pair, asking the model
    to identify which predicted items semantically match which gold items.
    Supports both local Ollama and OpenAI.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
    ):
        # temperature=0: deterministic output, no randomness in judgment
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
        elif provider == "claude" or provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=model, anthropic_api_key=api_key, temperature=0)
        else:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model=model, base_url=base_url, temperature=0)

    async def _call(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await self.llm.ainvoke([
            SystemMessage(
                content=(
                    "You are a precise semantic evaluation assistant for NLP benchmarks. "
                    "Output only valid JSON with no explanation."
                )
            ),
            HumanMessage(content=prompt),
        ])
        return response.content

    def _parse_pairs(self, raw: str) -> List[Tuple[int, int]]:
        """Extract [{pred, gold}] pairs from LLM JSON response."""
        try:
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not m:
                return []
            data = json.loads(m.group(0))
            return [
                (int(d["pred"]), int(d["gold"]))
                for d in data
                if "pred" in d and "gold" in d
            ]
        except Exception:
            return []

    def _count_pairs(
        self,
        pairs: List[Tuple[int, int]],
        n_pred: int,
        n_gold: int,
    ) -> int:
        """Deduplicate and count valid TP pairs (each gold at most once)."""
        used_gold: set = set()
        tp = 0
        for pred_i, gold_j in pairs:
            pi, gj = pred_i - 1, gold_j - 1  # 1-indexed → 0-indexed
            if 0 <= pi < n_pred and 0 <= gj < n_gold and gj not in used_gold:
                tp += 1
                used_gold.add(gj)
        return tp

    async def match_triples(self, predicted: List[dict], gold: List[dict]) -> int:
        if not predicted or not gold:
            return 0

        pred_lines = [
            f"{i+1}. ({t.get('subject','?')}, {t.get('relation','?')}, {t.get('object','?')})"
            for i, t in enumerate(predicted)
        ]
        gold_lines = [
            f"{j+1}. ({t.get('subject','?')}, {t.get('relation','?')}, {t.get('object','?')})"
            for j, t in enumerate(gold)
        ]

        prompt = (
            "Task: match each predicted triple to the semantically equivalent gold triple.\n"
            "Rules:\n"
            "- Entities are equivalent if they refer to the same real-world entity "
            "(minor name differences, abbreviations, and added qualifiers are OK).\n"
            "- Relations are equivalent if they express the same semantic relationship.\n"
            "- Each gold triple can be matched at most once.\n"
            "- Only include confident matches.\n\n"
            "Predicted triples:\n" + "\n".join(pred_lines) + "\n\n"
            "Gold triples:\n" + "\n".join(gold_lines) + "\n\n"
            'Output: JSON array [{\"pred\": <1-indexed>, \"gold\": <1-indexed>}, ...]\n'
            "Output [] if no matches."
        )

        raw = await self._call(prompt)
        pairs = self._parse_pairs(raw)
        return self._count_pairs(pairs, len(predicted), len(gold))

    async def match_entities(self, predicted: List[dict], gold: List[dict]) -> int:
        if not predicted or not gold:
            return 0

        pred_lines = [f"{i+1}. {_entity_str(e)}" for i, e in enumerate(predicted)]
        gold_lines = [f"{j+1}. {_entity_str(e)}" for j, e in enumerate(gold)]

        prompt = (
            "Task: match each predicted entity name to the semantically equivalent gold entity name.\n"
            "Rules:\n"
            "- Names are equivalent if they refer to the same real-world entity "
            "(minor differences, abbreviations, or added prefixes are OK).\n"
            "- Each gold entity can be matched at most once.\n"
            "- Only include confident matches.\n\n"
            "Predicted entities:\n" + "\n".join(pred_lines) + "\n\n"
            "Gold entities:\n" + "\n".join(gold_lines) + "\n\n"
            'Output: JSON array [{\"pred\": <1-indexed>, \"gold\": <1-indexed>}, ...]\n'
            "Output [] if no matches."
        )

        raw = await self._call(prompt)
        pairs = self._parse_pairs(raw)
        return self._count_pairs(pairs, len(predicted), len(gold))


# ── factory ───────────────────────────────────────────────────────────────────

def build_matcher(
    mode: str,
    threshold: Optional[float] = None,
    eval_provider: str = "ollama",
    eval_model: str = "llama3.1:8b",
    eval_base_url: str = "http://localhost:11434",
    eval_api_key: Optional[str] = None,
):
    """Instantiate the correct matcher from a mode string."""
    if mode == "embedding":
        t = threshold if threshold is not None else 0.75
        return EmbeddingMatcher(threshold=t)
    elif mode == "llm":
        from core.config import config
        api_key = eval_api_key
        if eval_provider == "gemini" and not api_key:
            api_key = config.GOOGLE_API_KEY
        elif eval_provider == "openai" and not api_key:
            api_key = config.OPENAI_API_KEY
        elif (eval_provider == "claude" or eval_provider == "anthropic") and not api_key:
            api_key = config.ANTHROPIC_API_KEY
            
        return LLMMatcher(
            provider=eval_provider,
            model=eval_model,
            base_url=eval_base_url,
            api_key=api_key,
        )
    else:  # jaccard (default)
        t = threshold if threshold is not None else 0.5
        return JaccardMatcher(threshold=t)
