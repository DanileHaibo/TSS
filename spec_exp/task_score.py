"""Per-task official quality metrics (no chat / MT-Bench LLM judge)."""
from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def output_equal(hypothesis: str, baseline: str) -> float:
    """Spec-Bench equal.py: 1.0 if greedy outputs are byte-identical."""
    return 1.0 if (hypothesis or "").strip() == (baseline or "").strip() else 0.0


# --- GSM8K accuracy ---------------------------------------------------------


def _extract_gsm8k_answer(text: str) -> str | None:
    text = (text or "").replace(",", "")
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1].strip()
    return (text or "").strip() or None


def gsm8k_accuracy(pred: str, ref: str) -> float:
    pa, ra = _extract_gsm8k_answer(pred), _extract_gsm8k_answer(ref)
    if pa is None or ra is None:
        return 0.0
    try:
        return 1.0 if abs(float(pa) - float(ra)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if pa.strip().lower() == ra.strip().lower() else 0.0


# --- HumanEval pass@1 -------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    text = text or ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).rstrip("\n")
    return text.rstrip("\n")


_HUMAN_EVAL_BY_ENTRY: dict[str, str] | None = None


def _humaneval_entry_index() -> dict[str, str]:
    global _HUMAN_EVAL_BY_ENTRY
    if _HUMAN_EVAL_BY_ENTRY is None:
        from human_eval.data import read_problems

        _HUMAN_EVAL_BY_ENTRY = {p["entry_point"]: tid for tid, p in read_problems().items()}
    return _HUMAN_EVAL_BY_ENTRY


def _resolve_humaneval_task_id(request_id: str, question: str | None) -> str | None:
    m = re.search(r"HumanEval/(\d+)", request_id)
    if m:
        return f"HumanEval/{m.group(1)}"
    if question:
        fm = re.search(r"def\s+(\w+)\s*\(", question)
        if fm:
            return _humaneval_entry_index().get(fm.group(1))
    return None


def humaneval_pass_at_1(hypothesis: str, request_id: str, *, question: str | None = None) -> float:
    task_key = _resolve_humaneval_task_id(request_id, question)
    if not task_key:
        return math.nan
    from human_eval.data import read_problems
    from human_eval.execution import check_correctness

    problems = read_problems()
    if task_key not in problems:
        return math.nan
    completion = _strip_code_fence(hypothesis)
    if not completion:
        return 0.0
    try:
        result = check_correctness(problems[task_key], completion, timeout=3.0)
        return 1.0 if result.get("passed") else 0.0
    except Exception as exc:
        logger.warning("human_eval %s: %s", task_key, exc)
        return 0.0


# --- ROUGE (summarization) --------------------------------------------------


def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {
        "rouge1": float(s["rouge1"].fmeasure),
        "rouge2": float(s["rouge2"].fmeasure),
        "rougeL": float(s["rougeL"].fmeasure),
    }


# --- BLEU (translation) -----------------------------------------------------


def bleu_score(pred: str, ref: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_bleu(pred, [ref]).score) / 100.0


# --- QA / RAG: EM + F1 ------------------------------------------------------


def _normalize_qa(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_em(pred: str, ref: str) -> float:
    return 1.0 if _normalize_qa(pred) == _normalize_qa(ref) else 0.0


def qa_f1(pred: str, ref: str) -> float:
    pt = _normalize_qa(pred).split()
    rt = _normalize_qa(ref).split()
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    common: dict[str, int] = {}
    for t in rt:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in pt:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    prec = num_same / len(pt)
    rec = num_same / len(rt)
    return 2 * prec * rec / (prec + rec)


def multiple_choice_accuracy(pred: str, ref: str) -> float:
    """Extract an A-D answer from a short generation and compare exactly."""
    text = (pred or "").strip().upper()
    patterns = (
        r"(?:FINAL\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*[\(\[]?([A-D])\b",
        r"^\s*[\(\[]?([A-D])[\)\].,:]?\s*$",
    )
    answer = None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            break
    if answer is None:
        standalone = re.findall(r"\b([A-D])\b", text)
        answer = standalone[-1] if standalone else None
    return 1.0 if answer == (ref or "").strip().upper() else 0.0


def rag_ref_text(ref: str) -> str:
    try:
        items = ast.literal_eval(ref)
        if isinstance(items, list):
            return " ".join(str(x) for x in items)
    except (SyntaxError, ValueError):
        pass
    return ref


# --- Unified API ------------------------------------------------------------


@dataclass
class ScoreDetail:
    primary: float
    metric: str
    submetrics: dict[str, float] = field(default_factory=dict)


def score_generation_detail(
    *,
    category: str,
    hypothesis: str,
    reference: str | None,
    request_id: str | None = None,
    question: str | None = None,
) -> ScoreDetail:
    hyp = (hypothesis or "").strip()
    if not hyp:
        return ScoreDetail(0.0, "empty")

    if reference is None:
        return ScoreDetail(math.nan, "no_reference")
    if isinstance(reference, list):
        reference = str(reference)
    ref = (reference or "").strip()
    if not ref:
        return ScoreDetail(math.nan, "no_reference")
    if category == "math_reasoning":
        acc = gsm8k_accuracy(hyp, ref)
        return ScoreDetail(acc, "accuracy", {"accuracy": acc})
    if category == "code":
        p1 = humaneval_pass_at_1(hyp, request_id or "", question=question)
        return ScoreDetail(p1, "pass@1", {"pass@1": p1})
    if category == "translation":
        b = bleu_score(hyp, ref)
        return ScoreDetail(b, "bleu", {"bleu": b})
    if category == "summarization":
        r = rouge_scores(hyp, ref)
        return ScoreDetail(r["rougeL"], "rougeL", r)
    if category == "qa":
        em, f1 = qa_em(hyp, ref), qa_f1(hyp, ref)
        return ScoreDetail(f1, "f1", {"em": em, "f1": f1})
    if category == "multiple_choice":
        acc = multiple_choice_accuracy(hyp, ref)
        return ScoreDetail(acc, "accuracy", {"accuracy": acc})
    if category == "rag":
        rt = rag_ref_text(ref)
        em, f1 = qa_em(hyp, rt), qa_f1(hyp, rt)
        return ScoreDetail(f1, "f1", {"em": em, "f1": f1})
    return ScoreDetail(math.nan, "unknown")


def score_generation(**kwargs: Any) -> float:
    return score_generation_detail(**kwargs).primary


def mean_task_score(
    *,
    category: str,
    hypotheses: dict[str, str],
    references: dict[str, str | None],
    questions: dict[str, str] | None = None,
    baseline_hypotheses: dict[str, str] | None = None,  # ignored; kept for callers
    **_kwargs: Any,
) -> float:
    return float(
        mean_task_score_detailed(
            category=category,
            hypotheses=hypotheses,
            references=references,
            questions=questions,
        )["mean_score"]
    )


def mean_task_score_detailed(
    *,
    category: str,
    hypotheses: dict[str, str],
    references: dict[str, str | None],
    questions: dict[str, str] | None = None,
    baseline_hypotheses: dict[str, str] | None = None,  # ignored
    **_kwargs: Any,
) -> dict[str, Any]:
    primaries: list[float] = []
    per_item: dict[str, dict[str, Any]] = {}
    sub_sums: dict[str, float] = {}
    sub_counts: dict[str, int] = {}

    for rid, hyp in hypotheses.items():
        detail = score_generation_detail(
            category=category,
            hypothesis=hyp,
            reference=references.get(rid),
            request_id=rid,
            question=questions.get(rid) if questions else None,
        )
        per_item[rid] = {
            "score": detail.primary,
            "metric": detail.metric,
            **detail.submetrics,
        }
        if not math.isnan(detail.primary):
            primaries.append(detail.primary)
        for k, v in detail.submetrics.items():
            if not math.isnan(v):
                sub_sums[k] = sub_sums.get(k, 0.0) + v
                sub_counts[k] = sub_counts.get(k, 0) + 1

    metric = next(iter(per_item.values()))["metric"] if per_item else "none"

    return {
        "mean_score": sum(primaries) / len(primaries) if primaries else math.nan,
        "metric": metric,
        "num_scored": len(primaries),
        "submetrics": {k: sub_sums[k] / sub_counts[k] for k in sub_sums},
        "per_item": per_item,
    }


def mean_output_equal(
    hypotheses: dict[str, str],
    baselines: dict[str, str],
) -> float:
    scores = [output_equal(hypotheses[k], baselines[k]) for k in hypotheses if k in baselines]
    return sum(scores) / len(scores) if scores else math.nan
