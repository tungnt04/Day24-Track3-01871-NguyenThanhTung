from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY or None)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:  # network / key / parse — degrade gracefully
        print(f"  ⚠️  pairwise_judge failed: {e}")
        return {"winner": "tie", "reasoning": "", "scores": {"A": 0.0, "B": 0.0}}

    winner = str(data.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    winner = "tie" if winner == "TIE" else winner
    scores = data.get("scores", {}) or {}
    clean_scores = {
        k: max(0.0, min(1.0, float(scores.get(k, 0.0) or 0.0)))
        for k in ("A", "B")
    }
    return {"winner": winner, "reasoning": str(data.get("reasoning", "")), "scores": clean_scores}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    position_consistent = (pass1["winner"] == winner_pass2)
    final = pass1["winner"] if position_consistent else "tie"

    p2_scores = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2={"A": p2_scores.get("B", 0.0), "B": p2_scores.get("A", 0.0)},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    n = len(judge_labels)
    if n == 0 or n != len(human_labels):
        return 0.0

    p_o = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    pj1, ph1 = judge_labels.count(1) / n, human_labels.count(1) / n
    pj0, ph0 = judge_labels.count(0) / n, human_labels.count(0) / n
    p_e = pj1 * ph1 + pj0 * ph0
    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0,
                "position_bias_count": 0,
                "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0,
                                      "total_decisive": 0},
                "interpretation": "Không có dữ liệu để đánh giá bias."}

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
                              "b_wins_b_longer": b_wins_b_longer,
                              "total_decisive": decisive},
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def _load_ground_truths() -> dict[int, str]:
    """Map question_id → ground_truth từ answers_50q.json (fallback: test_set_50q.json)."""
    from config import ANSWERS_PATH, TEST_SET_PATH
    for path in (ANSWERS_PATH, TEST_SET_PATH):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {item["id"]: item["ground_truth"] for item in data}
    return {}


def run_judge_on_human_set() -> dict:
    """Chạy LLM judge trên 10 câu có nhãn người → Cohen κ + bias report.

    Với mỗi câu:  A = model_answer (đã có sẵn),  B = ground_truth.
    judge_label = 1 (good) nếu A thắng hoặc hoà, 0 nếu B thắng.
    """
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    ground_truths = _load_ground_truths()

    human_labels, judge_labels, judge_results, per_q = [], [], [], []
    for item in human_data:
        qid = item["question_id"]
        a_model = item["model_answer"]
        a_gt = ground_truths.get(qid, item.get("human_note", ""))
        jr = swap_and_average(item["question"], a_model, a_gt)
        # judge_label = 1 (good) nếu model_answer (A) thắng / hoà so với ground_truth,
        # HOẶC đạt điểm tuyệt đối đủ cao (≥0.6) — con người đánh giá model_answer là
        # "đúng nhưng ngắn hơn gold" vẫn là 1, nên chỉ so winner thôi thì quá khắt khe.
        score_a = 0.5 * (jr.scores_pass1.get("A", 0.0) + jr.scores_pass2.get("A", 0.0))
        judge_label = 1 if (jr.final_winner != "B" or score_a >= 0.6) else 0

        human_labels.append(int(item["human_label"]))
        judge_labels.append(judge_label)
        judge_results.append(jr)
        per_q.append({
            "question_id": qid,
            "question": item["question"],
            "winner_pass1": jr.winner_pass1,
            "winner_pass2": jr.winner_pass2,
            "final_winner": jr.final_winner,
            "position_consistent": jr.position_consistent,
            "score_a": round(score_a, 3),
            "judge_label": judge_label,
            "human_label": int(item["human_label"]),
        })
        print(f"  Q{qid}: judge={judge_label} human={item['human_label']} "
              f"(final={jr.final_winner}, consistent={jr.position_consistent})")

    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(judge_results)
    agreement = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / len(human_labels)

    def _kappa_label(k: float) -> str:
        if k < 0:      return "poor"
        if k < 0.2:    return "slight"
        if k < 0.4:    return "fair"
        if k < 0.6:    return "moderate"
        if k < 0.8:    return "substantial"
        return "almost perfect"

    return {
        "n_questions": len(human_labels),
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": _kappa_label(kappa),
        "raw_agreement": round(agreement, 4),
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "bias_report": bias,
        "per_question": per_q,
    }


if __name__ == "__main__":
    print("Running LLM-as-Judge on 10 human-labeled questions...\n")
    report = run_judge_on_human_set()

    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nCohen's κ = {report['cohen_kappa']} ({report['kappa_interpretation']})")
    print(f"Raw agreement = {report['raw_agreement']}")
    print(f"Position bias rate = {report['bias_report']['position_bias_rate']}")
    print(f"Verbosity bias = {report['bias_report']['verbosity_bias']}")
    print("\nPhase B report saved → reports/judge_results.json")
