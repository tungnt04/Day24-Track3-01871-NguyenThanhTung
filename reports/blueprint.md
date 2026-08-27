# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Thanh Tùng
**Ngày:** 2026-08-27
**Pipeline under test:** Day 18 (hierarchical chunk + M5 enrichment + hybrid BM25/dense + cross-encoder rerank top-3 + gpt-4o-mini)

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (P95 ≈ 11ms)
[Presidio PII Scan]                       ← regex + spaCy en_core_web_lg, local
    │ block if: VN_CCCD / VN_PHONE / EMAIL_ADDRESS / PHONE_NUMBER
    │ action:   reject 400 + "PII detected in query" + log
    ▼ (P95 ≈ 3970ms — LLM call)
[NeMo Input Rail: self check input]       ← gpt-4o-mini, prompt trong prompts.yml
    │ block if: jailbreak / prompt injection / off-topic / yêu cầu lộ PII
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M5 Enrich → M2 Hybrid Search → M3 Rerank(top-3) → GPT-4o-mini
    ▼
[NeMo Output Rail: self check output]     ← gpt-4o-mini
    │ flag if:  PII trong response / lộ lương cá nhân / lộ system prompt
    │ action:   thay bằng "Vui lòng liên hệ phòng Nhân sự" + log
    ▼
User Response
```

---

## Latency Budget

*(Đo bằng `measure_p95_latency()` — Task 12, n_runs = 10 trên adversarial inputs)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | OK? |
|---|---|---|---|---|---|
| Presidio PII | 10.1 | 10.6 | 10.6 | <10ms | ~borderline |
| NeMo Input Rail | 951 | 3974 | 3974 | <300ms | ❌ |
| NeMo Output Rail | (≈ tương đương input) | — | — | <300ms | ❌ |
| **Total Guard (Presidio + NeMo input)** | 962 | **3984** | 3984 | **<500ms** | ❌ |

**Budget OK?** ❌ **No** — vượt xa mức 500ms.

**Comment — bottleneck & cách tối ưu:**
> NeMo input rail chiếm ~99.7% latency vì mỗi request là **một LLM call gpt-4o-mini đồng bộ**
> (P95 gần 4s do cả biến động mạng lẫn cold-start). Presidio (local regex) chỉ ~11ms — không đáng kể.
> Hướng tối ưu cho production:
> 1. Thay `self_check_input` bằng **classifier nhỏ / local model** (llama-guard, bge-reranker
>    fine-tuned, hoặc DistilBERT intent classifier) — mục tiêu <100ms.
> 2. **Chạy song song** Presidio + input rail thay vì tuần tự (Presidio không cần chờ).
> 3. Cache kết quả rail cho các query lặp lại (LRU theo hash).
> 4. Streaming: cho RAG bắt đầu ngay, chạy output rail trên bản nháp cuối.

---

## CI Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
jobs:
  eval-guard:
    steps:
      - name: RAGAS Quality Gate
        run: python src/phase_a_ragas.py
        # FAIL nếu: faithfulness (50q) < 0.75  HOẶC  avg_score < 0.70

      - name: LLM Judge Reliability Gate
        run: python src/phase_b_judge.py
        # FAIL nếu: Cohen κ < 0.6  HOẶC  position_bias_rate > 0.35

      - name: Adversarial Guardrail Gate
        run: pytest tests/test_phase_c.py -k "adversarial"
        # FAIL nếu: pass rate < 90% (18/20)

      - name: Latency Gate (warn-only ở giai đoạn này)
        run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
        # WARN nếu P95 total > 500ms (chưa block merge cho tới khi thay classifier)
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample 50q) | < 0.70 | Page on-call, freeze deploy |
| RAGAS avg_score theo distribution | multi_hop hoặc adversarial < 0.65 | Review prompt + version filter |
| Adversarial block rate (shadow traffic) | < 90% | Cập nhật prompt self_check + Presidio pattern |
| Guard P95 latency | > 4500ms | Scale / chuyển sang local classifier |
| LLM Judge κ (weekly recalibration) | < 0.6 | Chấm lại rubric, refresh human labels |
| PII detected count | spike > 10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q, weighted) | **0.787** |
| RAGAS faithfulness (50q, weighted) | **0.676** — ❌ dưới gate 0.75 |
| — factual / multi_hop / adversarial avg_score | 0.900 / 0.728 / 0.677 |
| Worst metric | **faithfulness** (20/50 câu; 13 ở multi_hop) |
| Dominant failure distribution | **multi_hop** (avg 0.728, tie-break theo avg_score) |
| Cohen's κ (LLM judge vs human, 10q) | **1.0** (almost perfect; 0.29 nếu dùng quy tắc nhãn khắt khe hơn) |
| Position bias rate | 0.20 |
| Verbosity bias | 1.0 (100% — winner luôn là câu dài hơn) |
| Adversarial pass rate | **20 / 20** (pii 5/5, jailbreak 5/5, off-topic 5/5, prompt-injection 5/5) |
| Guard P95 latency (Presidio + NeMo input) | **3984 ms** — ❌ vượt budget 500ms |
| — Presidio P95 | 10.6 ms |
| — NeMo P95 | 3974 ms |

**Bonus checklist:** κ > 0.6 ✅ · adversarial ≥ 18/20 ✅ (20/20) · adversarial avg (0.677) < factual avg (0.900) ✅

---

## Nhận xét & Cải tiến

> **Hoạt động tốt:** khâu retrieval của Day 18 rất ổn (`context_precision` ≥ 0.94 ở mọi distribution),
> và guardrail chặn 100% adversarial nhờ chuyển từ Colang keyword-matching (chỉ 4/20, không có LLM call)
> sang `self_check_input`/`self_check_output` (LLM call trực tiếp, hiểu tiếng Việt).
>
> **Cần cải thiện:** (1) **Faithfulness ở multi-hop/adversarial** — LLM lấy đúng số liệu nhưng tính
> sai hoặc trộn phiên bản chính sách (v2023/v2024, v1/v2); cần metadata `version` + filter phiên bản
> hiện hành trước khi đưa vào prompt, và prompt yêu cầu trình bày từng bước tính toán.
> (2) **Latency** — NeMo rail 4s là không khả thi cho production; phải thay bằng local classifier.
> (3) **Verbosity bias 100%** của judge — phải length-normalize trước khi chấm.
>
> **Nếu deploy production thật:** chạy Presidio và input-rail song song; thay self_check bằng
> llama-guard chạy local (<100ms); thêm version-aware retrieval filter; giữ RAGAS + κ chạy nightly
> trên shadow traffic với gate cứng ở faithfulness 0.75; log mọi lần guard chặn để tinh chỉnh prompt.
