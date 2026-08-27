# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Thanh Tùng
**Ngày:** 2026-08-27
**Test set:** `test_set_50q.json` (20 factual / 20 multi_hop / 10 adversarial)
**Pipeline:** Day 18 — hierarchical chunking + M5 enrichment + hybrid BM25/dense + cross-encoder rerank (top-3) + gpt-4o-mini

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.933 | 0.474 | 0.567 |
| answer_relevancy | 0.809 | 0.676 | 0.518 |
| context_precision | 0.983 | 0.942 | 0.942 |
| context_recall | 0.875 | 0.821 | 0.683 |
| **avg_score** | **0.900** | **0.728** | **0.677** |

**Quan sát chính:** retrieval gần như luôn tốt (`context_precision` ≥ 0.94 ở cả 3 distribution),
nhưng chất lượng câu trả lời sụp đổ ở `multi_hop` và `adversarial` — chủ yếu do **faithfulness**
(multi_hop 0.47, adversarial 0.57 so với factual 0.93). Nói cách khác: pipeline *tìm đúng tài liệu*
nhưng LLM *suy luận sai / bịa thêm* khi câu hỏi cần tính toán hoặc khi corpus có nhiều phiên bản chính sách.

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question (tóm tắt) | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | So sánh yêu cầu mật khẩu policy v1.0 vs v2.0 (độ dài, chu kỳ đổi) | 0.250 | faithfulness |
| 2 | adversarial | Manager có được dùng VPN cá nhân (NordVPN) khi WFH? | 0.333 | faithfulness |
| 3 | multi_hop | Manager thâm niên 12 năm: tổng phụ cấp/tháng + số ngày phép | 0.375 | faithfulness |
| 4 | adversarial | Bao lâu phải đổi mật khẩu một lần? (bẫy version v1 vs v2) | 0.396 | faithfulness |
| 5 | adversarial | Nghỉ bao nhiêu ngày phép năm? (bẫy v2023 vs v2024) | 0.417 | faithfulness |
| 6 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? (negation trap) | 0.417 | faithfulness |
| 7 | factual | Nam nhân viên nghỉ bao nhiêu ngày khi vợ sinh con? | 0.500 | faithfulness |
| 8 | multi_hop | Senior 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm? | 0.542 | answer_relevancy |
| 9 | multi_hop | Công tác trong nước 2 ngày, khách sạn 1.5tr/đêm — tính công tác phí | 0.620 | faithfulness |
| 10 | multi_hop | Tạm ứng 4tr vs 7tr: quy trình phê duyệt khác nhau thế nào | 0.622 | faithfulness |

→ 9/10 câu tệ nhất có `worst_metric = faithfulness`. 8/10 thuộc multi_hop hoặc adversarial.

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có `worst_metric` = row, thuộc `distribution` = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 2 | 13 | 5 | **20** |
| answer_relevancy | 13 | 4 | 1 | 18 |
| context_recall | 4 | 3 | 4 | 11 |
| context_precision | 1 | 0 | 0 | 1 |
| **Total** | 20 | 20 | 10 | 50 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** `multi_hop`
(multi_hop và factual cùng có 20 câu "worst_metric", nhưng multi_hop có avg_score 0.728 « factual 0.900,
nên multi_hop mới là nơi chất lượng thực sự thấp — tie-break theo avg_score.)

**Dominant metric:** `faithfulness` (20/50 câu, và 13 trong số đó nằm ở multi_hop)

**Lý do phân tích:**

> `context_precision`/`context_recall` cao đều → khâu retrieval của Day 18 (hybrid + rerank) hoạt động tốt,
> tài liệu đúng gần như luôn nằm trong top-3. Vấn đề nằm ở **bước sinh câu trả lời**:
> 1. **Multi-hop cần tính toán** (công tác phí, phụ cấp theo thâm niên, phép tích lũy): gpt-4o-mini
>    lấy đúng số liệu gốc nhưng cộng/nhân sai hoặc ghép nhầm điều kiện giữa 2 tài liệu → RAGAS
>    faithfulness phạt vì claim trong câu trả lời không truy vết được về context.
> 2. **Adversarial = version conflict**: corpus chứa `nghi_phep_nam_v2023.md` + `v2024.md`,
>    `mat_khau_v1.md` + `v2.md`. Cả 2 phiên bản đều được retrieve, LLM trộn số liệu của cả hai
>    (vd trả lời "12 ngày" theo v2023 trong khi chính sách hiện hành v2024 là 15 ngày) → faithfulness thấp.
> 3. **Negation trap** ("thử việc *có* được nghỉ phép năm *không*"): LLM trả lời khẳng định chung
>    thay vì bắt được ngoại lệ dành cho nhân viên thử việc.
> 4. `factual` chủ yếu bị trừ ở `answer_relevancy` (13 câu) chứ không phải faithfulness — câu trả lời
>    đúng nhưng lan man / kèm thông tin thừa, kéo answer_relevancy xuống dù nội dung chính xác.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness (multi_hop) | LLM tính toán sai / ghép nhầm điều kiện khi suy luận đa tài liệu | Prompt yêu cầu trình bày từng bước tính toán và trích dẫn câu nguồn; hạ `temperature=0`; thêm bước "self-check: mọi con số trong câu trả lời phải xuất hiện trong context" |
| faithfulness (adversarial) | Retrieve cả v2023 và v2024 → LLM trộn phiên bản | Thêm metadata `version`/`effective_date` cho chunk; filter chỉ giữ phiên bản mới nhất, hoặc nêu rõ trong prompt "nếu có nhiều phiên bản, chỉ dùng phiên bản hiện hành mới nhất" |
| answer_relevancy (factual) | Câu trả lời dài dòng, thừa thông tin | Prompt: "trả lời ngắn gọn, đúng trọng tâm câu hỏi, không thêm chính sách không được hỏi" |
| context_recall (adversarial) | Một số chunk chứa ngoại lệ/negation không lọt top-3 | Tăng `RERANK_TOP_K` lên 5 cho câu adversarial, hoặc thêm query expansion |

---

## 6. Nhận xét về Adversarial Distribution

> `avg_score`: adversarial **0.677** < multi_hop 0.728 < factual 0.900 — adversarial đúng là distribution
> khó nhất như thiết kế. 5/10 câu adversarial lọt bottom-10 (rank 2, 4, 5, 6 và 1 câu nữa).
> Pipeline **bị nhầm rõ rệt bởi version conflict**: rank 4 ("bao lâu đổi mật khẩu") và rank 5
> ("nghỉ bao nhiêu ngày phép") đều là câu hỏi thẳng nhưng điểm thấp vì corpus có 2 phiên bản
> và pipeline không có cơ chế ưu tiên phiên bản mới. Đây là bằng chứng cho thấy cần thêm
> **version-aware metadata filtering** trước khi đưa context vào LLM — retrieval thuần ngữ nghĩa
> không đủ khi hai tài liệu gần như giống hệt nhau về nội dung.
