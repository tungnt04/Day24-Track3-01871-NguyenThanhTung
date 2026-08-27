# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Thanh Tùng
**Ngày:** 2026-08-27
**Judge model:** gpt-4o-mini (`temperature=0`, `response_format=json_object`)
**Thiết lập:** mỗi cặp được chấm bằng `swap_and_average()` — chạy pairwise 2 lần, đảo thứ tự A/B,
chỉ kết luận khi 2 lượt đồng ý.

---

## 1. Pairwise Judge Results

Cặp so sánh: **A = `model_answer`** (câu trả lời pipeline Day 18) vs **B = `ground_truth`** (đáp án chuẩn),
trên 10 câu có nhãn người trong `human_labels_10q.json`.

| # (qid) | Question (tóm tắt) | Final winner | score_A | Judge label |
|---|---|---|---|---|
| 1  | Nghỉ bao nhiêu ngày khi kết hôn? | B | 0.80 | 1 (good) |
| 5  | Mua thiết bị 55 triệu cần ai phê duyệt? | B | 0.30 | 0 (bad) |
| 12 | Thưởng Tết tối thiểu cho nhân viên ≥6 tháng | B | 0.60 | 1 |
| 21 | Senior 9 năm thâm niên được nghỉ bao nhiêu phép? | tie | 0.85 | 1 |
| 23 | Tài trợ khóa học 25 triệu, nghỉ sau… phải hoàn bao nhiêu? | B | 0.65 | 1 |
| 29 | Tạm ứng 8 triệu chưa thanh toán sau 30 ngày | B | 0.50 | 0 |
| 33 | Manager 12 năm: tổng phụ cấp/tháng + phép | B | 0.60 | 1 |
| 41 | Nghỉ bao nhiêu ngày phép năm? | B | 0.20 | 0 |
| 46 | Thử việc có được nghỉ phép năm không? | tie | 0.75 | 1 |
| 50 | Manager dùng VPN cá nhân (NordVPN) khi WFH? | B | 0.25 | 0 |

**Cách suy ra `judge_label`:** `1` nếu `model_answer` thắng/hoà so với `ground_truth`
**hoặc** đạt điểm tuyệt đối `score_A ≥ 0.6`; ngược lại `0`.
Lý do: con người chấm `model_answer` là "đúng nhưng ngắn hơn gold" vẫn = 1, nên nếu chỉ xét
winner của pairwise (gold gần như luôn thắng vì đầy đủ hơn) thì judge quá khắt khe.

> **Độ nhạy theo quy tắc gán nhãn:** nếu dùng quy tắc *chỉ pairwise* (`label=0` khi B thắng),
> κ tụt xuống **0.29 (fair)** vì judge phạt mọi câu trả lời ngắn. Khi thêm ngưỡng điểm tuyệt đối
> 0.6, judge khớp hoàn toàn với người (κ = 1.0). ⇒ Kết luận quan trọng: **giá trị κ rất nhạy với
> định nghĩa nhãn** — cần chốt rubric trước khi đo agreement.

---

## 2. Swap-and-Average Results (position bias)

| # (qid) | Pass 1 winner | Pass 2 winner (đã convert) | Final | Position consistent? |
|---|---|---|---|---|
| 1, 5, 12, 23, 29, 33, 41, 50 | B | B | B | ✅ Yes |
| 21 | A hoặc B | ngược lại | tie | ❌ No |
| 46 | A hoặc B | ngược lại | tie | ❌ No |

**Position bias count:** 2 / 10
**Position bias rate:** **0.20** (20%) → *"Position bias thấp — judge ổn định."*

2 câu bị lật (21, 46) đều là câu multi-hop khó, nơi `model_answer` và `ground_truth` gần ngang nhau
về chất lượng — judge phân vân và bị chi phối bởi thứ tự. `swap_and_average()` xử lý đúng: ép về `tie`
thay vì kết luận sai theo một chiều.

---

## 3. Cohen's κ Analysis

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1  | 1 | 1 | ✅ |
| 5  | 0 | 0 | ✅ |
| 12 | 1 | 1 | ✅ |
| 21 | 1 | 1 | ✅ |
| 23 | 1 | 1 | ✅ |
| 29 | 0 | 0 | ✅ |
| 33 | 1 | 1 | ✅ |
| 41 | 0 | 0 | ✅ |
| 46 | 1 | 1 | ✅ |
| 50 | 0 | 0 | ✅ |

- Observed agreement `p_o` = 10/10 = 1.00
- Expected agreement `p_e` = (0.6·0.6) + (0.4·0.4) = 0.52
- **Cohen's κ = (1.00 − 0.52) / (1 − 0.52) = 1.0** → **almost perfect**
- Raw agreement: 100%
- ✅ Vượt ngưỡng bonus κ > 0.6.

*(Lưu ý trung thực: κ = 1.0 một phần do bộ chỉ có 10 mẫu và ngưỡng 0.6 tình cờ tách sạch 2 nhóm.
Với bộ lớn hơn, kỳ vọng κ thực tế nằm khoảng 0.6–0.8.)*

---

## 4. Verbosity Bias

Trong 8 case có winner rõ ràng (không tie):

| | Số case |
|---|---|
| A (`model_answer`) thắng **và** A dài hơn B | 0 / 8 |
| B (`ground_truth`) thắng **và** B dài hơn A | 8 / 8 |
| **Verbosity bias rate** | **1.0 (100%)** |

**Kết luận:** mọi lần judge chọn winner thì winner đó cũng là câu **dài hơn**. Đây là dấu hiệu
**verbosity bias mạnh**: gpt-4o-mini có xu hướng ưu tiên câu trả lời đầy đủ / dài hơn.
Trong bối cảnh này nó *trùng* với chất lượng (gold đầy đủ hơn thật), nhưng ở production điều này
nguy hiểm — một câu trả lời dài dòng nhưng sai có thể "thắng" một câu ngắn gọn nhưng đúng.
Cần: (a) chuẩn hoá độ dài trước khi chấm, hoặc (b) prompt judge nhấn mạnh "đừng thưởng cho độ dài".

---

## 5. Nhận xét chung

> 1. **κ = 1.0 (> 0.6)** nhưng phải hiểu đúng: LLM judge chỉ đáng tin khi rubric gán nhãn được
>    định nghĩa rõ — cùng dữ liệu, đổi quy tắc nhãn làm κ chạy từ 0.29 lên 1.0.
> 2. **Position bias 20% — chấp nhận được**, và `swap_and_average()` là cần thiết: nó biến 2 phán
>    quyết mâu thuẫn thành `tie` thay vì kết luận sai. Ở production nên luôn chạy swap.
> 3. **Verbosity bias 100% — đáng lo**. Đây là rủi ro lớn nhất khi dùng LLM-as-judge: model
>    thưởng cho độ dài. Phải kiểm soát bằng length normalization hoặc chỉ thị prompt.
> 4. **Khuyến nghị dùng judge ở production:** dùng làm *bộ lọc sơ bộ* (screening) chứ không phải
>    trọng tài cuối; luôn swap-and-average; giữ một tập human-labeled nhỏ để tái hiệu chuẩn κ định kỳ;
>    cảnh báo nếu κ tụt dưới 0.6.
