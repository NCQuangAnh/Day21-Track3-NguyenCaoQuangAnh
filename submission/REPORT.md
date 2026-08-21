# Lab 21 — Evaluation Report

**Họ tên**: <điền>  **MSSV**: <điền>  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | ticket CSKH → JSON triage (mặc định: 250 ticket train / 50 target / 15 regression) |
| Train / val | 250 train (seed 42) / 50 eval_target + 15 eval_regression |
| `max_length` | 256 — p95 đo được là 98 token *(results/token_stats.json, n=250, mean=93.1, p95=98, max=101)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs → 30 optimizer steps (áp dụng cho cả `correct` và 3 run đối chứng NB4) |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: `verdict: "reasoning preserved — safe to train on traces"`)*. Chuỗi render sau `apply_chat_template` giữ nguyên cặp `<think>...</think>` trước câu trả lời, không cần xử lý gì thêm.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị mask (system + user turn) không xuất hiện trong đoạn trên — chỉ có `</think>` đóng khối suy luận và JSON câu trả lời được tính loss, đúng như `question_is_masked: true` ghi nhận.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3249.0 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1083.7 |
| (c) LoRA fine-tune | 0.970 | 0.522 | 1.000 | 1384.6 |

**(b) có thật sự mạnh hơn (a) không?** Có — target nhảy từ 0.000 lên 0.765, format từ 0.000 lên 1.000, latency giảm gần 3x (3249ms → 1084ms, vì (a) hay sinh dài dòng/lạc đề còn (b) ép đúng khuôn JSON). Không sửa `OPTIMIZED_PROMPT` so với bản mặc định của lab — dùng nguyên bản, `optimized_prompt_sha` khớp với `results/baselines_frozen.json`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | **0.97** | 925.0 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5377 | **0.97** | 816.8 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.00** | 949.6 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 1016.8 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct` (32,456,704 vs 32,464,896, lệch 0.025% — trong ngưỡng <5%). Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

`attn_only` HOÀ tuyệt đối với `correct` trên target (0.97 = 0.97), dù chỉ gắn vào q,v thay vì toàn bộ lớp linear. Xếp theo train loss lại cho thứ tự ngược: `attn_only` có loss thấp hơn (0.5377 < 0.6263) — nếu chỉ nhìn loss sẽ kết luận `attn_only` "học tốt hơn", nhưng target score cho thấy hai run thực chất ngang nhau về khả năng thật. Điều này chỉ ra: ở cùng một ngân sách tham số, khi rank được nâng đủ cao để bù cho vị trí hẹp (r=283 so với r=16), rank trở thành đòn bẩy chính, không phải vị trí gắn adapter — miễn ngân sách tham số công bằng. Loss thấp hơn ở `attn_only` không phản ánh khả năng tổng quát tốt hơn, mà chỉ là artefact của việc tối ưu trên ít module hơn với magnitude gradient khác — bằng chứng trực tiếp cho việc *không được xếp hạng bằng train loss*.

**4.2 — `wrong_lr` chỉ khác đúng một con số (LR 1e-5 thay vì 1e-4). Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` có train loss cao nhất trong 4 run (1.5704, gần gấp 2.5 lần `correct`) và target rớt về 0.00, format cũng 0.00 — model gần như không học được gì ở LR thang full-fine-tune. Nếu chỉ nhìn con số loss tuyệt đối mà không biết LR đứng sau, dễ kết luận sai rằng "cấu hình LoRA này (all-linear, r=16) không phù hợp với bài toán" — trong khi vấn đề thực sự chỉ là LR thấp hơn 10 lần so với mức cần thiết cho LoRA, khiến model gần như đứng yên ở step 30. Đây đúng là Lỗi #2 (§10.3): LR đúng thang cho full fine-tune lại quá nhỏ cho LoRA.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` tiết kiệm 12.01 - 7.09 = 4.92 GB VRAM (~41%) so với `correct`, nhưng trả giá bằng target thấp hơn (0.94 so với 0.97, giảm 3 điểm phần trăm) và latency cao hơn đáng kể (1747.4ms so với 1384.6ms, +26%, do overhead dequantize khi generate). Số đo này ủng hộ một phần khuyến nghị "không dùng QLoRA cho Qwen3.5" — chênh lệch target không lớn nhưng chi phí latency là rõ ràng và nhất quán; nếu VRAM không phải nút thắt (như T4 16GB đủ chạy `correct` ở 12GB), không có lý do đánh đổi latency và độ chính xác để lấy phần VRAM tiết kiệm đó.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.236` · `valid_trace_rate = 0.00`

Verdict FAILED vì `regression` tụt 0.236 (0.758 → 0.522), vượt xa ngưỡng chấp nhận 0.020, dù `target` tăng mạnh +0.205 (0.765 → 0.970) so với baseline (b). Đây là trường hợp kinh điển của catastrophic forgetting: 250 mẫu train chỉ chứa ticket triage, không có dữ liệu phổ thông xen lẫn, nên 2 epoch (30 step) đủ để model "quên" một phần năng lực trả lời câu hỏi kiến thức chung để đổi lấy việc tối ưu hoá quá mức cho định dạng JSON hẹp. `valid_trace_rate = 0.00` càng củng cố nghi ngờ: khối `<think>` gần như không còn giữ được nội dung suy luận có ý nghĩa sau fine-tune, có thể vì response ngắn (chỉ JSON) khiến model học được cách bỏ qua suy luận thay vì giữ nó. Nói cách khác, bài toán ở đây không phải "LoRA không hoạt động" — nó hoạt động rất tốt trên đúng domain train — mà là thiết kế dữ liệu train thiếu đa dạng khiến cái giá phải trả là năng lực tổng quát. Theo deck §14.3, cách khắc phục chuẩn là trộn 1-5% dữ liệu phổ thông vào tập train để giữ lại năng lực nền trong lúc vẫn học được task chuyên biệt — đây là hướng thử nghiệm tiếp theo hợp lý nhất.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "...đặt chuột không dây... Cho tôi trả lạ[i]" (i=0) | doi_tra/cao/... | — | `{"intent":"doi_tra","urgency":"cao",...}` (score 1.0) | ✅ FT thắng |
| 2 | "...đặt ốp lưng điện thoại... Hoàn tiền. Sớm n[ha]" (i=1) | hoan_tien/... | — | `{"intent":"hoan_tien",...}` (score 1.0) | ✅ FT thắng |
| 3 | "1 km bằng bao nhiêu mét?" | — | "1 kilômét tương đương với 1000 mét." (score 1.00) | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` (score 0.00) | ❌ **FT thua** |
| 4 | "Một năm có bao nhiêu tháng?" | — | "Một năm bình thường có 12 tháng..." (score 1.00) | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` (score 0.00) | ❌ **FT thua** |
| 5 | "...đặt bình giữ nhiệt... Chưa thấy tiền." (i=3) | hoan_tien/... | — | `{"intent":"hoan_tien",...}` (score 0.75 — sai 1 field) | ⚠️ FT đúng phần lớn, sai 1 field |

Có mẫu chung nào ở các ca FT thua không? Có, và rất rõ — model không "trả lời sai" theo nghĩa thông thường, nó **quay lại y nguyên khuôn JSON triage 4 trường bất kể câu hỏi là gì**. Với "1 km bằng bao nhiêu mét?" và "Nước sôi ở bao nhiêu độ C?", model đều trả về `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, ...}` — không có nội dung trả lời thật, chỉ có cấu trúc triage. Đây là bằng chứng trực tiếp cho catastrophic forgetting ở mức hành vi: model không quên "kiến thức" (1km=1000m) mà quên luôn **cách sinh văn bản tự do** — 250 mẫu train toàn ticket→JSON đã khiến nó học rằng mọi input đều phải map ra 4 field đó, bất kể nội dung.

> Ghi chú: tập **target** (ticket triage) không có ca fine-tune thua rõ ràng nào — điểm thấp nhất là 0.75 (đúng 3/4 field), không có ca 0 điểm hay sai hoàn toàn. Bằng chứng "FT thua" thuyết phục nhất nằm ở tập **regression**, nơi verdict.json ghi nhận sụt -0.236 — đó là lý do 2 ví dụ THUA ở trên lấy từ regression set thay vì target set: chọn ca thắng cherry-pick trong target trong khi lờ đi chỗ FT thực sự thua ở regression sẽ làm sai lệch bức tranh tổng thể.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Không nên deploy bản fine-tune `correct` này ở dạng hiện tại. Trên đúng domain train (ticket CSKH → JSON triage), nó thắng áp đảo baseline đã prompt tối ưu (+0.205 target, format hoàn hảo, latency chấp nhận được) — nếu chỉ nhìn target, đây là một case fine-tune thành công rõ ràng. Nhưng cổng hồi quy bốn nhóm bắt được cái giá ẩn: model đánh đổi 0.236 điểm regression (khả năng trả lời câu hỏi phổ thông) để lấy độ chính xác trong domain hẹp. Đây chính xác là lý do lab thiết kế cổng hồi quy riêng biệt thay vì chỉ nhìn target score — nếu chỉ đo target, verdict sẽ là PASS và quyết định deploy sẽ sai. Đòn bẩy thật sự trong lab này không phải vị trí gắn adapter (attn_only và correct hoà nhau khi ngân sách tham số công bằng, mục 4.1) — mà là **chất lượng và độ đa dạng dữ liệu train**. 250 mẫu train thuần một domain, không có replay dữ liệu phổ thông, là nguyên nhân trực tiếp gây catastrophic forgetting, không phải cấu hình LoRA (rank, vị trí) hay ngay cả learning rate — 3 run đối chứng NB4 (attn_only, wrong_lr, qlora) đều chỉ so trong cùng domain train, không cái nào chạm vào vấn đề regression. Hướng đi đúng tiếp theo là trộn 1-5% dữ liệu phổ thông vào training set (deck §14.3) rồi đo lại cổng hồi quy, chứ không phải chỉnh rank hay vị trí adapter thêm nữa.

**Ba điều tôi học được**:
1. Train loss thấp không đồng nghĩa model tốt hơn — `attn_only` có loss thấp hơn `correct` nhưng target score bằng nhau, chứng minh trực tiếp rằng xếp hạng bằng loss (thay vì eval task thật) là sai lầm phương pháp, không chỉ lý thuyết suông.
2. Một fine-tune có thể thắng áp đảo trên chính benchmark nó được tối ưu cho, và vẫn là quyết định sai để deploy — nếu không đo riêng khả năng tổng quát bằng một tập độc lập (regression), sẽ không bao giờ phát hiện ra cái giá đã trả.
3. `EVAL_LIMIT` (smoke mode) không chỉ làm nhanh hơn — nó còn làm verdict kém tin cậy hẳn vì n quá nhỏ (n=8 ban đầu cho ra kết quả khác hẳn n=50 đầy đủ); phải luôn kiểm `smoke_mode` trong `results/` trước khi tin bất kỳ con số nào.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 5% dữ liệu phổ thông (câu hỏi chung, không liên quan ticket) vào 250 mẫu train, train lại `correct`, và so sánh trực tiếp `regression_delta` trước/sau — đây là bài kiểm chứng trực tiếp cho giả thuyết đưa ra ở mục 5.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/QuangAnh0112/lab21-vi-ticket-triage-lora
