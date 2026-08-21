# Lab 21 - Evaluation Report

Họ tên: Nguyễn Cao Quang Anh
MSSV: 2A202601352
Ngày: 2026-08-21
Tier: T4, Base model: unsloth/Qwen3.5-4B, GPU thực tế: Colab Free T4 16GB

Mọi con số dưới đây khớp với file trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | ticket CSKH -> JSON triage (250 ticket train, 50 target, 15 regression) |
| Train / val | 250 train (seed 42), 50 eval_target + 15 eval_regression |
| `max_length` | 256, tính theo p95 đo được là 98 token (`results/token_stats.json`, n=250, mean=93.1, p95=98, max=101) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs, 30 optimizer step, dùng chung cho `correct` và cả 3 run đối chứng ở NB4 |

Template có giữ khối `<think>` không: có. `results/template_check.json` ghi verdict "reasoning preserved - safe to train on traces". Chuỗi render sau `apply_chat_template` giữ nguyên cặp `<think>...</think>` trước câu trả lời, không cần xử lý thêm gì.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi không nằm trong loss | true |

3-5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần system và user turn không xuất hiện ở đây, chỉ có `</think>` đóng khối suy luận rồi tới JSON câu trả lời được tính loss. Khớp với `question_is_masked: true` trong file.

---

## 3. Ba baseline (NB2, đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3249.0 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1083.7 |
| (c) LoRA fine-tune | 0.970 | 0.522 | 1.000 | 1384.6 |

(b) có mạnh hơn (a) thật không: có. Target nhảy từ 0.000 lên 0.765, format từ 0.000 lên 1.000, latency giảm gần 3 lần (3249ms xuống 1084ms) vì (a) hay sinh dài dòng lạc đề còn (b) ép đúng khuôn JSON. Không sửa `OPTIMIZED_PROMPT` so với bản mặc định của lab, `optimized_prompt_sha` khớp với `results/baselines_frozen.json`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | target (NB5 muc 4) | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6267 | 0.97 | 921.1 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.5377 | 0.97 | 816.8 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.00 | 949.6 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.94 | 1016.8 | 7.09 |

Xếp hạng theo cột target, không theo train loss.

**4.1 - attn_only có cùng số tham số huấn luyện với correct (32,456,704 so với 32,464,896, lệch 0.025%, trong ngưỡng dưới 5%). Trên tập target nó thắng, thua hay hoà? Thứ tự đó có giống train loss không? Nói gì về rank so với vị trí gắn adapter?**

attn_only hoà tuyệt đối với correct trên target, 0.97 bằng 0.97, dù chỉ gắn vào q,v thay vì toàn bộ lớp linear. Xếp theo train loss lại ra thứ tự ngược: attn_only có loss thấp hơn (0.5377 so với 0.6267). Nếu chỉ nhìn loss sẽ nghĩ attn_only học tốt hơn, nhưng target score cho thấy hai run ngang nhau về khả năng thật. Cùng một ngân sách tham số, khi rank được nâng đủ cao để bù cho vị trí hẹp (r=283 so với r=16), rank mới là đòn bẩy chính, không phải vị trí gắn adapter, miễn ngân sách tham số công bằng. Loss thấp hơn ở attn_only không phản ánh khả năng tổng quát tốt hơn, chỉ là kết quả của việc tối ưu trên ít module hơn với magnitude gradient khác. Đây là lý do không được xếp hạng bằng train loss.

**4.2 - wrong_lr chỉ khác đúng một con số (LR 1e-5 thay vì 1e-4). Đường loss khác thế nào? Nếu chỉ nhìn loss mà không biết LR sẽ kết luận sai điều gì?**

wrong_lr có train loss cao nhất trong 4 run (1.5704, gần gấp 2.5 lần correct), target rớt về 0.00, format cũng 0.00. Model gần như không học được gì ở LR thang full-fine-tune. Nếu chỉ nhìn con số loss mà không biết LR đứng sau, dễ kết luận nhầm rằng cấu hình LoRA này (all-linear, r=16) không phù hợp với bài toán, trong khi vấn đề thực sự chỉ là LR thấp hơn 10 lần so với mức cần cho LoRA, khiến model gần như đứng yên ở step 30. Đây là lỗi số 2 trong deck (LR đúng thang cho full fine-tune lại quá nhỏ cho LoRA).

**4.3 - qlora tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo có ủng hộ khuyến nghị không dùng QLoRA cho dòng model này không?**

qlora tiết kiệm 12.01 trừ 7.09 bằng 4.92 GB VRAM, khoảng 41% so với correct, nhưng target thấp hơn (0.94 so với 0.97, giảm 3 điểm phần trăm) và latency cao hơn rõ rệt (1747.4ms so với 1384.6ms, tăng 26%, do overhead dequantize lúc generate). Số đo này ủng hộ một phần khuyến nghị không dùng QLoRA cho Qwen3.5: chênh lệch target không lớn nhưng chi phí latency là rõ ràng và nhất quán. Nếu VRAM không phải nút thắt, như T4 16GB đã đủ chạy correct ở 12GB, thì không có lý do đánh đổi latency và độ chính xác để lấy phần VRAM tiết kiệm đó.

---

## 5. Phán quyết (NB5)

Kết quả cổng hồi quy: FAILED
target delta = +0.205, regression delta = -0.236, valid_trace_rate = 0.00

Verdict FAILED vì regression tụt 0.236 (0.758 xuống 0.522), vượt xa ngưỡng chấp nhận 0.020, dù target tăng mạnh +0.205 (0.765 lên 0.970) so với baseline (b). Đây là catastrophic forgetting: 250 mẫu train chỉ có ticket triage, không trộn dữ liệu phổ thông, nên 2 epoch (30 step) đủ để model đánh mất một phần năng lực trả lời câu hỏi kiến thức chung để đổi lấy việc tối ưu quá mức cho định dạng JSON hẹp. valid_trace_rate bằng 0.00 càng củng cố điều này: khối `<think>` gần như không còn giữ nội dung suy luận có ý nghĩa sau fine-tune, có thể vì response ngắn (chỉ JSON) khiến model học cách bỏ qua suy luận thay vì giữ nó. Vấn đề ở đây không phải LoRA không hoạt động, nó hoạt động rất tốt trên đúng domain train, mà là dữ liệu train thiếu đa dạng nên cái giá phải trả là năng lực tổng quát. Theo deck muc 14.3, cách khắc phục chuẩn là trộn 1-5% dữ liệu phổ thông vào tập train để giữ năng lực nền trong lúc vẫn học được task chuyên biệt. Đây là hướng thử tiếp theo hợp lý nhất.

---

## 6. Định tính, có cả ca thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | đặt chuột không dây, cho tôi trả lại (i=0) | doi_tra, cao | - | intent doi_tra, urgency cao, ... (score 1.0) | FT thắng |
| 2 | đặt ốp lưng điện thoại, hoàn tiền sớm (i=1) | hoan_tien | - | intent hoan_tien, ... (score 1.0) | FT thắng |
| 3 | 1 km bằng bao nhiêu mét? | - | 1 kilômét tương đương với 1000 mét (score 1.00) | intent hoi_thong_tin, urgency thap, product null, sentiment trung_tinh (score 0.00) | FT thua |
| 4 | Một năm có bao nhiêu tháng? | - | Một năm bình thường có 12 tháng... (score 1.00) | intent hoi_thong_tin, urgency thap, product null, sentiment trung_tinh (score 0.00) | FT thua |
| 5 | đặt bình giữ nhiệt, chưa thấy tiền (i=3) | hoan_tien | - | intent hoan_tien, ... (score 0.75, sai 1 field) | FT đúng phần lớn, sai 1 field |

Có mẫu chung ở các ca FT thua không: có, và khá rõ. Model không trả lời sai theo nghĩa thông thường, nó quay lại y nguyên khuôn JSON triage 4 trường bất kể câu hỏi là gì. Với "1 km bằng bao nhiêu mét?" và "Nước sôi ở bao nhiêu độ C?", model đều trả về dạng intent hoi_thong_tin, urgency thap, product null, không có nội dung trả lời thật, chỉ có cấu trúc triage. Đây là catastrophic forgetting ở mức hành vi: model không quên kiến thức (1km = 1000m), nó quên cách sinh văn bản tự do. 250 mẫu train toàn ticket sang JSON đã khiến nó học rằng mọi input đều phải map ra 4 field đó, bất kể nội dung.

Ghi chú: tập target (ticket triage) không có ca fine-tune thua rõ ràng, điểm thấp nhất là 0.75 (đúng 3/4 field), không có ca 0 điểm hay sai hoàn toàn. Bằng chứng FT thua thuyết phục nhất nằm ở tập regression, nơi verdict.json ghi nhận sụt -0.236. Đó là lý do 2 ví dụ thua ở trên lấy từ regression set thay vì target set. Chọn ca thắng cherry-pick trong target trong khi bỏ qua chỗ FT thực sự thua ở regression sẽ làm sai lệch bức tranh tổng thể.

---

## 7. Kết luận và điều tôi học được

Kết luận: không nên deploy bản fine-tune correct này ở dạng hiện tại. Trên đúng domain train (ticket CSKH sang JSON triage), nó thắng áp đảo baseline đã prompt tối ưu (+0.205 target, format hoàn hảo, latency chấp nhận được). Nếu chỉ nhìn target, đây là một case fine-tune thành công. Nhưng cổng hồi quy bốn nhóm bắt được cái giá ẩn: model đánh đổi 0.236 điểm regression, tức khả năng trả lời câu hỏi phổ thông, để lấy độ chính xác trong domain hẹp. Đây là lý do lab thiết kế cổng hồi quy riêng thay vì chỉ nhìn target score, vì nếu chỉ đo target, verdict sẽ là PASSED và quyết định deploy sẽ sai. Đòn bẩy thật sự trong lab này không phải vị trí gắn adapter, vì attn_only và correct hoà nhau khi ngân sách tham số công bằng (mục 4.1), mà là chất lượng và độ đa dạng dữ liệu train. 250 mẫu train thuần một domain, không trộn dữ liệu phổ thông, là nguyên nhân trực tiếp gây catastrophic forgetting, không phải cấu hình LoRA hay learning rate. Ba run đối chứng ở NB4 (attn_only, wrong_lr, qlora) đều chỉ so trong cùng domain train, không cái nào chạm vào vấn đề regression. Hướng đi đúng tiếp theo là trộn 1-5% dữ liệu phổ thông vào training set rồi đo lại cổng hồi quy, chứ không phải chỉnh rank hay vị trí adapter thêm nữa.

Ba điều tôi học được:

1. Train loss thấp không đồng nghĩa model tốt hơn. attn_only có loss thấp hơn correct nhưng target score bằng nhau, chứng minh rằng xếp hạng bằng loss thay vì eval task thật là sai lầm phương pháp, không chỉ là lý thuyết suông.
2. Một fine-tune có thể thắng áp đảo trên chính benchmark nó được tối ưu cho, và vẫn là quyết định sai để deploy. Nếu không đo riêng khả năng tổng quát bằng một tập độc lập, sẽ không bao giờ phát hiện ra cái giá đã trả.
3. EVAL_LIMIT (smoke mode) không chỉ chạy nhanh hơn, nó còn làm verdict kém tin cậy vì n quá nhỏ. Chạy với n=8 ban đầu cho kết quả khác hẳn n=50 đầy đủ, nên phải luôn kiểm smoke_mode trong results trước khi tin bất kỳ con số nào.

Nếu có thêm 2 giờ nữa, tôi sẽ thử: trộn 5% dữ liệu phổ thông (câu hỏi chung, không liên quan ticket) vào 250 mẫu train, train lại correct, và so sánh trực tiếp regression delta trước và sau. Đây là bài kiểm chứng trực tiếp cho giả thuyết đưa ra ở mục 5.

---

## Phụ lục - thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai MASK_MODE, kèm valid_trace_rate)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub, link: https://huggingface.co/QuangAnh0112/lab21-vi-ticket-triage-lora
