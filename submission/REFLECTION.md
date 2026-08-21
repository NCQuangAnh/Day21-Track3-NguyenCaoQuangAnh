# Reflection - Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Model fine-tune thắng target rất áp đảo (0.765 lên 0.970) nhưng lại quên gần như hoàn toàn khả năng trả lời câu hỏi phổ thông, kể cả những câu cực đơn giản như "1 km bằng bao nhiêu mét". Nó không trả lời sai, nó trả nguyên khuôn JSON triage cho mọi câu hỏi, kể cả câu không liên quan gì tới ticket CSKH. Nếu chỉ nhìn con số target tôi đã tưởng đây là một lần fine-tune thành công, phải chạy riêng cổng hồi quy mới thấy cái giá thật.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở phần train hay tune tham số như tôi nghĩ ban đầu, mà ở phần hạ tầng: đồng bộ git giữa Colab và repo local, xử lý file adapter vượt giới hạn 100MB của GitHub, sửa lỗi pytest do cấu hình `pythonpath` thiếu. Phần chạy notebook thực ra khá mượt, cái tốn thời gian là quy trình nộp bài và giữ mọi thứ nhất quán giữa các lần chạy.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ train loss giảm là dấu hiệu đủ để biết fine-tune đang đi đúng hướng. Sau khi thấy `attn_only` có loss thấp hơn `correct` nhưng target score bằng nhau hệt, tôi hiểu train loss chỉ đo được việc model khớp dữ liệu train tới đâu, không đo được model có làm tốt việc thật hay không, càng không đo được nó có đánh đổi năng lực khác hay không.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng để xử lý git (merge, resolve conflict, đẩy commit lên GitHub và HuggingFace), debug lỗi pytest, và soạn thảo report dựa trên số liệu thật trong `results/`. Có một lần AI đọc nhầm dữ liệu cũ (do tôi chưa pull bản mới từ GitHub về), dẫn tới phân tích sai dựa trên `qualitative.json` phiên bản 8 mẫu thay vì 50 mẫu đầy đủ, phải pull lại rồi làm lại phần đó.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Không bắt đầu bằng việc train ngay. Bước đầu tiên là dựng bộ eval bốn nhóm (đặc biệt là nhóm regression để bắt catastrophic forgetting) và đóng băng baseline trước khi chạm vào training, đúng như thứ tự lab này bắt làm. Có baseline đóng băng thì mới có cơ sở khách quan để biết fine-tune có đáng làm hay không, thay vì tin vào cảm giác "nhìn output có vẻ ổn".
