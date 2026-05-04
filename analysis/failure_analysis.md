# Failure Analysis — Lab 18

**Nhóm:** Nguyễn Quang Minh (Dự án cá nhân)  
**Thành viên:** Nguyễn Quang Minh → M1, M2, M3, M4, M5 (Toàn bộ pipeline RAG)

---

## Điểm RAGAS

| Metric | Baseline | Production | Δ |
|--------|----------|------------|----|
| Faithfulness | 0.61 | 0.79 | +0.18 |
| Answer Relevancy | 0.64 | 0.81 | +0.17 |
| Context Precision | 0.58 | 0.84 | +0.26 |
| Context Recall | 0.55 | 0.78 | +0.23 |

---

## Top 5 lỗi (dựa trên dataset Nghị định 13/2023/NĐ-CP)

### #1 — Thiếu đầy đủ định nghĩa dữ liệu cá nhân
- **Câu hỏi:** Dữ liệu cá nhân theo Nghị định 13/2023/NĐ-CP là gì?
- **Kỳ vọng:** Định nghĩa đầy đủ gồm chữ, số, hình ảnh, âm thanh…
- **Kết quả:** Thiếu một số dạng dữ liệu (đặc biệt âm thanh/hình ảnh)
- **Metric yếu nhất:** Context Recall
- **Cây lỗi:** Output sai → Context thiếu → Chunk tách sai định nghĩa
- **Nguyên nhân:** Chunking làm vỡ cấu trúc định nghĩa pháp lý
- **Khắc phục:** Chunk theo cấu trúc (không cắt định nghĩa)

---

### #2 — Mô tả “đồng ý” chưa đủ chi tiết
- **Câu hỏi:** Sự đồng ý của chủ thể dữ liệu phải thể hiện như thế nào?
- **Kỳ vọng:** rõ ràng, tự nguyện, cụ thể, nhiều hình thức
- **Kết quả:** thiếu “điện tử / hành động xác nhận”
- **Metric yếu nhất:** Faithfulness
- **Cây lỗi:** Output sai → LLM rút gọn context → prompt chưa chặt
- **Nguyên nhân:** prompt generation không ép bám context
- **Khắc phục:** ép “chỉ trả lời từ context, không suy diễn”

---

### #3 — Sai thời hạn 72 giờ
- **Câu hỏi:** Thời hạn thông báo vi phạm dữ liệu cá nhân là bao lâu?
- **Kỳ vọng:** 72 giờ
- **Kết quả:** trả lời sai hoặc mơ hồ
- **Metric yếu nhất:** Answer Relevancy
- **Cây lỗi:** Output sai → retrieval không bắt keyword “vi phạm”
- **Nguyên nhân:** BM25 + embedding chưa ưu tiên số liệu
- **Khắc phục:** HyQA + tăng trọng số số (72h, 60 ngày)

---

### #4 — Thiếu điều kiện chuyển dữ liệu ra nước ngoài
- **Câu hỏi:** Chuyển dữ liệu cá nhân ra nước ngoài cần điều kiện gì?
- **Kỳ vọng:** DPIA + gửi Bộ Công an + thời hạn 60 ngày
- **Kết quả:** chỉ trả lời một phần
- **Metric yếu nhất:** Context Precision
- **Cây lỗi:** Output sai → context nhiễu → reranker chưa đủ mạnh
- **Nguyên nhân:** ranking chưa lọc tốt chunk liên quan
- **Khắc phục:** cải thiện cross-encoder reranker

---

### #5 — Thiếu danh sách dữ liệu nhạy cảm
- **Câu hỏi:** Dữ liệu cá nhân nhạy cảm gồm những loại nào?
- **Kỳ vọng:** chính trị, tôn giáo, sinh trắc học, sức khỏe, vị trí…
- **Kết quả:** danh sách bị thiếu một phần
- **Metric yếu nhất:** Context Recall
- **Cây lỗi:** Output sai → chunk bị tách list → thiếu context
- **Nguyên nhân:** chunking làm vỡ danh sách liệt kê
- **Khắc phục:** giữ nguyên block danh sách khi chunk

---

## Case study

**Câu hỏi:** Thời hạn thông báo vi phạm dữ liệu cá nhân là bao lâu?

### Phân tích lỗi:
1. Output đúng? → ❌ sai/mơ hồ
2. Context đúng? → ⚠️ chỉ một phần
3. Query đúng chưa? → ⚠️ thiếu trọng tâm “vi phạm”
4. Lỗi nằm ở: retrieval + chunking

---

### Nếu có thêm 1 giờ:
- Chunking theo cấu trúc pháp lý (định nghĩa + danh sách)
- Query expansion cho luật + số liệu
- Tăng trọng số cho điều kiện thời gian (72h, 60 ngày)