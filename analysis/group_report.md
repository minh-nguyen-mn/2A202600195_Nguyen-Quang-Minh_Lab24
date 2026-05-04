# Group Report — Lab 18

**Nhóm:** Nguyễn Quang Minh (Dự án cá nhân)  
**Ngày:** 04/05/2026

---

## Thành viên & module

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|------------|------------|
| Minh Nguyen | M1–M5 (Full RAG Pipeline) | ✓ | 32/32 |

---

## Kết quả

| Metric | Baseline | Production | Δ |
|--------|----------|------------|----|
| Faithfulness | 0.61 | 0.79 | +0.18 |
| Answer Relevancy | 0.64 | 0.81 | +0.17 |
| Context Precision | 0.58 | 0.84 | +0.26 |
| Context Recall | 0.55 | 0.78 | +0.23 |

---

## Key Findings

1. **Cải thiện lớn nhất:**
   → Context Precision nhờ hybrid search + reranker loại bỏ nhiễu

2. **Khó khăn lớn nhất:**
   → Dữ liệu pháp lý cần chunking theo cấu trúc (định nghĩa + danh sách)

3. **Phát hiện bất ngờ:**
   → Retrieval quan trọng hơn LLM trong bài toán QA pháp lý

---

## Ghi chú thuyết trình

1. So sánh RAGAS baseline vs production
2. Module hiệu quả nhất:
   → Hybrid retrieval + reranker
3. Case study:
   → lỗi 72 giờ do retrieval thiếu keyword “vi phạm”
4. Nếu có thêm thời gian:
   → query rewriting + chunking pháp lý + boosting số liệu