# Individual Reflection — Lab 18

**Tên:** Nguyễn Quang Minh 
**Module phụ trách:** M1, M2, M3, M4, M5 (Full Production RAG System)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:**
  - M1: Advanced Chunking (semantic, hierarchical, structure-aware)
  - M2: Hybrid Search (BM25 + Dense + RRF fusion)
  - M3: Cross-encoder reranking
  - M4: RAGAS evaluation + failure analysis
  - M5: Enrichment pipeline (summarization, HyQA, metadata, contextual prepending)
  - Integration pipeline (`pipeline.py` + `main.py`)

- **Các hàm/class chính đã viết:**
  - `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`
  - `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()`
  - `CrossEncoderReranker.rerank()`
  - `evaluate_ragas()`, `failure_analysis()`
  - `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`
  - `enrich_chunks()`
  - `build_pipeline()`, `run_query()`

- **Số tests pass:** 32/32 (local validation)

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:**
  - Hybrid retrieval (BM25 + dense embeddings)
  - Reranking bằng cross-encoder
  - RAG evaluation bằng RAGAS (faithfulness, precision, recall)
  - Enrichment pipeline trước embedding (HyQA, contextual prepending)

- **Điều bất ngờ nhất:**
  - Retrieval quality quan trọng hơn LLM trong bài toán QA pháp lý
  - Chunking sai cấu trúc làm giảm performance mạnh hơn cả model yếu

- **Kết nối với bài giảng:**
  - Slide về RAG pipeline architecture (retrieval → rerank → generation)
  - Slide về semantic search vs keyword search
  - Slide về evaluation metrics (context precision vs recall tradeoff)

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  - Qdrant setup lỗi trên Windows (connection refused / empty index)
  - Embedding model bge-m3 rất nặng → thiếu disk space + download fail
  - Chunking làm vỡ cấu trúc pháp lý (definition + list)

- **Cách giải quyết:**
  - Fallback logic trong pipeline (skip rerank nếu fail)
  - Retry + cache model download
  - Debug từng stage (chunk → search → rerank → LLM)
  - Thêm hybrid retrieval để giảm phụ thuộc 1 phương pháp

- **Thời gian debug:**
  - ~3–4 giờ (chủ yếu ở Qdrant + embedding + indexing errors)

---

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  - Không dùng Qdrant local (chuyển sang FAISS hoặc in-memory vector store)
  - Thiết kế chunking theo “legal structure-aware” từ đầu
  - Thêm query rewriting (HyQA) ngay từ phase 1
  - Cache embeddings để tránh download lại model lớn

- **Module muốn thử tiếp:**
  - Fine-tuning reranker cho tiếng Việt
  - Query expansion bằng LLM reasoning layer
  - Graph-based retrieval (knowledge graph for legal rules)

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |