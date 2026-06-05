# Data Expansion Plan

ChronoRAG hiện có 45 tài liệu local cho 3 topic MVP: RAG, AI Agent và Knowledge Distillation. Dataset gán nhãn/training hiện vẫn dựa trên batch 30 tài liệu ban đầu, còn 15 tài liệu mới dùng để mở rộng corpus retrieval và kiểm thử pipeline. Con số này đủ để demo ingestion, event detection, timeline và local RAG, nhưng chưa đủ để chatbot trả lời rộng như một trợ lý nghiên cứu lớn.

## Mục tiêu mở rộng

| Giai đoạn | Corpus mục tiêu | Mục đích |
| --- | ---: | --- |
| MVP hiện tại | 45 local docs | Demo end-to-end, train baseline, UI chạy ổn |
| Stage 1 | 100 docs | RAG trả lời ổn hơn, timeline ít rỗng hơn |
| Stage 2 | 200 docs | Đủ tốt cho demo nhóm và báo cáo đánh giá retrieval |
| Stage 3 | 500 docs | Corpus lớn hơn cho CV/project extension, cần batch processing nghiêm túc |

## Nguyên tắc

- Không commit raw PDF vào Git.
- Không thêm 500 tài liệu vào `metadata.csv` một lần.
- Mỗi batch chỉ nên intake 30-50 PDF, sau đó chạy parse, schema validation và kiểm tra word count.
- Candidate CSV chỉ là danh sách ứng viên, chưa phải nguồn đã approved.
- Chỉ đưa vào `metadata.csv` khi tài liệu đã tải được, parse được, không lỗi encoding và có chất lượng đủ tốt.

## Quy trình đề xuất

1. Thu thập candidate paper:

```bash
python scripts/13_collect_arxiv_candidates.py --max-per-topic 170
```

Nếu arXiv bị rate-limit, dùng OpenAlex:

```bash
python scripts/15_collect_openalex_candidates.py --max-per-topic 170
```

Script OpenAlex đã tạo manifest cân bằng:

```text
data/raw/candidate_sources_expansion_openalex.csv
```

2. Review candidate:

- Giữ paper đúng topic.
- Ưu tiên survey, benchmark, method paper nổi bật.
- Loại paper quá hẹp, trùng ý, hoặc không có timeline signal.

3. Intake batch nhỏ:

- Chọn 30-50 doc_id từ candidate CSV.
- Tải PDF vào `data/raw/{topic}/`.
- Thêm dòng approved vào `data/raw/metadata.csv`.
- Chạy:

```bash
python workflows/offline_pipeline.py
python scripts/10_validate_processed_outputs.py
```

4. Sau mỗi batch:

- Kiểm tra số docs/chunks/sentences.
- Kiểm tra word count, encoding, duplicate source.
- Test nhanh Chat RAG và timeline.

## Vì sao chưa tải thẳng 500 PDF?

500 PDF có thể làm pipeline chậm, tăng noise, và làm chất lượng timeline giảm nếu không review. Với ChronoRAG, chất lượng nguồn quan trọng hơn số lượng thô. Cách tốt nhất là mở rộng theo batch có kiểm định.

## Nguồn thu thập

Ưu tiên ban đầu dùng arXiv API vì có metadata ổn định, `published_date`, `authors`, abstract URL và PDF URL. arXiv API hỗ trợ `search_query`, `start`, `max_results`, `sortBy`, `sortOrder`; khi gọi nhiều request liên tiếp nên sleep khoảng 3 giây giữa các request.

OpenAlex là nguồn dự phòng tốt khi arXiv hoặc Semantic Scholar bị rate-limit. OpenAlex hỗ trợ search/filter trên works, có `is_oa`, `primary_location.pdf_url`, `open_access.oa_url`, citation count và DOI. Candidate từ OpenAlex vẫn cần review thủ công trước khi chuyển thành approved corpus.
