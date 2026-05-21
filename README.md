# ChronoRAG

ChronoRAG là project AI/NLP xây dựng **research assistant có nhận thức theo dòng thời gian** cho các chủ đề AI / ML / NLP.

Project không chỉ là chatbot đọc PDF. Mục tiêu của ChronoRAG là kết hợp:

- thu thập và xử lý tài liệu;
- truy xuất thông tin kiểu RAG;
- phát hiện câu sự kiện bằng ML/DL;
- phân loại loại sự kiện;
- trích xuất mốc thời gian;
- gom cụm và khử trùng lặp sự kiện;
- xây dựng timeline;
- hỏi đáp có trích dẫn nguồn.

## Trạng thái hiện tại

Repo hiện đang ở cuối **Sprint 3C: hoàn thiện tập dữ liệu gán nhãn**.

Các phần đã hoàn thành:

| Hạng mục | Trạng thái |
| --- | --- |
| Khung repo ban đầu | Đã xong |
| Streamlit mock UI | Đã xong |
| Chatbot local dạng keyword/template | Đã xong |
| Thu thập corpus cho MVP 3 topic | Đã xong |
| Parse PDF/MD/HTML/TXT | Đã xong |
| Làm sạch text, chunking, sentence splitting | Đã xong |
| Validate dữ liệu processed | Đã xong |
| Export dữ liệu để labeling | Đã xong |
| Pilot labeling 50 dòng | Đã xong |
| Full labeled dataset 1800 dòng | Đã xong |

Số liệu đã kiểm chứng:

| Artifact | Giá trị |
| --- | --- |
| Topic MVP | `rag`, `ai_agent`, `knowledge_distillation` |
| Số tài liệu corpus | 30 documents |
| Số chunks | 2,599 |
| Số sentences | 23,523 |
| Số dòng labeled | 1,800 |
| Event sentences | 298 |
| Non-event sentences | 1,502 |
| Validation labeled data | 0 errors |
| Unit tests | 106/106 passing |

Phân phối nhãn cuối:

| Label | Số lượng |
| --- | ---: |
| `none` | 1502 |
| `trend_application` | 132 |
| `method_proposed` | 96 |
| `benchmark` | 57 |
| `release` | 13 |

## Repo hiện làm được gì?

Hiện tại repo đã có thể:

- chạy pipeline offline để parse và xử lý corpus nếu raw files có sẵn ở local;
- tạo `documents.jsonl`, `chunks.jsonl`, `sentences.jsonl`;
- validate dữ liệu processed;
- validate dữ liệu labeled;
- chạy Streamlit app ở mức demo/mock;
- dùng `data/labeled/labeled_sentences.csv` để train model ở Sprint 4.

Repo **chưa có**:

- model event classifier đã train thật;
- DL/BiLSTM classifier;
- FAISS/Chroma vector DB hoàn chỉnh;
- timeline builder dùng prediction thật;
- RAG chatbot hoàn chỉnh bằng vector retrieval + LLM;
- báo cáo metrics model thật.

## Mốc tiếp theo

Mốc tiếp theo là **Sprint 4: train model**.

Sprint 4 sẽ biến dataset đã gán nhãn thành model thật:

- Binary classifier: dự đoán `is_event`.
- Event type classifier: dự đoán `method_proposed`, `release`, `benchmark`, `trend_application`.
- ML baseline: TF-IDF + Logistic Regression / Linear SVM.
- Optional DL baseline: PyTorch BiLSTM.
- Output cần có: saved models, metrics JSON, confusion matrix, phần nhận xét kết quả.

Lưu ý quan trọng cho Sprint 4:

- Split train/val/test theo `doc_id`, không split random từng sentence, để giảm data leakage.
- Dùng class balancing vì class `release` chỉ có 13 mẫu.
- Nếu máy local yếu, nên train trên Kaggle.

## Cấu trúc repo

```text
chrono-rag-assistant/
|-- app/
|   `-- streamlit_app.py
|-- configs/
|   |-- config.yaml
|   `-- labeling_config.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- labeled/
|   |-- vector_db/
|   `-- eval/
|-- docs/
|   |-- CHRONORAG_PLAN.md
|   `-- LABELING_GUIDE.md
|-- reports/
|-- saved_models/
|-- scripts/
|   |-- 01_ingest_documents.py
|   |-- 02_preprocess_documents.py
|   |-- 03_export_labeling_data.py
|   |-- 04_train_ml_classifier.py
|   |-- 05_train_dl_classifier.py
|   |-- 06_build_vector_index.py
|   |-- 07_precompute_predictions.py
|   |-- 08_build_timeline.py
|   |-- 09_evaluate_system.py
|   |-- 10_validate_processed_outputs.py
|   `-- 11_validate_labeled_data.py
|-- src/
|   |-- ingest/
|   |-- preprocessing/
|   |-- indexing/
|   |-- retrieval/
|   |-- models/
|   |-- timeline/
|   |-- generation/
|   |-- evaluation/
|   `-- utils/
|-- tests/
|-- workflows/
|   `-- offline_pipeline.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

Lưu ý: raw PDF đang được ignore bằng `.gitignore` qua rule `data/raw/**/*.pdf`. File PDF lớn nên chia sẻ qua Drive/Kaggle, không commit trực tiếp lên Git.

## Cài đặt local

Yêu cầu Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu cần cấu hình API key, tạo file `.env` từ mẫu:

```powershell
Copy-Item .env.example .env
```

Không commit `.env` hoặc API key.

## Các lệnh thường dùng

Chạy pipeline offline:

```powershell
python workflows/offline_pipeline.py
```

Validate processed outputs:

```powershell
python scripts/10_validate_processed_outputs.py
```

Export labeling candidates:

```powershell
python scripts/03_export_labeling_data.py
```

Validate labeled dataset:

```powershell
python scripts/11_validate_labeled_data.py --input data/labeled/labeled_sentences.csv --mode labeled
```

Chạy regression tests:

```powershell
python -m unittest tests/test_sprint0.py tests/test_sprint15_local_qa.py tests/test_sprint2_ingestion.py tests/test_sprint2_cleaning.py tests/test_sprint2_validation.py tests/test_label_validation.py
```

Chạy Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

## Các file dữ liệu quan trọng

| File | Vai trò |
| --- | --- |
| `data/labeled/labeling_candidates.csv` | 1,800 câu candidate ban đầu để gán nhãn |
| `data/labeled/labeling_candidates_sample.csv` | 50 câu pilot labeling |
| `data/labeled/draft_label_suggestions.csv` | Gợi ý nhãn nháp bằng rule |
| `data/labeled/review_queue.csv` | Các dòng cần review kỹ |
| `data/labeled/labeled_sentences.csv` | Dataset nhãn cuối dùng cho Sprint 4 |

Các cột nhãn chính:

| Cột | Ý nghĩa |
| --- | --- |
| `is_event` | `1` nếu câu là event sentence, `0` nếu không |
| `event_type` | `method_proposed`, `release`, `benchmark`, `trend_application`, hoặc `none` |
| `annotator` | Người review/gán nhãn |
| `label_method` | `human` với nhãn đã được người review |
| `notes` | Ghi chú cho case mơ hồ nếu cần |

## Train trên Kaggle

Nên dùng Kaggle nếu máy local yếu hoặc RAM thấp.

Các file tối thiểu cần upload:

```text
data/labeled/labeled_sentences.csv
configs/
src/
scripts/
requirements.txt
```

Lệnh train ML baseline dự kiến:

```bash
pip install -r requirements.txt
python scripts/04_train_ml_classifier.py --input data/labeled/labeled_sentences.csv --output-dir saved_models --eval-dir data/eval
```

Lệnh train DL baseline dự kiến:

```bash
pip install -r requirements.txt
python scripts/05_train_dl_classifier.py --input data/labeled/labeled_sentences.csv --output-dir saved_models --eval-dir data/eval
```

Sau khi train, tải output về repo:

```text
saved_models/
data/eval/
reports/figures/
```

Chỉ commit code, config, metrics nhỏ. Model checkpoint quá lớn thì để Drive/Kaggle output.

## Workflow nhóm

Quy trình làm việc đề xuất:

1. Pull code mới nhất từ GitHub.
2. Đọc README và docs liên quan.
3. Làm từng thay đổi nhỏ, đúng phạm vi.
4. Chạy validation/test phù hợp trước khi báo xong.
5. Commit các file liên quan cùng một nhóm thay đổi.
6. Review trước khi chuyển sang sprint tiếp theo.

Phần chia task cụ thể sẽ được quản lý riêng, không ghi trực tiếp trong README.

## Lưu ý khi commit

Không commit các file local/scratch:

```text
scratch/
scripts/clean_autogen.py
data/raw/ai_agent/agent_006_raw.html
docs/SPRINT_1_DATA_COLLECTION_PLAN.md
docs/SPRINT_2_INGESTION_PREPROCESSING_PLAN.md
```

Không commit:

```text
.env
raw PDFs lớn
model checkpoints quá nặng
```

## Tài liệu tham khảo

- `docs/CHRONORAG_PLAN.md`: plan tổng thể của project.
- `docs/LABELING_GUIDE.md`: guideline gán nhãn event sentence.
- `data/labeled/README.md`: mô tả schema dataset labeled.

