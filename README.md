# ChronoRAG

ChronoRAG là project AI/NLP xây dựng một **timeline-aware research assistant** cho các chủ đề AI / ML / NLP. Repo không chỉ làm chatbot đọc PDF, mà kết hợp RAG, ML event detection, phân loại event type, trích xuất mốc thời gian, xây timeline và hỏi đáp có citation.

## Trạng thái hiện tại

Repo đang ở mốc **Sprint 6 + UI migration foundation**:

| Hạng mục | Trạng thái |
| --- | --- |
| Corpus MVP 3 topic | Xong |
| Parse PDF/MD/HTML/TXT | Xong |
| Cleaning, chunking, sentence splitting | Xong |
| Processed/labeled data validation | Xong |
| Labeled dataset 1,800 câu | Xong |
| ML baseline TF-IDF + LogReg/LinearSVM/SGD | Xong |
| Precompute event predictions | Xong |
| BM25/hybrid local retrieval | Xong |
| Timeline builder từ prediction thật | Xong |
| FastAPI backend + React/Vite dashboard | Xong bản demo |

Số liệu đã kiểm tra gần nhất:

| Artifact | Giá trị |
| --- | ---: |
| Topic MVP | `rag`, `ai_agent`, `knowledge_distillation` |
| Documents | 30 |
| Chunks | 2,599 |
| Sentences | 23,523 |
| Labeled rows | 1,800 |
| Predicted events | 1,487 |
| Timeline events | 90 |
| Backend/UI regression tests | 174 passed |

## Repo hiện làm được gì?

- Chạy offline pipeline để tạo `documents.jsonl`, `chunks.jsonl`, `sentences.jsonl`.
- Validate processed data và labeled dataset.
- Train ML baseline cho `is_event` và `event_type`.
- Precompute event predictions từ model đã train.
- Build BM25 retrieval index.
- Build timeline từ event predictions, date extraction và clustering.
- Chạy chatbot local có citation dựa trên corpus.
- Chạy demo bằng dashboard FastAPI + React.

Lưu ý: chatbot mặc định là local RAG/template answerer có abstain gate, chưa cần API key để chạy demo. Nếu cấu hình `LLM_PROVIDER=openai` hoặc `LLM_PROVIDER=openrouter` trong `.env`, hệ thống sẽ dùng LLM để viết câu trả lời mượt hơn từ context đã retrieve; nếu thiếu key hoặc lỗi API thì tự fallback về local answerer. ML/DL event detection và timeline builder vẫn là phần lõi của project.

## Quick Start

### 1. Cài Python dependencies

Yêu cầu Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Verify các gói critical đã cài đầy đủ cho backend, ML và retrieval
python -c "import fastapi, pydantic, sklearn, rank_bm25; print('env ok')"
```

Nếu cần API key cho phần LLM sau này:

```powershell
Copy-Item .env.example .env
```

Sau đó chỉnh một trong hai cấu hình:

```powershell
# OpenAI-compatible API
LLM_PROVIDER=openai
OPENAI_API_KEY=...

# Hoặc OpenRouter
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
```

Không commit `.env` hoặc API key. Nếu để `LLM_PROVIDER=mock`, chatbot dùng local RAG/template answerer.

Nếu dùng LM Studio local:

```powershell
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=qwen/qwen3-4b
```

LM Studio cần bật server ở cổng `1234` và load một chat/instruct model. Key `lm-studio` chỉ là placeholder cho local server.

LLM (khi bật) chỉ chạy **sau** khi retrieval cục bộ tìm được context và **không** đạt abstain — tức nó chỉ "diễn đạt lại" câu trả lời từ chunk thật, không thay thế guard. Câu hỏi ngoài phạm vi corpus vẫn trả abstain dù có key. Mục tiêu: citation luôn bám vào nguồn thật, tránh hallucination.

### 2. Chuẩn bị artifacts nếu máy chưa có

```powershell
python workflows/offline_pipeline.py
python scripts/10_validate_processed_outputs.py
python scripts/11_validate_labeled_data.py --input data/labeled/labeled_sentences.csv --mode labeled
python scripts/04_train_ml_classifier.py
python scripts/07_precompute_predictions.py
python scripts/06_build_vector_index.py --skip-faiss
python scripts/08_build_timeline.py
```

### 3. Chạy dashboard mới FastAPI + React

Terminal 1:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev -- --port 5173
```

Mở:

- Frontend: `http://127.0.0.1:5173`
- Backend health: `http://127.0.0.1:8000/api/health`

## Các lệnh kiểm tra

```powershell
python -m compileall backend src scripts workflows tests
python -m unittest discover -s tests
cd frontend
npm run build
```

Kỳ vọng hiện tại: toàn bộ test Python pass và frontend build không lỗi TypeScript.

## Cấu trúc repo

```text
chrono-rag-assistant/
|-- backend/                # FastAPI backend cho React UI
|-- configs/                # Config pipeline/labeling
|-- data/
|   |-- raw/                # Raw corpus metadata/text; PDF lớn bị gitignore
|   |-- processed/          # documents/chunks/sentences/predictions/timeline
|   |-- labeled/            # labeling candidates + labeled_sentences.csv
|   |-- vector_db/          # BM25/FAISS artifacts
|   `-- eval/               # metrics, validation reports
|-- docs/                   # plan, labeling guide, architecture notes
|-- frontend/               # React/Vite dashboard
|-- reports/                # figures/report outputs
|-- saved_models/           # trained sklearn/PyTorch models
|-- scripts/                # runnable pipeline scripts
|-- src/                    # core modules
|-- tests/                  # regression tests
|-- workflows/              # offline/online orchestration
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Module chính

| Module | Vai trò |
| --- | --- |
| `src/ingest` | Load metadata và parse raw documents |
| `src/preprocessing` | Clean text, chunking, sentence splitting, labeling export/validation |
| `src/models` | ML baseline và inference wrapper |
| `src/indexing` | BM25 / vector index |
| `src/retrieval` | Simple/hybrid retrieval |
| `src/timeline` | Date extraction, clustering, timeline builder |
| `src/generation` | Template answerer có citation |
| `backend` | REST API cho dashboard mới |
| `frontend` | React UI theo style research dashboard |

## Dataset

Corpus MVP tập trung vào 3 topic:

- `rag`
- `ai_agent`
- `knowledge_distillation`

Các file quan trọng:

| File | Vai trò |
| --- | --- |
| `data/raw/metadata.csv` | Metadata nguồn tài liệu |
| `data/processed/documents.jsonl` | Document text sau parse/clean |
| `data/processed/chunks.jsonl` | Chunks cho retrieval |
| `data/processed/sentences.jsonl` | Sentences cho event detection |
| `data/labeled/labeled_sentences.csv` | Dataset nhãn cuối để train |
| `data/processed/event_predictions.jsonl` | Prediction từ ML baseline |
| `data/processed/timeline.json` | Timeline output |

Raw PDF lớn được ignore bằng `.gitignore`; chia sẻ qua Drive/Kaggle thay vì commit lên Git.

## Label schema

| Cột | Ý nghĩa |
| --- | --- |
| `is_event` | `1` nếu là event sentence, `0` nếu không |
| `event_type` | `method_proposed`, `release`, `benchmark`, `trend_application`, hoặc `none` |
| `annotator` | Người review/gán nhãn |
| `label_method` | `human` với nhãn đã được review |
| `notes` | Ghi chú cho case mơ hồ |

Class `release` hiện ít mẫu, nên khi train cần dùng class balancing và giải thích limitation này trong report.

## Train trên Kaggle

Nên dùng Kaggle nếu máy local yếu hoặc cần train DL.

Upload tối thiểu:

```text
configs/
data/labeled/labeled_sentences.csv
scripts/
src/
requirements.txt
```

Train ML baseline:

```bash
pip install -r requirements.txt
python scripts/04_train_ml_classifier.py
```

Sau khi train, tải output về:

```text
saved_models/
data/eval/
reports/figures/
```

Chỉ commit code/config/metrics nhỏ. Checkpoint lớn để Drive hoặc Kaggle output.

## Quy ước nhóm

1. Pull code mới nhất trước khi làm.
2. Làm thay đổi nhỏ, đúng phạm vi.
3. Chạy test/validation liên quan trước khi báo xong.
4. Không commit `.env`, raw PDF lớn, node_modules, model checkpoint lớn.
5. Nếu thay đổi pipeline, chạy lại artifacts tương ứng và ghi rõ lệnh đã chạy.
6. Nếu đổi UI, chạy cả `npm run build` và ít nhất một smoke test API.

## Security note

Các file `.pkl` dùng `pickle/joblib`. Không load model `.pkl` từ nguồn không tin cậy. Khi share model giữa thành viên, nên kèm checksum và xác nhận nguồn.

## Tài liệu tham khảo

- `docs/CHRONORAG_PLAN.md`: plan tổng thể.
- `docs/LABELING_GUIDE.md`: guideline gán nhãn event sentence.
- `data/labeled/README.md`: schema dataset labeled.
