# ChronoRAG

ML-enhanced Timeline-Aware Research Assistant for AI/ML/NLP topics.

ChronoRAG is not a simple PDF chatbot. The project combines document ingestion, RAG-style retrieval, ML/DL event sentence detection, event type classification, date extraction, event deduplication, timeline building, and citation-backed research Q&A.

## Current Status

The repo is currently at the end of **Sprint 3C: Labeled Dataset Ready**.

Completed:

| Area | Status |
| --- | --- |
| Repo skeleton | Done |
| Streamlit mock UI | Done |
| Local keyword/template Q&A | Done |
| Corpus acquisition | Done for 3-topic MVP |
| PDF/MD/HTML/TXT parsing | Done |
| Text cleaning, chunking, sentence splitting | Done |
| Processed data validation | Done |
| Labeling export | Done |
| Pilot labeling | Done |
| Full labeled dataset | Done |

Current verified dataset:

| Artifact | Value |
| --- | --- |
| Topics | `rag`, `ai_agent`, `knowledge_distillation` |
| Corpus size | 30 documents |
| Processed chunks | 2,599 |
| Processed sentences | 23,523 |
| Labeled rows | 1,800 |
| Event rows | 298 |
| Non-event rows | 1,502 |
| Validation | 0 errors |
| Unit tests | 106/106 passing |

Final label distribution:

| Label | Count |
| --- | ---: |
| `none` | 1502 |
| `trend_application` | 132 |
| `method_proposed` | 96 |
| `benchmark` | 57 |
| `release` | 13 |

Next sprint:

**Sprint 4: ML Baseline Classifier**

The next work is training real local/Kaggle models from `data/labeled/labeled_sentences.csv`.

## What Works Right Now

You can currently:

- Run the offline preprocessing pipeline if the raw local corpus is present.
- Validate processed JSONL outputs.
- Validate labeled CSV data.
- Run the Streamlit app with local mock-RAG style Q&A.
- Use the labeled dataset for ML training.

You cannot yet:

- Use a trained event classifier in the app.
- Use FAISS/Chroma vector retrieval as the main retrieval engine.
- Generate production timelines from ML predictions.
- Evaluate trained model metrics.

Those are Sprint 4+ tasks.

## Repository Structure

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

Important note: raw PDFs are ignored by Git with `data/raw/**/*.pdf`. Share large/raw PDF files through Drive/Kaggle, not normal Git commits.

## Setup

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` if needed:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env` or API keys.

## Common Commands

Run offline pipeline:

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

Validate final labeled dataset:

```powershell
python scripts/11_validate_labeled_data.py --input data/labeled/labeled_sentences.csv --mode labeled
```

Run regression tests:

```powershell
python -m unittest tests/test_sprint0.py tests/test_sprint15_local_qa.py tests/test_sprint2_ingestion.py tests/test_sprint2_cleaning.py tests/test_sprint2_validation.py tests/test_label_validation.py
```

Run Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

## Data Files

Important labeled files:

| File | Purpose |
| --- | --- |
| `data/labeled/labeling_candidates.csv` | 1,800 unlabeled candidate rows exported from processed sentences |
| `data/labeled/labeling_candidates_sample.csv` | 50-row pilot labeling sample |
| `data/labeled/draft_label_suggestions.csv` | Rule-based draft suggestions for full labeling |
| `data/labeled/review_queue.csv` | Rows selected for manual review |
| `data/labeled/labeled_sentences.csv` | Final labeled dataset for Sprint 4 training |

Final label columns:

| Column | Meaning |
| --- | --- |
| `is_event` | `1` if the sentence is a timeline/event sentence, else `0` |
| `event_type` | `method_proposed`, `release`, `benchmark`, `trend_application`, or `none` |
| `annotator` | Human reviewer name or ID |
| `label_method` | `human` for reviewed labels |
| `notes` | Optional rationale for ambiguous rows |

## Sprint 4 Plan

Sprint 4 trains the first real ML models.

### Task A: ML Baseline

Owner: team member 1

Files to own:

```text
scripts/04_train_ml_classifier.py
src/models/ml_baseline.py
data/eval/ml_baseline_metrics.json
saved_models/ml_*.pkl
reports/figures/ml_*.png
```

Required models:

- Binary event classifier: `is_event`
- Multiclass event type classifier: `event_type`
- Baselines: TF-IDF + Logistic Regression, TF-IDF + Linear SVM

Required details:

- Split by `doc_id`, not random sentence split.
- Use `class_weight="balanced"`.
- Report Accuracy, Precision, Recall, F1.
- Save confusion matrix plots.
- Document that `release` is a low-sample class.

### Task B: DL Baseline

Owner: team member 2

Files to own:

```text
scripts/05_train_dl_classifier.py
src/models/bilstm_classifier.py
src/models/dataset.py
data/eval/dl_bilstm_metrics.json
saved_models/bilstm_*.pt
reports/figures/dl_*.png
```

Required model:

- PyTorch BiLSTM event sentence classifier.

Nice to have:

- Multiclass event type BiLSTM.
- Training curve plot.

### Task C: Integration and Review

Owner: Tlam / reviewer

Responsibilities:

- Review metrics.
- Check class imbalance.
- Compare ML vs DL results.
- Decide which model is used in the demo pipeline.

## Kaggle Training Guide

Use Kaggle if local RAM is limited.

Minimum files to upload:

```text
data/labeled/labeled_sentences.csv
configs/
src/
scripts/
requirements.txt
```

Recommended Kaggle commands:

```bash
pip install -r requirements.txt
python scripts/04_train_ml_classifier.py --input data/labeled/labeled_sentences.csv --output-dir saved_models --eval-dir data/eval
```

For DL:

```bash
pip install -r requirements.txt
python scripts/05_train_dl_classifier.py --input data/labeled/labeled_sentences.csv --output-dir saved_models --eval-dir data/eval
```

After training, download outputs back into:

```text
saved_models/
data/eval/
reports/figures/
```

Commit small metrics/config/code files. Avoid committing very large model checkpoints.

## Team Workflow

Recommended loop:

1. Plan task and define file ownership.
2. Implement in the assigned files only.
3. Run the relevant command locally or on Kaggle.
4. Save outputs in the expected folders.
5. Run validation/tests.
6. Open review before moving to the next sprint.

Do not mix unrelated changes into one commit.

## Current Git Hygiene

Safe commit groups:

```powershell
git add README.md
git commit -m "update project readme"
```

Do not accidentally commit local scratch files:

```text
scratch/
scripts/clean_autogen.py
data/raw/ai_agent/agent_006_raw.html
docs/SPRINT_1_DATA_COLLECTION_PLAN.md
docs/SPRINT_2_INGESTION_PREPROCESSING_PLAN.md
```

## Reference Docs

- `docs/CHRONORAG_PLAN.md`: full project plan.
- `docs/LABELING_GUIDE.md`: annotation guideline for event labels.
- `data/labeled/README.md`: labeled dataset schema and labeling notes.

