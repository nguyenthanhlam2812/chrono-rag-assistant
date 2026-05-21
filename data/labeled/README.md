# ChronoRAG Labeled Dataset Directory

This directory contains candidate sentences extracted for event labeling, pilot sample files, and reports.

---

## 1. Directory Structure

- **`labeling_candidates.csv`**: The main dataset of 1800 sentences selected for annotation (UTF-8-sig format).
- **`labeling_candidates_sample.csv`**: A balanced pilot subset of 50 sentences for training/alignment.
- **`labeling_export_report.json`**: Sampling statistics detailing topic, doc, and score bucket allocations.
- **`README.md`**: This guide.

---

## 2. Dataset Schema

The candidate CSV files contain the following fields:

| Column Name | Description | Export State | Labeled State |
| :--- | :--- | :--- | :--- |
| `sentence_id` | Globally unique identifier (e.g., `agent_001_s42`) | Populated | Populated |
| `doc_id` | Document ID from metadata.csv (e.g., `agent_001`) | Populated | Populated |
| `chunk_id` | Chunk ID from chunk processing | Populated | Populated |
| `topic` | High-level topic (`rag`, `ai_agent`, `knowledge_distillation`) | Populated | Populated |
| `title` | Document title (parsed from metadata.csv) | Populated | Populated |
| `source_url` | Source URL of the origin document | Populated | Populated |
| `year` | Document publication year | Populated | Populated |
| `sentence` | The text snippet to evaluate | Populated | Populated |
| `is_event` | Binary event indicator (`0` or `1`) | Blank | `0` or `1` |
| `event_type` | Event class (`method_proposed`, `release`, `benchmark`, `trend_application`, `none`) | Blank | Valid type |
| `annotator` | Annotator identity/name | Blank | Populated |
| `label_method` | Labeling method. Use `human` after a student annotator reviews and accepts the label. | Blank | `human` |
| `notes` | Free-form notes/comments | Blank | Optional |

---

## 3. Workflow Commands

### Exporting Candidates
To re-generate the dataset candidates (1800 rows) and the pilot sample (50 rows) from processed sentences, run:
```bash
python scripts/03_export_labeling_data.py
```

### Validating Candidates (Pre-label Mode)
To check the schema and ensure no formatting errors exist before starting annotation:
```bash
python scripts/11_validate_labeled_data.py --input data/labeled/labeling_candidates.csv --mode prelabel
```

### Validating Completed Annotations (Labeled Mode)
To run strict checks on final labeled files, enforcing valid event type mappings, annotator names, and methods:
```bash
python scripts/11_validate_labeled_data.py --input data/labeled/labeling_candidates.csv --mode labeled
```

---

## 4. Student Labeling Workflow

1. **Setup**:
   - Open `data/labeled/labeling_candidates.csv` in Excel or any CSV editor (UTF-8-sig format prevents encoding issues).
2. **Guidelines Review**:
   - Review [LABELING_GUIDE.md](file:///d:/chrono-rag-assistant/docs/LABELING_GUIDE.md) for event class definitions and positive/negative examples.
3. **Pilot Practice**:
   - Complete the 50 pilot rows in `labeling_candidates_sample.csv` first. Run validation check in labeled mode:
     ```bash
     python scripts/11_validate_labeled_data.py --input data/labeled/labeling_candidates_sample.csv --mode labeled
     ```
4. **Annotating candidates**:
   - Fill in:
     - `is_event` (`0` or `1`)
     - `event_type` (`method_proposed`, `release`, `benchmark`, `trend_application`, or `none`)
     - `annotator` (e.g., your username or name)
     - `label_method` (`human`)
     - `notes` (optional rationale)
5. **Final Check**:
   - Run the validation CLI in `labeled` mode on the main candidate file to catch any missing fields or incorrect combinations. Ensure it exits with `0 errors`.
