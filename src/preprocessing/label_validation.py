import csv
import json
from pathlib import Path

def validate_labeled_data(csv_path, sentences_jsonl_path, mode="prelabel"):
    """
    Validates a labeled or pre-labeled CSV file for ChronoRAG dataset integrity.
    
    Args:
        csv_path (str or Path): Path to the CSV file to validate.
        sentences_jsonl_path (str or Path): Path to the original processed sentences.jsonl.
        mode (str): Validation mode, either "prelabel" or "labeled".
        
    Returns:
        dict: A dictionary containing 'errors' (list of str), 'warnings' (list of str), and 'stats' (dict).
    """
    csv_path = Path(csv_path)
    sentences_jsonl_path = Path(sentences_jsonl_path)
    
    report = {
        "errors": [],
        "warnings": [],
        "stats": {
            "total_rows": 0,
            "topic_counts": {},
            "doc_counts": {},
            "is_event_counts": {},
            "event_type_counts": {},
            "label_method_counts": {}
        }
    }
    
    # 1. Load original sentence IDs for reference validation
    valid_sentence_ids = set()
    if sentences_jsonl_path.exists():
        try:
            with open(sentences_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "sentence_id" in data:
                            valid_sentence_ids.add(data["sentence_id"])
        except Exception as e:
            report["errors"].append(f"Failed to read processed sentences file: {e}")
            return report
    else:
        # If the file does not exist, we log a warning or error. Since we are in regression tests,
        # it might not exist if we run in a detached test dir. Let's warn but not fail unless we are in strict mode.
        report["warnings"].append(f"Original processed sentences file not found: {sentences_jsonl_path}. Reference checking skipped.")

    # 2. Check CSV existence
    if not csv_path.exists():
        report["errors"].append(f"Target CSV file not found: {csv_path}")
        return report

    # 3. Read and validate CSV structure/schema
    required_cols = [
        "sentence_id", "doc_id", "chunk_id", "topic", "title", "source_url",
        "year", "sentence", "is_event", "event_type", "annotator", "label_method", "notes"
    ]
    
    rows = []
    try:
        # Excel-compatible CSV might start with UTF-8 BOM, let's open with utf-8-sig
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            if not headers:
                report["errors"].append("CSV file is empty or has no header row.")
                return report
                
            # Check for missing columns
            missing_cols = [col for col in required_cols if col not in headers]
            if missing_cols:
                report["errors"].append(f"CSV missing required columns: {missing_cols}")
                return report
                
            for row in reader:
                rows.append(row)
    except Exception as e:
        report["errors"].append(f"Failed to parse CSV file: {e}")
        return report

    total_rows = len(rows)
    report["stats"]["total_rows"] = total_rows
    if total_rows == 0:
        report["errors"].append("CSV contains no data rows.")
        return report

    # 4. Row-by-row validation
    seen_sentence_ids = set()
    valid_event_types = {"method_proposed", "release", "benchmark", "trend_application", "none"}
    valid_label_methods = {"human"}
    
    for idx, row in enumerate(rows, start=2): # 1-based index including header
        sentence_id = row.get("sentence_id", "").strip()
        doc_id = row.get("doc_id", "").strip()
        sentence_text = row.get("sentence", "").strip()
        is_event = row.get("is_event", "").strip()
        event_type = row.get("event_type", "").strip()
        annotator = row.get("annotator", "").strip()
        label_method = row.get("label_method", "").strip()
        topic = row.get("topic", "").strip()

        # Check: unique sentence_id
        if not sentence_id:
            report["errors"].append(f"Row {idx}: Missing sentence_id.")
        else:
            if sentence_id in seen_sentence_ids:
                report["errors"].append(f"Row {idx}: Duplicate sentence_id '{sentence_id}'.")
            seen_sentence_ids.add(sentence_id)
            
            # Check: sentence_id exists in processed sentences
            if valid_sentence_ids and sentence_id not in valid_sentence_ids:
                report["errors"].append(f"Row {idx}: sentence_id '{sentence_id}' does not exist in processed sentences.jsonl.")

        # Check: empty sentence text
        if not sentence_text:
            report["errors"].append(f"Row {idx}: Empty sentence text.")
            
        # Warning: sentence length < 20 or > 500 characters
        if sentence_text:
            text_len = len(sentence_text)
            if text_len < 20 or text_len > 500:
                report["warnings"].append(
                    f"Row {idx}: Sentence '{sentence_id}' length {text_len} is outside threshold (20-500 chars)."
                )

        # Statistics accumulation
        if topic:
            report["stats"]["topic_counts"][topic] = report["stats"]["topic_counts"].get(topic, 0) + 1
        if doc_id:
            report["stats"]["doc_counts"][doc_id] = report["stats"]["doc_counts"].get(doc_id, 0) + 1
        if is_event:
            report["stats"]["is_event_counts"][is_event] = report["stats"]["is_event_counts"].get(is_event, 0) + 1
        if event_type:
            report["stats"]["event_type_counts"][event_type] = report["stats"]["event_type_counts"].get(event_type, 0) + 1
        if label_method:
            report["stats"]["label_method_counts"][label_method] = report["stats"]["label_method_counts"].get(label_method, 0) + 1

        # 5. Mode-specific check: labeled or prelabel
        if mode == "labeled":
            # is_event must be '0' or '1'
            if is_event not in {"0", "1"}:
                report["errors"].append(
                    f"Row {idx}: is_event must be '0' or '1', found '{is_event}'."
                )
            
            # event_type must be valid
            if event_type not in valid_event_types:
                report["errors"].append(
                    f"Row {idx}: event_type must be one of {list(valid_event_types)}, found '{event_type}'."
                )
            
            # semantic compatibility checks
            if is_event == "0" and event_type != "none":
                report["errors"].append(
                    f"Row {idx}: event_type must be 'none' when is_event is '0'."
                )
            if is_event == "1" and event_type == "none":
                report["errors"].append(
                    f"Row {idx}: event_type must not be 'none' when is_event is '1'."
                )
                
            # annotator must not be blank
            if not annotator:
                report["errors"].append(f"Row {idx}: annotator is blank in labeled mode.")
                
            # label_method must be valid
            if label_method not in valid_label_methods:
                report["errors"].append(
                    f"Row {idx}: label_method must be one of {list(valid_label_methods)}, found '{label_method}'."
                )
        else:
            # prelabel mode validation for partial entries
            if is_event not in {"", "0", "1"}:
                report["errors"].append(
                    f"Row {idx}: is_event must be blank, '0', or '1', found '{is_event}'."
                )
            if event_type not in ({""} | valid_event_types):
                report["errors"].append(
                    f"Row {idx}: event_type must be blank or one of {list(valid_event_types)}, found '{event_type}'."
                )
            if label_method not in ({""} | valid_label_methods):
                report["errors"].append(
                    f"Row {idx}: label_method must be blank or one of {list(valid_label_methods)}, found '{label_method}'."
                )
            # Semantic compatibility in prelabel mode if both fields are partially supplied
            if is_event in {"0", "1"} and event_type in valid_event_types:
                if is_event == "0" and event_type != "none":
                    report["errors"].append(
                        f"Row {idx}: event_type must be 'none' when is_event is '0'."
                    )
                if is_event == "1" and event_type == "none":
                    report["errors"].append(
                        f"Row {idx}: event_type must not be 'none' when is_event is '1'."
                    )

    # 6. Global statistics validation (Warnings only)
    # Warning: any doc contributes > 8% of total rows
    for doc_id, count in report["stats"]["doc_counts"].items():
        ratio = count / total_rows
        if ratio > 0.08:
            report["warnings"].append(
                f"Document '{doc_id}' contributes {count} rows ({ratio:.2%}), exceeding the 8.00% dominance threshold."
            )

    # Warning: topic distribution is outside 30.0% - 36.6%
    for topic, count in report["stats"]["topic_counts"].items():
        ratio = count / total_rows
        if ratio < 0.300 or ratio > 0.366:
            report["warnings"].append(
                f"Topic '{topic}' has distribution {ratio:.2%} ({count}/{total_rows}), which is outside the target 30.0%-36.6% range."
            )

    return report
