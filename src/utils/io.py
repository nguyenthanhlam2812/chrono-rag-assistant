import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union

def read_text(file_path: Union[str, Path]) -> str:
    """Read contents of a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(file_path: Union[str, Path], content: str) -> None:
    """Write text content to a file, creating parent directories if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    data = []
    if not Path(file_path).exists():
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(file_path: Union[str, Path], data: List[Dict[str, Any]], append: bool = False) -> None:
    """Write a list of dictionaries to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def read_csv(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    data = []
    if not Path(file_path).exists():
        return data
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data

def write_csv(file_path: Union[str, Path], data: List[Dict[str, Any]], fieldnames: List[str] = None) -> None:
    """Write a list of dictionaries to a CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
