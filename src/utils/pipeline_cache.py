from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import read_csv, read_jsonl


class PipelineCache:
    """Small manifest-based cache for expensive offline pipeline stages.

    The cache is deliberately simple and transparent. A stage can be reused
    only when its output JSONL exists and its stored fingerprint matches the
    current raw files/config fingerprint. It never changes the output schema.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.manifest_path = cache_dir / "offline_pipeline_manifest.json"
        self._manifest = self._load_manifest()

    def load_jsonl_if_valid(
        self,
        stage: str,
        fingerprint: str,
        output_path: Path,
    ) -> List[Dict[str, Any]] | None:
        if not self.enabled or not output_path.exists():
            return None
        entry = self._manifest.get(stage, {})
        if entry.get("fingerprint") != fingerprint:
            return None
        rows = read_jsonl(output_path)
        return rows if rows else None

    def mark_valid(self, stage: str, fingerprint: str, output_path: Path, count: int) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest[stage] = {
            "fingerprint": fingerprint,
            "output_path": str(output_path),
            "count": count,
        }
        self.manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.enabled or not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def fingerprint_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_corpus_fingerprint(
    metadata_csv_path: Path,
    raw_dir: Path,
    parser_config: Dict[str, Any],
) -> str:
    rows = read_csv(metadata_csv_path)
    file_stats = []
    for row in rows:
        local_path = str(row.get("local_path", "") or "").strip()
        if not local_path:
            continue
        path = raw_dir / local_path
        file_stats.append(_file_stat_payload(path))
    return fingerprint_payload(
        {
            "metadata": _file_stat_payload(metadata_csv_path),
            "rows": rows,
            "files": file_stats,
            "parser_config": parser_config,
        }
    )


def chained_fingerprint(*parts: Any) -> str:
    return fingerprint_payload(list(parts))


def _file_stat_payload(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
