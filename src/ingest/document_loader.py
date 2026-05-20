from pathlib import Path
from typing import List, Dict, Any
from src.utils.io import read_csv, read_text
from src.utils.logger import setup_logger

logger = setup_logger("document_loader")

def load_documents(metadata_path: Path, raw_dir: Path) -> List[Dict[str, Any]]:
    """
    Load documents defined in metadata_path from raw_dir.
    Returns a list of document dictionaries following the schema.
    """
    logger.info(f"Loading metadata from {metadata_path}")
    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}")
        return []
        
    rows = read_csv(metadata_path)
    documents = []

    for row in rows:
        doc_id = str(row["doc_id"])
        local_path_rel = str(row["local_path"])
        full_local_path = raw_dir / local_path_rel

        logger.info(f"Reading document {doc_id} from {full_local_path}")
        if not full_local_path.exists():
            logger.warning(f"File not found: {full_local_path}. Skipping.")
            continue

        try:
            text = read_text(full_local_path)

            # Parse authors list
            authors_str = (row.get("authors") or "").strip()
            authors = [author.strip() for author in authors_str.split(";") if author.strip()]

            # Parse year
            year_val = (row.get("year") or "").strip()
            year = int(float(year_val)) if year_val else None

            doc_dict = {
                "doc_id": doc_id,
                "title": str(row.get("title") or ""),
                "topic": str(row.get("topic") or ""),
                "source_type": str(row.get("source_type") or ""),
                "source_url": str(row.get("source_url") or ""),
                "published_date": str(row.get("published_date") or ""),
                "year": year,
                "authors": authors,
                "local_path": str(local_path_rel),
                "text": text,
                "retrieved_at": str(row.get("retrieved_at") or "")
            }
            documents.append(doc_dict)
        except Exception as e:
            logger.error(f"Failed to read/parse {full_local_path}: {e}")

    logger.info(f"Successfully loaded {len(documents)} documents.")
    return documents
