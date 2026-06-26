from pathlib import Path
from typing import List, Dict, Any
from src.utils.io import read_csv
from src.utils.logger import setup_logger
from src.ingest.document_parser import parse_file

logger = setup_logger("document_loader")

def load_documents(
    metadata_path: Path,
    raw_dir: Path,
    *,
    extract_tables: bool = False,
    ocr: bool = False,
    pdf_backend: str = "pymupdf",
) -> List[Dict[str, Any]]:
    """
    Load documents defined in metadata_path from raw_dir.
    Returns a list of document dictionaries following the schema.

    ``extract_tables``, ``ocr`` and ``pdf_backend`` are forwarded to the PDF
    parser (see ``parse_file``); defaults preserve the current output.
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

        try:
            text = parse_file(
                full_local_path, doc_id=doc_id,
                extract_tables=extract_tables, ocr=ocr, pdf_backend=pdf_backend,
            )

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
        except FileNotFoundError:
            logger.warning(f"[{doc_id}] File not found: {full_local_path}. Skipping.")
        except ValueError as ve:
            logger.warning(f"[{doc_id}] Skipping {full_local_path}: {ve}")
        except Exception as e:
            logger.error(f"[{doc_id}] Failed to read/parse {full_local_path}: {e}")

    logger.info(f"Successfully loaded {len(documents)} documents.")
    return documents
