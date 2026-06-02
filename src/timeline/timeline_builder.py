import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import setup_logger
from src.timeline.deduplicate import deduplicate_events

logger = setup_logger("timeline_builder")

def build_timeline(
    events: List[Dict[str, Any]],
    topic: str,
    output_path: Path = None
) -> Dict[str, Any]:
    """
    Build a timeline from deduplicated events.
    """
    # 1. Sort events by extracted_year (ascending), then by event_prob (descending)
    # We must handle None years carefully, putting them at the end or excluding them
    valid_events = [e for e in events if e.get("extracted_year") is not None]
    
    valid_events.sort(key=lambda e: (
        e.get("extracted_year", 9999),
        -e.get("event_prob", 0.0)
    ))

    # 2. Format each event into the timeline schema
    formatted_events = []
    for i, e in enumerate(valid_events):
        event_id = f"{topic}_evt_{i+1:04d}"
        
        # Use existing cluster sources if available, else just use the event's doc
        sources = e.get("cluster_sources", [{
            "doc_id": e.get("doc_id", ""),
            "title": e.get("title", ""),
            "source_url": e.get("source_url", ""),
            "chunk_id": e.get("chunk_id", "")
        }])
        
        # Truncate title/text for display title
        text = e.get("text", "")
        title = text[:80] + ("..." if len(text) > 80 else "")

        formatted_events.append({
            "event_id": event_id,
            "date": e.get("date_text", str(e.get("extracted_year"))),
            "year": e.get("extracted_year"),
            "event_type": e.get("event_type", "none"),
            "title": title,
            "representative_sentence": text,
            "confidence": e.get("event_prob", 0.0),
            "sources": sources,
            "cluster_size": e.get("cluster_size", 1)
        })

    # 3. Wrap in timeline envelope
    timeline = {
        "topic": topic,
        "generated_at": datetime.now().isoformat(),
        "total_events": len(formatted_events),
        "events": formatted_events
    }

    # 4. Save JSON if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

    return timeline

def build_all_timelines(
    predictions_path: Path,
    output_dir: Path,
    similarity_threshold: float = 0.78,
    year_diff_threshold: int = 1,
    model_name: str = 'all-MiniLM-L6-v2'
) -> Dict[str, Any]:
    """
    Build timelines for all topics from predictions.jsonl.
    """
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = read_jsonl(predictions_path)
    
    # Filter to only event sentences
    events = [p for p in predictions if p.get("is_event") == 1]
    logger.info(f"Filtered {len(events)} event sentences out of {len(predictions)} total predictions")
    
    # Group by topic
    topic_events = {}
    for e in events:
        topic = e.get("topic", "unknown")
        if topic not in topic_events:
            topic_events[topic] = []
        topic_events[topic].append(e)
        
    combined_timelines = {"timelines": {}}
    
    # Process each topic
    for topic, t_events in topic_events.items():
        logger.info(f"Processing topic '{topic}' with {len(t_events)} candidate events")
        
        # Deduplicate
        deduped = deduplicate_events(
            t_events, 
            similarity_threshold=similarity_threshold,
            year_diff_threshold=year_diff_threshold,
            model_name=model_name
        )
        logger.info(f"  After deduplication: {len(deduped)} clusters (reduction: {100 * (1 - len(deduped)/len(t_events)):.1f}%)")
        
        # Build timeline
        topic_output_path = output_dir / f"timeline_{topic}.json"
        timeline = build_timeline(deduped, topic, output_path=topic_output_path)
        logger.info(f"  Saved topic timeline to {topic_output_path}")
        
        combined_timelines["timelines"][topic] = timeline
        
    # Save combined timeline
    combined_path = output_dir / "timeline.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_timelines, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved combined timeline to {combined_path}")
        
    return combined_timelines
