import os
import re
import csv
import json
import random
from pathlib import Path

class SentenceScorer:
    """Scores sentences based on patterns, verbs, and entities to estimate event likelihood."""
    def __init__(self, config):
        self.verbs = config["scoring"]["event_verbs"]
        self.entities = config["scoring"]["entity_names"]
        self.date_patterns = [re.compile(p, re.IGNORECASE) for p in config["scoring"]["date_patterns"]]

    def score(self, text):
        score = 0
        # 1. Date patterns (weighted 2)
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            score += 2 * len(matches)

        # 2. Verbs (weighted 1)
        for verb in self.verbs:
            matches = re.findall(rf"\b{re.escape(verb)}\b", text, re.IGNORECASE)
            score += len(matches)

        # 3. Entities (weighted 1)
        for entity in self.entities:
            matches = re.findall(rf"\b{re.escape(entity)}\b", text, re.IGNORECASE)
            score += len(matches)

        return score


def parse_simple_yaml(text):
    import re
    config = {}
    current_key_path = []
    
    for line in text.splitlines():
        # Remove comments and strip whitespace from the right
        stripped = line.split('#')[0].rstrip()
        if not stripped:
            continue
            
        # Determine indentation
        indent = len(line) - len(line.lstrip())
        
        # Check list item
        list_match = re.match(r'^\s*-\s*(.*)$', stripped)
        if list_match:
            val = list_match.group(1).strip()
            # Remove quotes if present
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            val = val.replace('\\\\', '\\')
            
            # Find the parent list in config
            parent = config
            for kp in current_key_path:
                parent = parent[kp]
            parent.append(val)
            continue
            
        # Check key-value pair or section
        kv_match = re.match(r'^\s*([a-zA-Z0-9_\-]+)\s*:\s*(.*)$', stripped)
        if kv_match:
            key = kv_match.group(1).strip()
            val = kv_match.group(2).strip()
            
            # Adjust key path based on indentation (assuming 2 spaces per indent level)
            level = indent // 2
            current_key_path = current_key_path[:level]
            
            if val == "":
                # It's a new section/dictionary/list
                if key in {"event_verbs", "entity_names", "date_patterns"}:
                    new_val = []
                else:
                    new_val = {}
                
                # Assign to parent
                if not current_key_path:
                    config[key] = new_val
                else:
                    parent = config
                    for kp in current_key_path:
                        parent = parent[kp]
                    parent[key] = new_val
                current_key_path.append(key)
            else:
                # Key with value
                if val.lower() == "true":
                    val_parsed = True
                elif val.lower() == "false":
                    val_parsed = False
                elif val.lower() == "null":
                    val_parsed = None
                else:
                    try:
                        if "." in val:
                            val_parsed = float(val)
                        else:
                            val_parsed = int(val)
                    except ValueError:
                        # String value, remove quotes
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        val_parsed = val
                
                # Assign to parent
                if not current_key_path:
                    config[key] = val_parsed
                else:
                    parent = config
                    for kp in current_key_path:
                        parent = parent[kp]
                    parent[key] = val_parsed
                current_key_path.append(key)
                
    return config


class LabelingExporter:
    """Handles parsing, scoring, deterministic sampling, and exporting of labeling datasets."""
    def __init__(self, config_path, sentences_path, metadata_path):
        self.config_path = Path(config_path)
        self.sentences_path = Path(sentences_path)
        self.metadata_path = Path(metadata_path)

        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.config = parse_simple_yaml(content)

        self.scorer = SentenceScorer(self.config)

    def load_metadata_titles(self):
        """Maps doc_id to its official title from metadata.csv."""
        titles = {}
        if not self.metadata_path.exists():
            return titles
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                titles[row["doc_id"]] = row["title"]
        return titles

    def load_sentences(self):
        """Loads processed sentences from sentences.jsonl."""
        sentences = []
        with open(self.sentences_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sentences.append(json.loads(line))
        return sentences

    def allocate_doc_budgets(self, doc_sentence_counts, target_total):
        """Allocates target_total budget among documents using deterministic round-robin."""
        min_per_doc = self.config["sampling"]["min_per_doc"]
        per_doc_cap = self.config["sampling"]["per_doc_cap"]
        small_doc_threshold = self.config["sampling"]["small_doc_threshold"]

        k_d = {}
        max_allowed_d = {}
        for doc_id, n_d in doc_sentence_counts.items():
            k_d[doc_id] = min(n_d, min_per_doc)
            max_allowed_d[doc_id] = n_d if n_d <= small_doc_threshold else min(n_d, per_doc_cap)

        remaining_budget = target_total - sum(k_d.values())
        if remaining_budget < 0:
            # Over-allocated already (extremely rare since min_per_doc * number of docs << target_total)
            # Scale back deterministically
            sorted_docs = sorted(k_d.keys())
            for doc_id in sorted_docs:
                if remaining_budget >= 0:
                    break
                k_d[doc_id] = max(0, k_d[doc_id] - 1)
                remaining_budget += 1
            return k_d

        # Distribute remaining budget in round-robin fashion
        sorted_docs = sorted(k_d.keys())
        while remaining_budget > 0:
            allocated_in_round = False
            for doc_id in sorted_docs:
                if remaining_budget <= 0:
                    break
                if k_d[doc_id] < max_allowed_d[doc_id]:
                    k_d[doc_id] += 1
                    remaining_budget -= 1
                    allocated_in_round = True
            if not allocated_in_round:
                # No document can accept more sentences (limit reached)
                break

        return k_d

    def sample_from_doc(self, sentences, k_d, rng):
        """Samples k_d sentences from a document distributed by score buckets with deterministic fallback."""
        # Score and assign bucket
        scored_sentences = []
        for s in sentences:
            score = self.scorer.score(s["text"])
            if score >= 3:
                bucket = "high"
            elif 1 <= score <= 2:
                bucket = "medium"
            else:
                bucket = "low"
            scored_sentences.append({
                "sentence": s,
                "score": score,
                "original_bucket": bucket
            })

        # Separate pools
        high_pool = [s for s in scored_sentences if s["original_bucket"] == "high"]
        med_pool = [s for s in scored_sentences if s["original_bucket"] == "medium"]
        low_pool = [s for s in scored_sentences if s["original_bucket"] == "low"]

        # Sort to ensure stable starting point before shuffle
        high_pool.sort(key=lambda x: x["sentence"]["sentence_id"])
        med_pool.sort(key=lambda x: x["sentence"]["sentence_id"])
        low_pool.sort(key=lambda x: x["sentence"]["sentence_id"])

        # Deterministic shuffle
        rng.shuffle(high_pool)
        rng.shuffle(med_pool)
        rng.shuffle(low_pool)

        # Target bucket counts
        n_high = round(self.config["sampling"]["bucket_ratio"]["high"] * k_d)
        n_med = round(self.config["sampling"]["bucket_ratio"]["medium"] * k_d)
        n_low = k_d - n_high - n_med

        selected = []
        stats = {
            "requested": {"high": n_high, "medium": n_med, "low": n_low},
            "actual": {"high": 0, "medium": 0, "low": 0},
            "fallback_used": False
        }

        # Helper to draw from pool
        def draw(n, preferences):
            drawn = []
            for pool in preferences:
                needed = n - len(drawn)
                if needed <= 0:
                    break
                take = min(needed, len(pool))
                for _ in range(take):
                    drawn.append(pool.pop(0))
            return drawn

        drawn_high = draw(n_high, [high_pool, med_pool, low_pool])
        drawn_med = draw(n_med, [med_pool, high_pool, low_pool])
        drawn_low = draw(n_low, [low_pool, med_pool, high_pool])

        # Record bucket stats
        for item in drawn_high:
            stats["actual"][item["original_bucket"]] += 1
            if item["original_bucket"] != "high":
                stats["fallback_used"] = True
        for item in drawn_med:
            stats["actual"][item["original_bucket"]] += 1
            if item["original_bucket"] != "medium":
                stats["fallback_used"] = True
        for item in drawn_low:
            stats["actual"][item["original_bucket"]] += 1
            if item["original_bucket"] != "low":
                stats["fallback_used"] = True

        selected = drawn_high + drawn_med + drawn_low
        # Re-sort selected by original sentence_id order
        selected.sort(key=lambda x: x["sentence"]["sentence_id"])
        return selected, stats

    def export(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        titles = self.load_metadata_titles()
        sentences = self.load_sentences()

        # Group by topic
        topic_sentences = {}
        for s in sentences:
            topic = s["topic"]
            topic_sentences.setdefault(topic, []).append(s)

        rng = random.Random(self.config["sampling"]["random_seed"])

        final_samples = []
        report_data = {
            "config": self.config["sampling"],
            "total_processed_sentences": len(sentences),
            "topics": {}
        }

        for topic in sorted(topic_sentences.keys()):
            topic_list = topic_sentences[topic]
            # Group by doc_id
            doc_groups = {}
            for s in topic_list:
                doc_groups.setdefault(s["doc_id"], []).append(s)

            doc_counts = {doc_id: len(lst) for doc_id, lst in doc_groups.items()}
            per_topic_target = self.config["sampling"]["per_topic_target"]
            doc_budgets = self.allocate_doc_budgets(doc_counts, per_topic_target)

            topic_selected = []
            topic_report = {
                "total_sentences": len(topic_list),
                "target_sampled": per_topic_target,
                "doc_allocations": {},
                "bucket_stats": {"high": 0, "medium": 0, "low": 0}
            }

            for doc_id in sorted(doc_groups.keys()):
                doc_sentences = doc_groups[doc_id]
                k_d = doc_budgets[doc_id]
                sampled_items, stats = self.sample_from_doc(doc_sentences, k_d, rng)
                topic_selected.extend(sampled_items)

                # Update report
                topic_report["doc_allocations"][doc_id] = {
                    "total_available": len(doc_sentences),
                    "sampled": k_d,
                    "bucket_stats": stats
                }
                for b in ["high", "medium", "low"]:
                    topic_report["bucket_stats"][b] += stats["actual"][b]

            final_samples.extend(topic_selected)
            report_data["topics"][topic] = topic_report

        # Sort final samples by sentence_id to keep order pristine
        final_samples.sort(key=lambda x: x["sentence"]["sentence_id"])

        # Write labeling_candidates.csv
        candidates_csv_path = output_path / "labeling_candidates.csv"
        headers = [
            "sentence_id", "doc_id", "chunk_id", "topic", "title",
            "source_url", "year", "sentence", "is_event", "event_type",
            "annotator", "label_method", "notes"
        ]

        with open(candidates_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in final_samples:
                s = item["sentence"]
                writer.writerow([
                    s["sentence_id"],
                    s["doc_id"],
                    s["chunk_id"],
                    s["topic"],
                    titles.get(s["doc_id"], ""),
                    s["source_url"],
                    s["year"],
                    s["text"],
                    "",  # is_event (blank)
                    "",  # event_type (blank)
                    "",  # annotator (blank)
                    "",  # label_method (blank)
                    ""   # notes (blank)
                ])

        # Write labeling_export_report.json
        report_data["total_sampled_candidates"] = len(final_samples)
        report_json_path = output_path / "labeling_export_report.json"
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        # Generate pilot sample (labeling_candidates_sample.csv) - exactly 50 rows
        # Balance across topics: rag:17, ai_agent:17, knowledge_distillation:16
        pilot_samples = []
        # Group final_samples by topic
        samples_by_topic = {}
        for item in final_samples:
            samples_by_topic.setdefault(item["sentence"]["topic"], []).append(item)

        pilot_targets = {
            "rag": 17,
            "ai_agent": 17,
            "knowledge_distillation": 16
        }

        # Reset rng for pilot selection to keep it separate but deterministic
        pilot_rng = random.Random(self.config["sampling"]["random_seed"])

        for topic in sorted(pilot_targets.keys()):
            target_count = pilot_targets[topic]
            items = samples_by_topic.get(topic, [])

            # We want a mix of high, medium, low score sentences
            high_items = [x for x in items if x["original_bucket"] == "high"]
            med_items = [x for x in items if x["original_bucket"] == "medium"]
            low_items = [x for x in items if x["original_bucket"] == "low"]

            # Targets within the pilot topic:
            # Let's allocate roughly equal parts high, med, low
            p_high = target_count // 3
            p_med = target_count // 3
            p_low = target_count - p_high - p_med

            # Shuffle pools deterministically
            pilot_rng.shuffle(high_items)
            pilot_rng.shuffle(med_items)
            pilot_rng.shuffle(low_items)

            topic_pilot = []
            topic_pilot.extend(high_items[:p_high])
            topic_pilot.extend(med_items[:p_med])
            topic_pilot.extend(low_items[:p_low])

            # Fill if there's any remaining target deficit
            all_remaining = [x for x in items if x not in topic_pilot]
            pilot_rng.shuffle(all_remaining)
            deficit = target_count - len(topic_pilot)
            topic_pilot.extend(all_remaining[:deficit])

            pilot_samples.extend(topic_pilot)

        # Sort pilot samples by sentence_id
        pilot_samples.sort(key=lambda x: x["sentence"]["sentence_id"])

        sample_csv_path = output_path / "labeling_candidates_sample.csv"
        with open(sample_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in pilot_samples:
                s = item["sentence"]
                writer.writerow([
                    s["sentence_id"],
                    s["doc_id"],
                    s["chunk_id"],
                    s["topic"],
                    titles.get(s["doc_id"], ""),
                    s["source_url"],
                    s["year"],
                    s["text"],
                    "",  # is_event
                    "",  # event_type
                    "",  # annotator
                    "",  # label_method
                    ""   # notes
                ])

        print(f"Export completed: {len(final_samples)} rows in candidates, 50 rows in sample.")
        return len(final_samples), len(pilot_samples)
