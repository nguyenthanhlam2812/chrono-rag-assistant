# ChronoRAG Annotation Guidelines

This guide defines the methodology for annotating sentences as temporal/chronological events for the ChronoRAG dataset. 

---

## 1. Schema & Variables

Every row in the candidate dataset contains the following labeling columns:
- **`is_event`**: Binary indicator (`0` or `1`). Set to `1` if the sentence mentions a distinct technological milestone or event, and `0` otherwise.
- **`event_type`**: The specific class of the event. Must be one of:
  - `method_proposed`
  - `release`
  - `benchmark`
  - `trend_application`
  - `none` (must be `none` if `is_event = 0`, and must not be `none` if `is_event = 1`).
- **`annotator`**: The name or ID of the person annotating the sentence (e.g. `human` or student name/ID).
- **`label_method`**: The method of labeling. Must be one of: `human`, `llm_reviewed`, `llm_only`.
- **`notes`**: Free-form text for annotation rationales or questions.

---

## 2. Event Types & Definitions

An event is defined as a specific occurrence or technological milestone situated in time, usually containing verbs of action (e.g., *propose*, *release*, *outperform*, *apply*) and involving entities (models, datasets, frameworks).

### 1. `method_proposed`
- **Definition**: The introduction or proposal of a new method, architecture, algorithm, model design, or conceptual framework.
- **Key Indicators**: Sentence describes the birth of a technique (e.g. "We propose...", "We introduce...", "X is a new approach that...").

### 2. `release`
- **Definition**: The public release of an artifact, such as open-source code, weight checkpoints, pre-trained models, software packages, api platforms, or tools.
- **Key Indicators**: Sentence mentions release or availability (e.g., "released on GitHub", "is open source", "made publicly available").

### 3. `benchmark`
- **Definition**: The introduction of a new evaluation suite, test set, or benchmark, or the reporting of specific benchmark results comparing models.
- **Key Indicators**: "We evaluate on...", "outperforms Y on SQuAD by X%", "new benchmark suite".

### 4. `trend_application`
- **Definition**: Synthesizing a general research direction, adoption trend, or applying an existing method to a new domain/task (rather than proposing a new method).
- **Key Indicators**: "Recently, there is a trend...", "We apply BERT to clinical text classification".

### 5. `none`
- **Definition**: No chronological event is present in the sentence. This includes generic definitions, general background facts, background math formulas, or table captions.

---

## 3. Positive Examples (10)

1. **`method_proposed`**
   - *Sentence*: "In this work, we propose REALM, which is a novel retrieval-augmented language model pre-training framework."
   - *Rationale*: Proposes a new model architecture/framework.
2. **`method_proposed`**
   - *Sentence*: "To address this, we introduce Toolformer, a model trained to decide which tools to use."
   - *Rationale*: Proposes a new tool-using model.
3. **`release`**
   - *Sentence*: "We release all code, parameters, and checkpoints at github.com/facebookresearch/dpr."
   - *Rationale*: Describes public code/weights release.
4. **`release`**
   - *Sentence*: "LangChain was released as an open-source library in October 2022."
   - *Rationale*: Clearly specifies the public launch event of a framework.
5. **`benchmark`**
   - *Sentence*: "We evaluate our approach on the HotpotQA and FEVER datasets, demonstrating state-of-the-art results."
   - *Rationale*: Details model evaluation on specific benchmarks.
6. **`benchmark`**
   - *Sentence*: "Our proposed model achieves an accuracy of 85.3% on MMLU, outperforming GPT-3.5 by 4.2%."
   - *Rationale*: Reports a specific benchmark performance comparison.
7. **`trend_application`**
   - *Sentence*: "Recently, retrieval-augmented generation has become a dominant trend for reducing LLM hallucinations."
   - *Rationale*: Summarizes a research trend in the NLP domain.
8. **`trend_application`**
   - *Sentence*: "We apply the concept of knowledge distillation to compress dense passage retrievers for mobile devices."
   - *Rationale*: Describes an application of an existing technique (knowledge distillation) to a different domain (dense retrieval).
9. **`method_proposed`**
   - *Sentence*: "We present Self-RAG, a framework that trains language models to self-reflect using critiques."
   - *Rationale*: Proposes/presents a new framework (Self-RAG).
10. **`release`**
    - *Sentence*: "The code and models are made publicly available at huggingface.co/models."
    - *Rationale*: States public code/model availability.

---

## 4. Negative Examples (10)

1. **`none`**
   - *Sentence*: "Retrieval-Augmented Generation (RAG) is a technique that fetches external knowledge to ground LLMs."
   - *Rationale*: Generic definition, no specific event or milestone occurred.
2. **`none`**
   - *Sentence*: "Let $x$ be the input sequence and $y$ be the corresponding output token distribution."
   - *Rationale*: Mathematical definition/notation.
3. **`none`**
   - *Sentence*: "Table 2 shows the hyperparameters used for pre-training our language models."
   - *Rationale*: Structural reference to a table/figure in the text.
4. **`none`**
   - *Sentence*: "The Transformer architecture relies entirely on self-attention mechanisms."
   - *Rationale*: Fact/description of a standard architecture without proposing or releasing anything new in the context.
5. **`none`**
   - *Sentence*: "Future work will explore extending our method to multi-lingual settings."
   - *Rationale*: Future plans, not a concrete past/present event.
6. **`none`**
   - *Sentence*: "We describe the details of our experimental setup in Appendix A."
   - *Rationale*: Document structural description.
7. **`none`**
   - *Sentence*: "The dataset contains 10,000 training instances and 1,000 validation instances."
   - *Rationale*: Static dataset description.
8. **`none`**
   - *Sentence*: "Previous studies have attempted to solve this issue through various fine-tuning strategies."
   - *Rationale*: Generic summary of literature without listing specific models, papers, or trends.
9. **`none`**
   - *Sentence*: "We define the objective function as the negative log-likelihood of the target token."
   - *Rationale*: Equation definition.
10. **`none`**
    - *Sentence*: "The remainder of this paper is structured as follows."
    - *Rationale*: Formatting/organizational sentence.

---

## 5. Borderline / Ambiguous Cases

- **Case 1: Discussing another paper's proposal**
  - *Sentence*: "Vaswani et al. (2017) introduced the Transformer architecture based on self-attention."
  - *Annotation*: `is_event = 1`, `event_type = method_proposed`.
  - *Rationale*: Although the sentence refers to third-party work, it refers to a distinct, named method proposal event.
- **Case 2: Vague benchmarks/evaluations**
  - *Sentence*: "We test our models on several standard benchmarks."
  - *Annotation*: `is_event = 0`, `event_type = none`.
  - *Rationale*: Too vague. A valid `benchmark` event must specify actual dataset names (e.g. SQuAD, TriviaQA, GLUE) or concrete score differences.
- **Case 3: Incremental feature release vs new proposal**
  - *Sentence*: "We add a new retrieval API endpoint to the LlamaIndex package."
  - *Annotation*: `is_event = 1`, `event_type = release`.
  - *Rationale*: Fits under the release of an updated artifact/tool functionality.

---

## 6. Critical Dataset Design Guidelines

1. **Global `sentence_id`**: The `sentence_id` is a globally unique identifier (e.g. `agent_001_s231`) mapped from the document ID. Annotators should keep this ID intact. Do not re-index or modify the IDs in the CSV.
2. **Document-Level Splits**: When partitioning this dataset into train, validation, and test splits, **splits must be performed at the document level (`doc_id`)**, not at the sentence level. Performing sentence-level random splits will lead to severe data leakage (e.g., test sentences describing the evaluation of a model alongside training sentences describing its proposal from the same paper).
