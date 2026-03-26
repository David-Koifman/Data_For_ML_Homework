# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
# Full automated pipeline (no interaction)
python run_pipeline.py <<< ""

# Run a single agent
python agents/data_collection_agent.py
python agents/data_quality_agent.py
python agents/annotation_agent.py
python agents/al_agent.py

# Execute a notebook
jupyter nbconvert --to notebook --execute notebooks/eda.ipynb --output eda.ipynb

# Install dependencies (use anaconda python if Jupyter is involved)
pip3 install -r requirements.txt
/opt/anaconda3/bin/pip install -r requirements.txt   # for notebooks
```

## Environment

- Requires `ANTHROPIC_API_KEY` in `.env` (used by AnnotationAgent and ALAgent for LLM calls)
- Two Python environments exist: system `python3` and `/opt/anaconda3`. Jupyter notebooks run under anaconda — install packages there if notebooks fail with ModuleNotFoundError.
- `python-dotenv` loads `.env` automatically in agent code.

## Architecture

**5 Claude Code skills** in `skills/` (registered in `.claude/skills/`):
- `/data-collection` → `skills/data_collection/SKILL.md` + `scripts/data_collection_agent.py`
- `/data-quality` → `skills/data_quality/SKILL.md` + `scripts/data_quality_agent.py`
- `/annotation` → `skills/annotation/SKILL.md` + `scripts/annotation_agent.py`
- `/active-learning` → `skills/active_learning/SKILL.md` + `scripts/al_agent.py`
- `/pipeline` → `skills/pipeline/SKILL.md` (master skill, orchestrates the 4 above)

Each SKILL.md describes what the agent should do, when to ask the user (HITL points), and references the script in `scripts/` as a ready-to-use tool.

**4 Python agents** in `agents/` (canonical source; `skills/*/scripts/` are copies):
- `DataCollectionAgent` — reads `config.yaml`, loads HuggingFace datasets + scrapes URLs. Output: `data/raw/collected.parquet` with columns `[text, label, source, collected_at]`
- `DataQualityAgent` — IQR outlier detection, deduplication, missing values, class imbalance. Has `llm_explain()` via Claude API for bonus.
- `AnnotationAgent` — zero-shot classification via Claude API (haiku model), outputs `auto_label` + `confidence` columns. Low-confidence examples (< 0.7) go to `review_queue.csv` for HITL.
- `ActiveLearningAgent` — entropy/margin/random query strategies with LogisticRegression + TF-IDF. Has `llm_analyze()` via Claude API.

**`run_pipeline.py`** chains all 4 agents sequentially with a blocking HITL pause at step 4. It is the non-interactive fallback; the primary workflow is via `/pipeline` skill in Claude Code.

## Data flow

```
config.yaml → DataCollectionAgent → data/raw/collected.parquet
                                  → DataQualityAgent → data/raw/collected_clean.parquet
                                                      → AnnotationAgent → data/labeled/collected_labeled.parquet
                                                                        → [HITL: review_queue.csv] → data/labeled/reviewed.parquet
                                                                                                    → ActiveLearningAgent → models/final_model.pkl
```

## config.yaml

Controls data sources for `DataCollectionAgent`. Edit this file to change the dataset or scraping target before running the pipeline. Supports `hf_dataset` (HuggingFace) and `scrape` source types.

## Key output locations

| Artifact | Path |
|---|---|
| Raw data | `data/raw/collected.parquet` |
| Cleaned data | `data/raw/collected_clean.parquet` |
| Labeled data | `data/labeled/collected_labeled.parquet` |
| HITL review queue | `review_queue.csv` → `review_queue_corrected.csv` |
| Trained model | `models/final_model.pkl` + `models/vectorizer.pkl` |
| Reports & plots | `reports/` |
| Notebooks | `notebooks/eda.ipynb`, `quality_analysis.ipynb`, `annotation_analysis.ipynb`, `al_experiment.ipynb` |

## Claude API usage

Three agents use Anthropic API (model `claude-haiku-*`):
- `AnnotationAgent.auto_label()` — primary usage, batch zero-shot classification
- `DataQualityAgent.llm_explain()` — explains detected issues
- `ActiveLearningAgent.llm_analyze()` — analyzes learning curve results

The API returns JSON wrapped in markdown code blocks — always strip ` ```json ` markers before `json.loads()`.
