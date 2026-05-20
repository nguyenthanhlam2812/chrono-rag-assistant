# Makefile for running ChronoRAG components

.PHONY: install ingest preprocess label-export train-ml build-index precompute build-timeline evaluate offline-all app test

install:
	pip install -r requirements.txt

ingest:
	python scripts/01_ingest_documents.py

preprocess:
	python scripts/02_preprocess_documents.py

label-export:
	python scripts/03_export_labeling_data.py

train-ml:
	python scripts/04_train_ml_classifier.py

train-dl:
	python scripts/05_train_dl_classifier.py

build-index:
	python scripts/06_build_vector_index.py

precompute:
	python scripts/07_precompute_predictions.py

build-timeline:
	python scripts/08_build_timeline.py

evaluate:
	python scripts/09_evaluate_system.py

offline-all: ingest preprocess build-index precompute evaluate

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/
