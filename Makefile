VENV     = .venv
PYTHON   = $(VENV)/bin/python
PYTEST   = $(VENV)/bin/pytest
UVICORN  = $(VENV)/bin/uvicorn
PORT     = 8000

.PHONY: help install test pytest serve run

# ─── Default ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Agente Autónomo de Comparación de Contratos"
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  Targets:"
	@echo "    install   Install all dependencies from requirements.txt"
	@echo "    test      Run the full pytest suite (alias: pytest)"
	@echo "    pytest    Run the full pytest suite"
	@echo "    serve     Start the FastAPI server on port $(PORT)"
	@echo "    run       Run the CLI pipeline (default test contracts)"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -r requirements.txt

# ─── Tests ────────────────────────────────────────────────────────────────────
test: pytest

pytest:
	$(PYTEST) tests/ -v

# ─── API Server ───────────────────────────────────────────────────────────────
serve:
	$(PYTHON) -m uvicorn src.api:app --reload --port $(PORT)

# ─── CLI ──────────────────────────────────────────────────────────────────────
run:
	$(PYTHON) src/main.py
