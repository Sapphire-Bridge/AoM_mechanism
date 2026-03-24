.PHONY: tables check archival-check mom-paper reviewer-assets reviewer-check reproduction one-result-check one-result-check-gpu

RESULTS_DIR ?= results
TABLES_OUT_DIR ?= tables_out
PYTHON ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)
VENV ?= .venv
ARCHIVAL_CHECK_ARGS ?=
MOM_PAPER_ARGS ?=
REVIEWER_CHECK_ARGS ?=
REVIEWER_ASSETS_ARGS ?=
ONE_RESULT_ARGS ?=
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
DEPS_STAMP := $(VENV)/.deps-installed

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

$(DEPS_STAMP): requirements.txt | $(VENV_PYTHON)
	$(VENV_PIP) install -r requirements.txt
	touch $(DEPS_STAMP)

# MAKE_TABLES_STRICT=1 -> fail on missing inputs (no --skip_missing).
# Default -> skip missing inputs but warn to stderr.
tables:
	python scripts/make_tables.py --results_dir $(RESULTS_DIR) --out_dir $(TABLES_OUT_DIR) $(if $(filter 1,$(MAKE_TABLES_STRICT)),,--skip_missing)

check: $(DEPS_STAMP)
	$(VENV_PYTHON) -m pytest -q
	$(VENV_PYTHON) scripts/check_evidence_contract.py
	$(VENV_PYTHON) scripts/check_evidence_contract_fields.py
	$(VENV_PYTHON) scripts/run_paper.py smoke

archival-check: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/archive_readiness_check.py $(ARCHIVAL_CHECK_ARGS)

mom-paper: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_mom_paper.py $(MOM_PAPER_ARGS)

reviewer-check: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/reviewer_quickcheck.py $(REVIEWER_CHECK_ARGS)

reviewer-assets: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/prepare_reviewer_assets.py $(REVIEWER_ASSETS_ARGS)

reproduction: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_mom_paper.py --local_files_only $(MOM_PAPER_ARGS)


one-result-check: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_one_result_check.py --local_files_only $(ONE_RESULT_ARGS)

one-result-check-gpu: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_one_result_check.py --device auto --require_accelerator --local_files_only $(ONE_RESULT_ARGS)
