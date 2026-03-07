.PHONY: tables archival-check

RESULTS_DIR ?= results
TABLES_OUT_DIR ?= tables_out

# MAKE_TABLES_STRICT=1 -> fail on missing inputs (no --skip_missing).
# Default -> skip missing inputs but warn to stderr.
tables:
	python scripts/make_tables.py --results_dir $(RESULTS_DIR) --out_dir $(TABLES_OUT_DIR) $(if $(filter 1,$(MAKE_TABLES_STRICT)),,--skip_missing)

archival-check:
	python scripts/archive_readiness_check.py
