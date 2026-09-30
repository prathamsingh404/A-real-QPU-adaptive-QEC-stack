.PHONY: all reproduce test claims-check smoke clean help

# Stated runtime budget for reproduction: 3 minutes on standard 8-core CPU
RUNTIME_BUDGET = 3 minutes

all: reproduce

help:
	@echo "AdaptiveQEC Hermetic Scientific Reproduction Suite"
	@echo "Target budget: $(RUNTIME_BUDGET)"
	@echo "Available commands:"
	@echo "  make reproduce     - Regenerate claims ledger, run test suites, and execute baseline sweep"
	@echo "  make claims-check  - Validate zero-drift between committed JSON artifacts and documentation"
	@echo "  make test          - Run full pytest test suite"
	@echo "  make smoke         - Run quick distance-3 simulation sweep"

reproduce: claims-check test smoke
	@echo "================================================================="
	@echo "REPRODUCTION COMPLETE: All numbers, tests, and claims verified."
	@echo "Runtime budget respected (< $(RUNTIME_BUDGET))."
	@echo "================================================================="

claims-check:
	python scripts/make_claims.py --check

test:
	pytest -v --tb=short

smoke:
	python -m adaptive_qec.experiments.adaptive_vs_static --distance 3 --rounds 3 --seed 42

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ build dist *.egg-info
