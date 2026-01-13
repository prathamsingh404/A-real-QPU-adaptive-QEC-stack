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
