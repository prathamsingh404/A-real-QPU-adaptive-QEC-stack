.PHONY: reproduce test claims-check smoke paper
reproduce:
	python scripts/reproduce.py

claims-check:
	python scripts/make_claims.py --check
	python scripts/verify_lock.py

test:
	python -m pytest -q

smoke:
	python experiments/run_suite.py --pilot --output scratch/pilot.json --manifest scratch/pilot-manifest.json

paper:
	python -m adaptive_qec.analysis.make_paper_tables
	python -m adaptive_qec.analysis.make_paper_figures
	tectonic papers/main.tex
