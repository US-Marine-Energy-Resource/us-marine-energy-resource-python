EXAMPLE_PY   := examples/tidal_site_assessment.py
EXAMPLE_NB   := examples/tidal_site_assessment.ipynb
EXAMPLE_EXEC := examples/tidal_site_assessment_executed.ipynb

.PHONY: all notebook execute clean lint test check-deps readme

all: notebook

# ---------------------------------------------------------------------------
# README (generated from README.qmd via Quarto)
# ---------------------------------------------------------------------------

## readme: render README.qmd → README.md
readme:
	quarto render README.qmd --to gfm --output README.md
	@echo "README.md updated."

# ---------------------------------------------------------------------------
# Notebook conversion
# ---------------------------------------------------------------------------

## notebook: convert the percent-format Python script to .ipynb
notebook: $(EXAMPLE_NB)

$(EXAMPLE_NB): $(EXAMPLE_PY)
	jupytext --to notebook $(EXAMPLE_PY) -o $(EXAMPLE_NB)

## execute: run the notebook end-to-end (requires S3 access + dev deps)
execute: $(EXAMPLE_NB)
	jupyter nbconvert \
		--to notebook \
		--execute $(EXAMPLE_NB) \
		--output $(EXAMPLE_EXEC) \
		--ExecutePreprocessor.timeout=1200

## clean: remove generated notebook artefacts
clean:
	rm -f $(EXAMPLE_NB) $(EXAMPLE_EXEC)

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

## lint: run ruff linter + formatter check + pyright type checker
# TODO(lint-relax): add examples back to the ruff lines once its files are committed.
lint:
	ruff check us_marine_energy_resource
	ruff format --check us_marine_energy_resource
	pyright
	# Docstring section completeness. Scoped to the modules already brought up
	# to the standard. Widen the path as more modules follow.
	numpydoc lint $(shell find us_marine_energy_resource/wave_hindcast us_marine_energy_resource/explore -name '*.py')

## test: run the full pytest suite
test:
	pytest tests/ -v

## check-deps: verify all notebook tools are installed
check-deps:
	@python -c "import jupytext" 2>/dev/null || \
		(echo "ERROR: jupytext not installed. Run: pip install -e '.[dev]'" && exit 1)
	@python -c "import folium"   2>/dev/null || \
		(echo "ERROR: folium not installed.   Run: pip install -e '.[dev]'" && exit 1)
	@python -c "import notebook" 2>/dev/null || \
		(echo "ERROR: notebook not installed. Run: pip install -e '.[dev]'" && exit 1)
	@echo "All notebook dependencies found."
