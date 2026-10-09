PY ?= python3
NB := notebooks/lab1_rlc_frequency.ipynb

.PHONY: all install data test notebook slides clean

all: data test notebook slides

install:
	$(PY) -m pip install -e ".[cma,test,notebook]"

data:
	$(PY) -c "from rlcfreq import data; data.fetch_all(verbose=True)"

test:
	$(PY) -m pytest -q

notebook:
	@if $(PY) -c "import nbconvert, ipykernel" 2>/dev/null; then \
		$(PY) -m jupyter nbconvert --to notebook --execute --inplace $(NB) --ExecutePreprocessor.timeout=3600; \
	else \
		$(PY) tools/execute_notebook.py $(NB); \
	fi

slides:
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error slides.tex && latexmk -c

clean:
	rm -rf .pytest_cache **/__pycache__ slides/*.aux slides/*.log slides/*.nav slides/*.out slides/*.snm slides/*.toc slides/*.fls slides/*.fdb_latexmk slides/*.xdv
