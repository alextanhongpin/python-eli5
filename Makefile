# cp ../Makefile Makefile
#import sys

#!"{sys.executable}" --version
#!which "{sys.executable}"
#!jupyter labextension list

-include .env
export

jupyter: 
	@uv run jupyter lab


install:
	@uv add jupyterlab-vim jupyterlab-code-formatter ipywidgets black isort
	@uv add numpy==1.23.0 matplotlib seaborn scikit-learn==1.2.0
	#@uv run jupyter labextension install @jupyter-widgets/jupyterlab-manager
	#@uv run jupyter labextension install jupyter-matplotlib


preview:
	uv run quarto preview

render: create_docs_dir
	uv run quarto render

# Similar to convert, but only convert the diff files.
# You need to 'git add .' the files before running this command.
render_diff: create_docs_dir
	$(foreach file, $(shell git diff HEAD --name-only | grep .ipynb), uv run quarto render $(file) --to gfm --output-dir docs/)


create_docs_dir:
	mkdir -p docs


lint:
	@uv run black *.py

# uv run quarto create project
# uv init --python 3.10 
#
define quarto_config
# uv run quarto create project
#
# https://quarto.org/docs/reference/formats/markdown/gfm.html
project:
  title: "python applied machine learning"
  output-dir: docs
  render:
    - "**/*.ipynb"
    - "!source" # Exclude source

code-fold: false
format: gfm # GitHub Flavored Markdown
number-sections: true
toc: true
endef

init_quarto:
	@echo "$$quarto_config" > _quarto.yaml
