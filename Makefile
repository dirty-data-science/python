# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        = python3
SPHINXOPTS    =
SPHINXBUILD   = $(PYTHON) -m sphinx

ALLSPHINXOPTS   = -d build/doctrees $(SPHINXOPTS) .


.PHONY: help clean html web pickle htmlhelp latex changes linkcheck zip

all: html-noplot

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  pdf       to make PDF from LaTeX, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  install   to upload to github the web pages"
	@echo "  zip       to create the zip file with examples and doc"

clean:
	-rm -rf build/*
	-rm -rf intro/scipy/auto_examples/ intro/matplotlib/auto_examples/ intro/summary-exercises/auto_examples advanced/mathematical_optimization/auto_examples/ advanced/advanced_numpy/auto_examples/ advanced/image_processing/auto_examples advanced/scipy_sparse/auto_examples packages/3d_plotting/auto_examples packages/statistics/auto_examples/ packages/scikit-image/auto_examples/ packages/scikit-learn/auto_examples intro/numpy/auto_examples guide/auto_examples

test:
	MATPLOTLIBRC=build_tools $(PYTHON) -m pytest --doctest-glob '*.rst' --ignore advanced/advanced_numpy/examples/myobject_test.py --ignore advanced/interfacing_with_c/numpy_c_api/test_cos_module_np.py --ignore advanced/interfacing_with_c/ctypes/cos_module.py --ignore advanced/interfacing_with_c/swig_numpy/test_cos_doubles.py --ignore advanced/interfacing_with_c/cython_numpy/test_cos_doubles.py --ignore advanced/interfacing_with_c/ctypes_numpy/cos_doubles.py --ignore advanced/interfacing_with_c/ctypes_numpy/test_cos_doubles.py --ignore advanced/interfacing_with_c/numpy_shared/test_cos_doubles.py

test-stop-when-failing:
	MATPLOTLIBRC=build_tools $(PYTHON) -m pytest -x --doctest-glob '*.rst' --ignore advanced/advanced_numpy/examples/myobject_test.py --ignore advanced/interfacing_with_c/numpy_c_api/test_cos_module_np.py --ignore advanced/interfacing_with_c/ctypes/cos_module.py --ignore advanced/interfacing_with_c/swig_numpy/test_cos_doubles.py --ignore advanced/interfacing_with_c/cython_numpy/test_cos_doubles.py --ignore advanced/interfacing_with_c/ctypes_numpy/cos_doubles.py --ignore advanced/interfacing_with_c/ctypes_numpy/test_cos_doubles.py --ignore advanced/interfacing_with_c/numpy_shared/test_cos_doubles.py

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

html:
	mkdir -p build/html build/doctrees
	# This line makes the build a bit more lengthy, and the
	# the embedding of images more robust
	rm -rf build/html/_images
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

cleandoctrees:
	rm -rf build/doctrees

pickle:
	mkdir -p build/pickle build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web build/pickle"
	@echo "to start the sphinx-web server."

web: pickle

linkcheck:
	mkdir -p build/linkcheck build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

