# Makefile for Sphinx documentation

SPHINXBUILD   = python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@echo "Makefile for Sphinx docs"
	@echo "  make html    to build HTML documentation"

clean:
	@rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	echo ""
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html"