@echo off
setlocal
rem Use python -m sphinx to avoid relying on sphinx-build being on PATH
set SPHINXBUILD=python -m sphinx
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %*
echo Build finished. The HTML pages are in %BUILDDIR%\html
