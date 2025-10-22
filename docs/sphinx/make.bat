@echo off
setlocal
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %*
echo Build finished. The HTML pages are in %BUILDDIR%\html
