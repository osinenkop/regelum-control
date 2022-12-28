##############
# READ OR DIE# ============>     pip install sphinx==4.0 -U
############## ============>     conda install pandoc # or you might instead just use `apt install pandoc`, depending on your environment
# if you install sphinx==3, it will not work. if you install sphinx==5 it will not work in the intended way.


# Before running make, you need to make sure that all dependencies are installed and sphinx
# your sphinx is fresh (i.e somewhere around 5.3.0). On the website (of sphinx) it tells you to
# use `apt install python3-sphinx`, but THIS IS A TRAP. If you do this you'll just install an
# outdated version of sphinx. You probably want to `pip install -U sphinx` instead. Also be sure
# to also install the theme via  `pip install sphinx_rtd_theme`.


# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build


.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).


github:
	@make clean
	@rm -rf ./modules/*
	@python3 ./strip_readme.py
	@make html
	@rm -rf ../docs/*
	@cp -a build/html/. ../docs
	@rm -f ./README.rst
	@rm -f ../README.md
	@mv ./README.md ../README.md


help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)



%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


