# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python3 -msphinx
SPHINXPROJ    = PennyLane
SOURCEDIR     = .
BUILDDIR      = _build
PUMLS         = $(wildcard ./dev/arch/*puml)
DIAGRAMS      = $(subst dev/arch,_static/arch,$(patsubst %puml,%svg,$(PUMLS)))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean diagrams

diagrams: $(DIAGRAMS)
$(DIAGRAMS): _static/arch/%svg: dev/arch/%puml dev/arch/common.iuml
	plantuml -tsvg -o $(shell cd $(@D); pwd) $<

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
.DEFAULT: Makefile
	# $(MAKE) $(DIAGRAMS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf doxyoutput/ api/ _build/
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
