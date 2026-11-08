all: main.pdf
main.pdf: main.tex refs.bib sections/*.tex
	pdflatex -interaction=nonstopmode main >/dev/null
	bibtex main >/dev/null || true
	pdflatex -interaction=nonstopmode main >/dev/null
	pdflatex -interaction=nonstopmode main >/dev/null
	@grep -c "LaTeX Warning: Citation" main.log || true
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
.PHONY: all clean
