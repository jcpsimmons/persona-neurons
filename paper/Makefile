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
arxiv: main.pdf
	rm -rf arxiv && mkdir -p arxiv/sections arxiv/figures
	cp main.tex main.bbl arxiv/
	cp sections/*.tex arxiv/sections/
	cp figures/*.pdf arxiv/figures/
	cd arxiv && tar czf ../persona-neurons-arxiv.tar.gz . && cd .. && rm -rf arxiv
	@echo "wrote persona-neurons-arxiv.tar.gz"; tar tzf persona-neurons-arxiv.tar.gz | wc -l
