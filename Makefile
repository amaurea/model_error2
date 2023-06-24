figs = subpix/model_error_toy.pdf subpix/model_resids.pdf subpix/model_error_toy_noerr.pdf  subpix/model_resid_cumps.pdf subpix/ps.pdf common_mode/common.pdf nearest_neigh/path.pdf nearest_neigh/vals.pdf subpix/model_error_toy_2d.pdf subpix/model_error_toy_2d_noise.pdf subpix/tfun_lin_1d.pdf pixwin/linear_defs.pdf pixwin/linear_p.pdf pixwin/pixwins.pdf gain/gain_toy1d_bias.pdf gain/gain_toy2d_noise.pdf gain/gain_toy2d_bias.pdf
figs_aa = subpix/model_error_toy_aa.pdf subpix/model_resids_aa.pdf subpix/model_error_toy_noerr_aa.pdf subpix/model_resid_cumps_aa.pdf subpix/ps_aa.pdf common_mode/common_aa.pdf nearest_neigh/path.pdf nearest_neigh/vals.pdf subpix/model_error_toy_2d_aa.pdf subpix/model_error_toy_2d_noise_aa.pdf subpix/tfun_lin_1d_aa.pdf pixwin/linear_defs_aa.pdf pixwin/linear_p_aa.pdf pixwin/pixwins_aa.pdf gain/gain_toy1d_bias_aa.pdf gain/gain_toy2d_noise_aa.pdf gain/gain_toy2d_bias_aa.pdf

ps_files := $(wildcard subpix/toy2d*ps.txt)

main.pdf: main.tex refs.bib $(figs) FORCE
	pdflatex -interaction=batchmode main.tex && bibtex main && pdflatex -interaction=batchmode main.tex && pdflatex -interaction=batchmode main.tex
	#pdflatex main.tex

main_aa.pdf: main_aa.tex refs.bib $(figs_aa) FORCE
	pdflatex -interaction=batchmode main_aa.tex && bibtex main_aa && pdflatex -interaction=batchmode main_aa.tex && pdflatex -interaction=batchmode main_aa.tex

figs: $(figs)

.subpix_run: subpix/model_error_toy.py
	cd subpix; python model_error_toy.py
	touch .subpix_run
subpix/signal.txt subpix/maps.txt subpix/resids.txt subpix/ps.txt: .subpix_run
.subpix_plot: subpix/plot.gpi subpix/signal.txt subpix/maps.txt subpix/resids.txt subpix/ps.txt
	cd subpix; gnuplot plot.gpi
	touch .subpix_plot
subpix/model_error_toy.svg subpix/model_resid.svg subpix/model_resids.svg subpix/model_error_toy_noerr.svg subpix/model_resid_cumps.svg subpix/ps.svg: .subpix_plot

.subpix_plot_2d: subpix/plot_2d.gpi $(ps_files)
	cd subpix; gnuplot plot_2d.gpi
	touch .subpix_plot_2d
subpix/model_error_toy_2d.svg subpix/model_error_toy_2d_noise.svg subpix/tfun_lin_1d.svg: .subpix_plot_2d
	touch $@
.common_run: common_mode/common_mode_test.py
	cd common_mode; python common_mode_test.py
	touch .common_run
common_mode/long.txt common_mode/short.txt: .common_run
.common_plot: common_mode/plot.gpi common_mode/long.txt common_mode/short.txt
	cd common_mode; gnuplot plot.gpi
	touch .common_plot
common_mode/common.svg: .common_plot

nearest_neigh/vals.txt: nearest_neigh/nearest_neigh.py
	cd nearest_neigh; python nearest_neigh.py
nearest_neigh/%.svg: nearest_neigh/plot_%.gpi nearest_neigh/vals.txt
	cd nearest_neigh; gnuplot plot_$*.gpi

%.pdf: %.svg
	inkscape $^ -o $@
%.png: %.svg
	convert -density 150 $^ $@

.PHONY: FORCE figs clean cleanall
.SUFFIXES:

clean:
	rm -f main.aux main.bbl main.blg main.log main.out
cleanall: clean
	rm -f {subpix,common_mode,nearest_neigh}/*.{txt,svg,pdf} .{subpix,common}_{run,plot}
cleanimg: clean
	rm -f {subpix,common_mode,nearest_neigh}/*.{svg,pdf} .{subpix,common}_{run,plot}
