import numpy as np

def calc_bias(ftheory, fsignal, fnoise, obias, onoise):
	theory = np.loadtxt(ftheory).T
	signal = np.loadtxt(fsignal).T
	noise  = np.loadtxt(fnoise).T
	# Compute the retained power fraction
	ratio   = signal.copy()
	ratio[1] /= theory[1]
	# Debias noise spectrum
	noise[1] /= ratio[1]
	# And output
	np.savetxt(obias,  ratio.T, fmt="%15.7e")
	np.savetxt(onoise, noise.T, fmt="%15.7e")

def calc_bias_fb(ftheory, fsignal, ftheory_ideal, fsignal_ideal, fnoise, obias, onoise):
	theory = np.loadtxt(ftheory).T
	signal = np.loadtxt(fsignal).T
	theory_ideal = np.loadtxt(ftheory_ideal).T
	signal_ideal = np.loadtxt(fsignal_ideal).T
	noise  = np.loadtxt(fnoise).T
	# First calc the ideal ratio and debias using it
	ratio_ideal     = signal_ideal.copy()
	ratio_ideal[1] /= theory_ideal[1]
	signal[1] /= ratio_ideal[1]
	noise [1] /= ratio_ideal[1]
	# Then compute the remaining bias
	ratio     = signal.copy()
	ratio[1] /= theory[1]
	# Debias noise spectrum
	noise[1] /= ratio[1]
	# And output
	np.savetxt(obias,  ratio.T, fmt="%15.7e")
	np.savetxt(onoise, noise.T, fmt="%15.7e")

for case in ["binned", "ml"]+["ml_cap_%d" % i for i in range(1,7)]:
	calc_bias("gain_toy2d_binned_signal_ps.txt", "gain_toy2d_%s_signal_ps.txt" % case, "gain_toy2d_%s_noise_ps.txt" % case, "gain_toy2d_%s_bias_ps.txt" % case, "gain_toy2d_%s_noise_debiased_ps.txt" % case)
calc_bias_fb("gain_toy2d_binned_signal_ps.txt", "gain_toy2d_fbin_signal_ps.txt", "gain_toy2d_theory_ps.txt", "gain_toy2d_fbinsim_signal_ps.txt", "gain_toy2d_fbin_noise_ps.txt", "gain_toy2d_fbin_bias_ps.txt", "gain_toy2d_fbin_noise_debiased_ps.txt")
