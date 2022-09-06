import numpy as np

# Some helper functions

def rfftfreq(n, d=1.0): return np.arange(n//2+1)/(n*d)
def unitvec(n, i):
	res = np.zeros(n)
	res[i] = 1
	return res

def func_to_mat(func, n):
	return np.array([func(unitvec(n,i)) for i in range(n)]).T

def dewin(x):
	f  = np.fft.rfftfreq(len(x))
	return np.fft.irfft(np.fft.rfft(x)/np.sinc(f),n=len(x)).real

def wrap(x): return np.concatenate([x,x[:1]])

npix  = 10
nsamp = 100

# Pointing matrix
pix = np.arange(nsamp).astype(float)*npix/nsamp
P   = np.zeros((nsamp,npix))
for i, p in enumerate(pix):
	P[i,int(np.round(pix[i]))%npix] = 1

# Filter/Noise model
freq = rfftfreq(nsamp)
nfreq= len(freq)
fknee= 0.03
# the max stuff just avoids making the mean infinitely uncertain
ips  = 1/(1+(np.maximum(freq,freq[1]/2)/fknee)**-3.5)
def filter_func(x):
	return np.fft.irfft(ips*np.fft.rfft(x), n=len(x))
F = func_to_mat(filter_func, nsamp)

# Signal (noise-free)
def build_signal(pix): return np.sin(2*np.pi*pix/npix)
signal = build_signal(pix)

# Plain binned map
map_binned = np.linalg.solve((P.T.dot(P)), P.T.dot(signal))

# Maximum-likelihood map
map_ml = np.linalg.solve((P.T.dot(F).dot(P)),P.T.dot(F.dot(signal)))

# Filter+bin with observation matrix
map_fb = np.linalg.solve(P.T.dot(P), P.T.dot(F).dot(signal))
obsmat = np.linalg.inv(P.T.dot(P)).dot(P.T.dot(F).dot(P))
map_fb_deobs = np.linalg.solve(obsmat, map_fb)

# Filter-bin with transfun
nsim = 1000
sim_ips = np.zeros(npix//2+1)
sim_ops = np.zeros(npix//2+1)
for i in range(nsim):
	sim_imap = np.random.standard_normal(npix)
	sim_omap = np.linalg.solve(P.T.dot(P), P.T.dot(F).dot(P).dot(sim_imap))
	sim_ips += np.abs(np.fft.rfft(sim_imap))**2
	sim_ops += np.abs(np.fft.rfft(sim_omap))**2
tf = (sim_ops/sim_ips)**0.5
map_fb_detrans = np.fft.irfft(np.fft.rfft(map_fb)/tf, n=npix)

# Generalized destriping with minimum segment length
I   = np.eye(nsamp)
iCa = np.linalg.inv(np.linalg.inv(F) - I)
Z   = I-P.dot(np.linalg.solve(P.T.dot(P), P.T))
a   = np.linalg.solve(Z+iCa, Z.dot(signal))
map_ds = np.linalg.solve(P.T.dot(P), P.T.dot(signal - a))

# Finally, let's see what happens when the signal actually follows the model
signal2         = P.dot(map_binned)
map_binned2     = np.linalg.solve(P.T.dot(P), P.T.dot(signal2))
map_ml2         = np.linalg.solve((P.T.dot(F).dot(P)),P.T.dot(F.dot(signal2)))
map_fb2         = np.linalg.solve(P.T.dot(P), P.T.dot(F).dot(signal2))
map_fb_deobs2   = np.linalg.solve(obsmat, map_fb2)
map_fb_detrans2 = np.fft.irfft(np.fft.rfft(map_fb2)/tf, n=npix)
a2 = np.linalg.solve(Z+iCa, Z.dot(signal2))
map_ds2 = np.linalg.solve(P.T.dot(P), P.T.dot(signal2 - a2))

# Output pixel centers
opix= np.arange(npix+1)

np.savetxt("signal.txt", np.array([pix, signal, signal2]).T, fmt="%15.7e")
np.savetxt("maps.txt",   np.array([
	opix,
	wrap(dewin(map_binned)),
	wrap(dewin(map_ml)),
	wrap(dewin(map_fb_deobs)),
	wrap(dewin(map_fb_detrans)),
	wrap(dewin(map_ds)),
	# No pixwin deconv needed because the signal itself is pixelated
	wrap(map_binned2),
	wrap(map_ml2),
	wrap(map_fb_deobs2),
	wrap(map_fb_detrans2),
	wrap(map_ds2),
]).T, fmt="%15.7e")
np.savetxt("ps.txt", np.array([freq, 1/ips]).T, fmt="%15.7e")


# Let's also demonstrate why this bias happens, by comparing 3 cases:
# 1. The plain binned model, which is bias-free and matches what we would expect to get
# 2. A null model, with just zero
# 3. The ML solution
# For each of these we will show the signal, model and residual in real space, fourier
# space and chi(f) to demonstrate that the ML solution really is plausibly the best
# fit, given the model's limitations.

model_plain = P.dot(map_binned)
model_zero  = np.zeros(nsamp)
model_ml    = P.dot(map_ml)

models      = np.array([model_plain,model_zero,model_ml])
resids      = signal-models
resid_ps    = np.abs(np.fft.rfft(resids))**2
resid_chisq = resid_ps*ips
cum_ps      = np.cumsum(resid_ps[:,::-1],1)[:,::-1]
cum_chisq   = np.cumsum(resid_chisq[:,::-1],1)[:,::-1]

np.savetxt("resids.txt", np.concatenate([pix[None],signal[None],models,resids],0).T, fmt="%15.7e")
np.savetxt("chisqs.txt", np.concatenate([freq[None],resid_ps,resid_chisq,cum_ps,cum_chisq]).T, fmt="%15.7e")
