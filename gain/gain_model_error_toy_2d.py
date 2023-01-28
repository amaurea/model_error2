# This program simulates large-scale power loss from the interaction of
# detector-relative gain errors with large-scale correlated noise.
# We will simulate a crosslinked scanning pattern where each detector
# covers one row/col of pixels. Each sample hits a pixel center, so we
# won't be testing subpixel model errors here.

import numpy as np, scipy, time
import cg
from pixell import utils, mpi

def fmul(f,x):
	if np.asarray(f).size == 1: return f*x
	else: return np.fft.irfft2(np.fft.rfft2(x)*f,s=x.shape).real

def calc_l(shape):
	ly = np.fft.fftfreq(shape[-2])[:,None]
	lx = np.fft.fftfreq(shape[-1])[None,:]
	l  = (ly**2+lx**2)**0.5
	return l

def demean(x): return x-np.mean(x)

# We will simulate a rectangular detector array which scans
# across a rectangle of pixels with the given shape, first
# rowwise and then colwise, in such a way that all detectors
# hit all rows and cols.
seed   = 0
ashape = (2,2)
ndet   = ashape[0]*ashape[1]
mshape = (200,200)
npix   = mshape[0]*mshape[1]
nsim   = 400
comm   = mpi.COMM_WORLD
sigma_gain = 0.1

# Define the rowwise scanning pattern for the (0,0)-detector
pix_pat1 = np.mgrid[:mshape[0],:mshape[1]].reshape(2,-1) # [{y,x},npix]
# The other detectors have the same pattern, just with a pixel
# offset equal to their detector offset
detoffs  = np.mgrid[:ashape[0],:ashape[1]].reshape(2,-1)
pix_pat1 = pix_pat1[:,None,:] + detoffs[:,:,None] # [{y,x},ndet,npix]
# Wrap around detectors that go off the map
pix_pat1[0] %= mshape[0]
pix_pat1[1] %= mshape[1]
# The columnwise scanning pattern
pix_pat2 = pix_pat1[::-1]
# The full pattern is just the first pattern followed by the second
pix   = np.concatenate([pix_pat1,pix_pat2],-1) # [{y,x,ndet,nsamp]
nsamp = pix.shape[-1]
# Build our pointing matrix. This translates from between TOD [ndet*nsamp]
# to flattened pixels [ny*nx], and so has logical shape [ndet*nsamp,ny*nx].
y, x = pix.reshape(2,-1)
P    = scipy.sparse.csr_array((np.full(ndet*nsamp,1),(np.arange(ndet*nsamp),y*mshape[1]+x)),shape=(ndet*nsamp,npix))

# Build the raw (pre-gain-error) inverse noise matrix.
# Noise will consist a detector-uncorrelated white noise floor
# with stddev 1, and a common-mode 1/f spectrum with stddev of 1 at fknee.
# N  = W + VEV', V = [1 1 1 1 1...]'
# iN = N" = W" - W"V(E" + V'W"V)"V'W" (woodbury)
# diagonal in fourier-direction, correlated in detector-direction. So can
# evaluate this independently per fourier-mode.
# V'W"V = sum(W")
freq  = np.fft.rfftfreq(nsamp)
# Per-scan single-detecotr noise matrix
# We want the noise to be pretty correlated.
# Let's make the fknee freq 1/50th of the horizontal side length
fknee = 0.5*50/mshape[1]
iE  = np.clip((np.maximum(freq,freq[1]/2)/fknee)**3.5, 1e-8, 1e8)
iW  = np.ones((ndet,len(freq)))
# Build matrix that represents noise matrix
class NmatCommonGainerr:
	def __init__(self, iW, iE, gainerr=None):
		self.iW = iW # [ndet,nfreq]
		self.iE = iE # [nfreq]
		self.gainerr = gainerr if gainerr is not None else np.full(len(iW),1.0)
		self.kernel = 1/(iE + np.sum(iW,0))
	def apply(self, tod):
		ftod = np.fft.rfft(tod) / self.gainerr[:,None]
		ftod = self.iW * ftod - self.iW*self.kernel*np.sum(self.iW*ftod,0)
		tod  = np.fft.irfft(ftod, n=tod.shape[-1])
		tod /= self.gainerr[:,None]
		return tod
	def sim(self, nsamp, nogainerr=False):
		ndet    = self.iW.shape[0]
		nwhite  = np.fft.irfft(np.fft.rfft(np.random.randn(ndet,nsamp)) * self.iW**-0.5, n=nsamp)
		ncommon = np.fft.irfft(np.fft.rfft(np.random.randn(nsamp)) * self.iE**-0.5,      n=nsamp)
		tod = nwhite + ncommon
		if not nogainerr:
			tod *= self.gainerr[:,None]
		return tod

# Version of noise model that hasn't been polluted by gain errros
iN_true = NmatCommonGainerr(iW, iE)

# Build the signal. It will be a simple 1/l**2 spectrum. We simulate this directly
# on the grid of the actual samples
l      = calc_l(mshape)
lnorm  = 1
C      = (np.maximum(l,l[0,1]/2)/lnorm)**-2
# Make band-limited by applying a beam. nscan/nside translates from
# the target pixels to the sample spacing
bsigma = 3
B      = np.exp(-0.5*l**2*bsigma**2)

def sim_signal_map(C, B):
	"""Simulate a signal map with shape mshape and signal-covmat C and beam B"""
	signal_map = np.fft.ifft2(np.fft.fft2(np.random.standard_normal(B.shape))*C**0.5*B).real
	return signal_map

# We need iterative methods to solve these. Scipy's conjugate gradient implementation
# has some issues (though they might not be relevant for this simple case), so let's
# just define our own solver.

def mapmaker_ml(tod, P, iN):
	"""tod[ndet,nsamp] → map[ny,nx]"""
	b = P.T.dot(iN.apply(tod).reshape(-1))
	def A(x): return P.T.dot(iN.apply(P.dot(x).reshape(tod.shape)).reshape(-1))
	solver = cg.CG(A, b)
	while solver.err > 1e-8: solver.step()
	return solver.x.reshape(mshape)
def mapmaker_bin(tod, P):
	return scipy.sparse.linalg.spsolve(P.T.dot(P), P.T.dot(tod.reshape(-1))).reshape(mshape)
def mapmaker_filter_bin(tod, P, F):
	return scipy.sparse.linalg.spsolve(P.T.dot(P), P.T.dot(F.apply(tod).reshape(-1))).reshape(mshape)
# No destriper for now. It's messier with multiple detectors

def make_maps(tod, P, iN): #, iNw, iNc):
	names, maps = [], []
	names.append("binned"); maps.append(mapmaker_bin(tod, P))
	names.append("ml");     maps.append(mapmaker_ml (tod, P, iN))
	# Instead of implementing all the sims in the mapmaker_filter_bin step, we split
	# it into two parts: Just the filter+bin part and a simulation step. With this
	# we can get the tfun as fbinsim/binned. The point of this is to avoid having
	# a sim-loop inside a sim-loop
	names.append("fbin");    maps.append(mapmaker_filter_bin(tod, P, iN))
	for cap in range(1,7):
		names.append("ml_cap_%d" % cap)
		iN_capped = NmatCommonGainerr(iN.iW/10**cap, iN.iE, iN.gainerr)
		maps.append(mapmaker_ml(tod, P, iN_capped))
	#for blen in [4,16,64]:
	#	names.append("destripe_prior_%03d" % blen)
	#	print(names[-1])
	#	maps.append(mapmaker_destripe(tod, P, iNw=iNw, iNc=iNc, blen=blen))
	#for blen in [4,16,64]:
	#	names.append("destripe_plain_%03d" % blen)
	#	print(names[-1])
	#	maps.append(mapmaker_destripe(tod, P, iNw=iNw, iNc=0,   blen=blen))
	maps = np.array(maps)
	return names, maps

def radial_bin(ps2d, bscale=1):
	shape = ps2d.shape
	ps2d  = ps2d.reshape(-1,shape[-2],shape[-1])
	ls    = calc_l(shape)
	bsize = ls[0,1]*bscale
	lpix  = (ls.reshape(-1)/bsize).astype(int)
	ps1d  = np.array([np.bincount(lpix, p.reshape(-1)) for p in ps2d])
	ls1d  = np.bincount(lpix, ls.reshape(-1))
	hits  = np.bincount(lpix)
	ps1d /= hits
	ls1d /= hits
	return ps1d.reshape(shape[:-2]+ps1d.shape[1:]), ls1d

def calc_ps2d(map):
	return np.abs(np.fft.fft2(map))**2 / (map.shape[-2]*map.shape[-1])

def calc_spectrum(map, bscale=1):
	ps2d  = calc_ps2d(map)
	return radial_bin(ps2d, bscale=bscale)

sspecs = 0
nspecs = 0
nok    = 0
for si in range(comm.rank, nsim, comm.size):
	print("sim %3d" % (si))
	np.random.seed(seed*nsim+si)
	# Draw the random gain errors. Constant per detector. Log-normal with sigma_gain stddev
	gainerr = np.exp(np.random.randn(ndet)*sigma_gain)
	# Set up the gain-erred noise model
	iN = NmatCommonGainerr(iW, iE, gainerr)
	# Draw signal and noise
	signal_map = sim_signal_map(C, B)
	signal_true= P.dot(signal_map.reshape(-1)).reshape(ndet,nsamp)
	noise_true = iN_true.sim(nsamp)
	# Apply gain errors
	signal  = signal_true * gainerr[:,None]
	noise   = noise_true  * gainerr[:,None]
	# Can do these separately because of linearity
	names, smaps = make_maps(signal, P, iN) #, iNw, iNc)
	names, nmaps = make_maps(noise,  P, iN) #, iNw, iNc)
	# Filter+bin ideal case. This uses the true signal, but still uses the gain-affected noise
	# model, as ultimately the weighting will be measured from the data (how else would one know
	# which detectors are more or less sensitive?)
	names.append("fbinsim")
	smaps = np.concatenate([smaps, [mapmaker_filter_bin(signal_true, P, iN)]])
	nmaps = np.concatenate([nmaps, [mapmaker_filter_bin(noise_true,  P, iN)]])
	my_sspecs, ls1d = calc_spectrum(smaps)
	my_nspecs, ls1d = calc_spectrum(nmaps)
	sspecs += my_sspecs
	nspecs += my_nspecs
	nok    += 1
	if si == 0: # Only first sim map as example
		np.save("gain_toy2d_input_signal_map.npy", demean(signal_map))
		for ni, name in enumerate(names):
			np.save("gain_toy2d_%s_signal_map.npy" % (name), demean(smaps[ni]))
			np.save("gain_toy2d_%s_noise_map.npy"  % (name), demean(nmaps[ni]))
			np.save("gain_toy2d_%s_data_map.npy"   % (name), demean(smaps[ni]+nmaps[ni]))
			## Evaluate model
			#model = P.dot(smaps[ni].reshape(-1))
			#np.savetxt("toy2d_%s_signal_model.txt"   % (name), np.array([signal, model]).T, fmt="%15.7e")
sspecs = utils.allreduce(sspecs, comm)
nspecs = utils.allreduce(nspecs, comm)
nok    = comm.allreduce(nok)
sspecs /= nok
nspecs /= nok
# Output mean signal and noise spectra
if comm.rank == 0:
	for ni, name in enumerate(names):
		np.savetxt("gain_toy2d_%s_signal_ps.txt" % (name), np.array([ls1d, sspecs[ni]]).T, fmt="%15.7e")
		np.savetxt("gain_toy2d_%s_noise_ps.txt"  % (name), np.array([ls1d, nspecs[ni]]).T, fmt="%15.7e")

# Output theoretical spectrum on same ls as our measurements
if comm.rank == 0:
	np.savetxt("gain_toy2d_theory_ps.txt", np.array([ls1d, (ls1d/lnorm)**-2*np.exp(-ls1d**2*bsigma**2)]).T, fmt="%15.7e")
