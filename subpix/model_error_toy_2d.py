import numpy as np, scipy, time
import cg

def fmul(f,x):
	if np.asarray(f).size == 1: return f*x
	else: return np.fft.irfft(np.fft.rfft(x)*f,n=len(x)).real

def calc_l(n):
	ly = np.fft.fftfreq(n)[:,None]
	lx = np.fft.fftfreq(n)[None,:]
	l  = (ly**2+lx**2)**0.5
	return l

def demean(x): return x-np.mean(x)

seed = 0
# Sill simulate two crosslinked scanning patterns
# covering a square of pixels with shape (nside,nside).
# Each scanning pattern will have nscan equi-spaced rows/columns
# each with length nscan
nside = 100
nscan = nside*4
npix  = nside**2
nsim  = 100

# We will simulate two crosslinked scanning patterns
pix_pat1 = (np.mgrid[:nscan,:nscan]*nside/nscan).reshape(2,-1)
pix_pat2 = pix_pat1[::-1] # swap x and y for other pattern
pix      = np.concatenate([pix_pat1,pix_pat2],1)
nsamp    = pix.shape[1]

# Build a nearest neighbor sparse pointing matrix
iy, ix  = np.floor(pix+0.5).astype(int)%nside
P_nn    = scipy.sparse.csr_array((np.full(nsamp,1),(np.arange(nsamp),iy*nside+ix)),shape=(nsamp,npix))
# Build a bilinear pointing matrix. Here a sample with coordinates
# y,x has value
# (1-ry)*(1-rx)*val[iy,ix] + (1-ry)*rx*val[iy,ix+1] + ry*(1-rx)*val[iy+1,ix] + ry*rx*val[iy+1,ix+1]
# where iy = floor(y) and ry = y-iy, etc.
# We want the pixel centers to be the control points for the interpolation.
# To get this we need floor(pix) instead of floor(pix+0.5)
#pix_left  = np.floor(pix+0.5).astype(int)
pix_left  = np.floor(pix).astype(int)
ry, rx    = pix-pix_left
iy1,ix1   = pix_left % nside
iy2,ix2   = (pix_left+1)% nside
P_lin     = scipy.sparse.csr_array((
	np.concatenate([(1-ry)*(1-rx), (1-ry)*rx, ry*(1-rx), ry*rx]),
	(np.tile(np.arange(nsamp),4),
		np.concatenate([iy1*nside+ix1, iy1*nside+ix2, iy2*nside+ix1, iy2*nside+ix2])
	)), shape=(nsamp,npix))
# load the linear interp tfun
freq_lin_1d, tfun_lin_1d = np.loadtxt("tfun_lin_1d.txt").T

# Build the inverse noise matrix. We want the noise to be pretty correlated, so
# let's have the fknee correspond to 1/30th of a side length. We can't
# afford to store the full thing, so just store the fourier-diagonal
fknee = 0.5*30/nscan
freq  = np.fft.rfftfreq(nsamp)
iN    = 1/(1+(np.maximum(freq,freq[1]/2)/fknee)**-3.5)
# Truncate the power law at a high but not extreme value. This is
# realistic and also avoids convergence problems for the destriper
iN    = np.maximum(iN, np.max(iN)*1e-8)
iNw   = 1
iNw_destripe = iNw * (1+1e-1)
iNc   = 1/(1/iN - 1/iNw_destripe)

# Build the signal. It will be a simple 1/l**2 spectrum. We simulate this directly
# on the grid of the actual samples

l      = calc_l(nscan)*nscan/nside # express in units of output pixels
lnorm  = 1
C      = (np.maximum(l,l[0,1]/2)/lnorm)**-2
# Make band-limited by applying a beam. nscan/nside translates from
# the target pixels to the sample spacing
bsigma = 3
B      = np.exp(-0.5*l**2*bsigma**2)

def sim_signal(C, B, nscan):
	signal_map = np.fft.ifft2(np.fft.fft2(np.random.standard_normal((nscan,nscan)))*C**0.5*B).real
	#print("FIXME 1d")
	#signal_map[:] = np.mean(signal_map,0)[None,:]
	signal = np.concatenate([signal_map.reshape(-1), signal_map.T.reshape(-1)])
	return signal, signal_map
def sim_noise(iN, nsamp):
	noise  = fmul(iN**-0.5, np.random.standard_normal(nsamp))
	#noise -= np.mean(noise)
	return noise

#signal_map = np.fft.ifft2(np.fft.fft2(np.random.standard_normal((nscan,nscan)))*C**0.5*B).real
#signal = np.concatenate([signal_map.reshape(-1), signal_map.T.reshape(-1)])
## Let's make some noise too
#noise  = fmul(iN**-0.5, np.random.standard_normal(nsamp))
#noise -= np.mean(noise)
#
#data   = signal+noise

# We need iterative methods to solve these. Scipy's conjugate gradient implementation
# has some issues (though they might not be relevant for this simple case), so let's
# just define our own solver.

def mapmaker_ml(tod, P, iN):
	b = P.T.dot(fmul(iN,tod))
	def A(x): return P.T.dot(fmul(iN,P.dot(x)))
	solver = cg.CG(A, b)
	while solver.err > 1e-8: solver.step()
	return solver.x.reshape(nside,nside)
def mapmaker_bin(tod, P):
	return scipy.sparse.linalg.spsolve(P.T.dot(P), P.T.dot(tod)).reshape(nside, nside)
#def mapmaker_bin_simple(tod, P):
#	return (P.T.dot(tod)/P.T.dot(P).diagonal()).reshape(nside,nside)
def mapmaker_destripe(tod, P, iNw=1, iNc=0, blen=1):
	# Build baseline projector
	nsamp = tod.size
	nbase = (nsamp+blen-1)//blen
	# Convert iNc to iCa. The difference is that iNc is the sample inverse covariance
	# while iCa is the baseline inverse covariance.
	if np.asarray(iNc).ndim == 0: iCa = iNc
	else: iCa = iNc[:nbase//2+1]
	Q     = scipy.sparse.csr_array((np.full(nsamp,1), (np.arange(nsamp), np.arange(nsamp)//blen)), shape=(nsamp,nbase))
	PNP = P.T.dot(np.atleast_1d(iNw)[:,None]*P)
	def iPNP(x): return scipy.sparse.linalg.spsolve(PNP, x)
	def Z(x): return x-P.dot(iPNP(P.T.dot(iNw*x)))
	b     = Q.T.dot(iNw*Z(tod))
	def A(x): return Q.T.dot(iNw*Z(Q.dot(x))) + fmul(iCa,x)
	solver = cg.CG(A, b)
	while solver.err > 1e-11:
		solver.step()
	a = solver.x
	m = iPNP(P.T.dot(iNw*(tod-Q.dot(a))))
	return m.reshape(nside,nside)

def make_maps(tod, P, iN, iNw, iNc):
	names, maps = [], []
	names.append("binned"); maps.append(mapmaker_bin(tod, P))
	names.append("ml");     maps.append(mapmaker_ml (tod, P, iN))
	#for cap in range(1,7):
	#	names.append("ml_cap_%d" % cap)
	#	maps.append(mapmaker_ml(tod, P, np.maximum(iN, np.min(iN)*10**cap)))
	#for blen in [1,4,16,64]:
	#	names.append("destripe_prior_%03d" % blen)
	#	maps.append(mapmaker_destripe(tod, P, iNw=iNw, iNc=iNc, blen=blen))
	#for blen in [1,4,16,64]:
	#	names.append("destripe_plain_%03d" % blen)
	#	maps.append(mapmaker_destripe(tod, P, iNw=iNw, iNc=0,   blen=blen))
	maps = np.array(maps)
	return names, maps

def radial_bin(ps2d, bscale=1):
	shape = ps2d.shape
	ps2d  = ps2d.reshape(-1,shape[-2],shape[-1])
	ls    = calc_l(shape[-1])
	bsize = ls[0,1]*bscale
	lpix  = (ls.reshape(-1)/bsize).astype(int)
	ps1d  = np.array([np.bincount(lpix, p.reshape(-1)) for p in ps2d])
	ls1d  = np.bincount(lpix, ls.reshape(-1))
	hits  = np.bincount(lpix)
	ps1d /= hits
	ls1d /= hits
	return ps1d.reshape(shape[:-2]+ps1d.shape[1:]), ls1d

def calc_pixwin_raw(n): return 1
def calc_pixwin_nn(n, scale=1):
	s = np.sinc(np.fft.fftfreq(n)/scale)**2
	return s[:,None]*s[None,:]
#def calc_pixwin_lin(n):
#	s = 3/(2+np.cos(2*np.pi*np.fft.fftfreq(n)))
#	return s[:,None]*s[None,:]
#def calc_pixwin_lin(n):
#	return 1
#	q = 2*np.pi*np.fft.fftfreq(n)
#	s = 1/(4/q**2*(1-np.cos(q))**2)
#	s[~np.isfinite(s)] = 1
#	return s[:,None]*s[None,:]
def calc_pixwin_lin(n):
	fout = np.abs(np.fft.fftfreq(n))
	s    = np.interp(fout, freq_lin_1d, tfun_lin_1d)**2
	pixwin = s[:,None]*s[None]
	return pixwin

def calc_ps2d(map):
	return np.abs(np.fft.fft2(map))**2 / (map.shape[-2]*map.shape[-1])

def calc_spectrum(map, pixwin=1, bscale=1):
	ps2d  = calc_ps2d(map)
	ps2d /= pixwin
	return radial_bin(ps2d, bscale=bscale)

pmats   = {"nn":P_nn, "lin":P_lin}
pixwins = {"nn":calc_pixwin_nn(nside)/calc_pixwin_nn(nside,nscan/nside), "lin":calc_pixwin_lin(nside)}
# The pixwin ratio is there to compensate for the limited sub-resolution

#for pname in ["lin"]:
for pname in ["nn", "lin"]:
	np.random.seed(seed)
	P      = pmats[pname]
	pixwin = pixwins[pname]
	sspecs = 0
	nspecs = 0
	nok    = 0
	for si in range(nsim):
		print("P %-4s sim %3d" % (pname, si))
		signal, signal_map = sim_signal(C, B, nscan)
		# Compensate for change in fourier units between high-res and
		# target grids. This step would not be necessary if I were more
		# careful with how the units are defined, but this is good enough
		# for now
		signal     *= nscan/nside
		signal_map *= nscan/nside
		noise = sim_noise(iN, nsamp)
		# Can do these separately because of linearity
		names, smaps = make_maps(signal, P, iN, iNw, iNc)
		names, nmaps = make_maps(noise,  P, iN, iNw, iNc)
		my_sspecs, ls1d = calc_spectrum(smaps, pixwin)
		my_nspecs, ls1d = calc_spectrum(nmaps, pixwin)
		sspecs += my_sspecs
		nspecs += my_nspecs
		nok    += 1
		if si == 0: # Only first sim map as example
			np.save("toy2d_input_signal_map.npy", demean(signal_map))
			for ni, name in enumerate(names):
				np.save("toy2d_%s_%s_signal_map.npy" % (name, pname), demean(smaps[ni]))
				np.save("toy2d_%s_%s_noise_map.npy"  % (name, pname), demean(nmaps[ni]))
				np.save("toy2d_%s_%s_data_map.npy"   % (name, pname), demean(smaps[ni]+nmaps[ni]))
				## Evaluate model
				#model = P.dot(smaps[ni].reshape(-1))
				#np.savetxt("toy2d_%s_%s_signal_model.txt"   % (name, pname), np.array([signal, model]).T, fmt="%15.7e")
	sspecs /= nok
	nspecs /= nok
	# Output mean signal and noise spectra
	for ni, name in enumerate(names):
		np.savetxt("toy2d_%s_%s_signal_ps.txt" % (name, pname), np.array([ls1d, sspecs[ni]]).T, fmt="%15.7e")
		np.savetxt("toy2d_%s_%s_noise_ps.txt"  % (name, pname), np.array([ls1d, nspecs[ni]]).T, fmt="%15.7e")

# Output theoretical spectrum on same ls as our measurements
np.savetxt("toy2d_theory_ps.txt", np.array([ls1d, (ls1d/lnorm)**-2*np.exp(-ls1d**2*bsigma**2)]).T, fmt="%15.7e")


#r = 10
## Make destriped map
#from pixell import enplot
##map_destripe = mapmaker_destripe(data, P_near, iNw=iNw_destripe, iNc=iNc, blen=1)
##enplot.write("test_destripe_mllim", enplot.plot(map_destripe, grid=False, upgrade=4))
##map_destripe_lin = mapmaker_destripe(data, P_lin, iNw=iNw_destripe, iNc=iNc, blen=1)
##enplot.write("test_destripe_mllim_lin", enplot.plot(map_destripe_lin, grid=False, upgrade=4))
##map_destripe = mapmaker_destripe(data, P_near, iNw=iNw_destripe, blen=1)
##enplot.write("test_destripe_binlim", enplot.plot(map_destripe, grid=False, upgrade=4))
##map_destripe_lin = mapmaker_destripe(data, P_lin, iNw=iNw_destripe, blen=1)
##enplot.write("test_destripe_binlim_lin", enplot.plot(map_destripe_lin, grid=False, upgrade=4))
#map_destripe = mapmaker_destripe(data, P_near, iNw=iNw_destripe, iNc=iNc, blen=1)
#enplot.write("test_destripe_pri1", enplot.plot(map_destripe, grid=False, upgrade=4, range=r))
#map_destripe_lin = mapmaker_destripe(data, P_lin, iNw=iNw_destripe, iNc=iNc, blen=1)
#enplot.write("test_destripe_pri1_lin", enplot.plot(map_destripe_lin, grid=False, upgrade=4, range=r))
#
#map_binned     = mapmaker_bin_near(data, P_near)
#map_binned_lin = mapmaker_bin(data, P_lin)
#
#map_ml     = mapmaker_ml(data, P_near, iN)
#map_ml_lin = mapmaker_ml(data, P_lin,  iN)
#
#enplot.write("test",        enplot.plot(signal_map, grid=False, range=r))
#enplot.write("test_binned", enplot.plot(map_binned, grid=False, upgrade=4, range=r))
#enplot.write("test_binned_lin", enplot.plot(map_binned_lin, grid=False, upgrade=4, range=r))
#enplot.write("test_ml",     enplot.plot(map_ml, grid=False, upgrade=4, range=r))
#enplot.write("test_ml_lin", enplot.plot(map_ml_lin, grid=False, upgrade=4, range=r))
#1/0
