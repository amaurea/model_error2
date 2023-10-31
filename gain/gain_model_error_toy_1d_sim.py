import numpy as np

# This is a numerical confirmation of the formula from
# equation 17/18 in the paper.

def ubash(A, n):
	"""Convert linear operator A to an (n,n) matrix"""
	Amat = np.zeros((n,n))
	for i in range(n):
		u = np.zeros(n)
		u[i] = 1
		Amat[:,i] = A(u)
	return Amat

nsamp  = 1000
nfreq  = nsamp//2+1
gain   = np.array([1,1.1])[:,None] # None makes it broadcast correctly later
igain  = 1/gain
ndet   = len(gain)
lknee  = 1000
l      = np.arange(nfreq)*2000/nfreq

# Set up a noise model consisting of 100% detector-correlated
# atmospheric noise with spectrum (l/lknee)**-3 plus uncorrelated
# white noise with power 1:
I    = np.eye(ndet)
C    = np.ones((ndet,ndet))
# Max just avoids division by zero
Natm = (np.maximum(1e-6,l)/lknee)**-3
Nl   = C[:,:,None]*Natm + I[:,:,None]
iNl  = np.linalg.inv(Nl.T).T # matrix inverse along detector axis

def iN(x):
	# The noise model is modulated by the gain errors.
	# In real life this would happen because the detector noise level
	# is measured from gain-affected data
	return igain*np.fft.irfft(np.sum(iNl*np.fft.rfft(igain*x),1), n=nsamp).real

# Draw the random signal
signal = 5+np.random.randn(nsamp)
# Data is the signal affected by the gain error, plus noise
# But we skip the noise here since the result is linear in the noise,
# and we only care about the expectation value
data   = gain*signal

# Build right-hand side
rhs    = np.sum(iN(data),0)
# Build the left-hand-side
def A(x): return np.sum(iN(x),0)
Amat = ubash(A, nsamp)
# Solve
mhat = np.linalg.solve(Amat, rhs)

# Calculate the transfer function
ps_signal = np.abs(np.fft.rfft(signal))**2
ps_mhat   = np.abs(np.fft.rfft(mhat  ))**2
tfun      = (ps_mhat/ps_signal)**0.5
# Overall scaling not interesting, just shape
tfun     /= np.max(tfun)

np.savetxt("gain_1d_toy_sim.txt", np.array([l,tfun, ps_signal,ps_mhat]).T, fmt="%15.7e")
