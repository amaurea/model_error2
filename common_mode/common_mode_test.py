# This experiment tested for model error bias when one has
# * wrong relative gains
# * a strong common mode in the noise model
# * large differences in detector sensitivity
# Conclusion: No transfer function observed, just an overall
# scale-independent gain error in the result.

# I then modified the experiment to have
# * wrong relative gains
# * a strong common mode in the noise model
# * detectors don't all point in the same direction
# This resulted in a large transfer function
#
# But I got too much power instead of too little.
#
# There is also a phase shift at low l. I though that
# this might smear out and lead to loss power with
# crosslinking, but that didn't happen. Depending on
# how the gain errors are done, the transfun either
# disappears or becomes smaller.
#
# Conclusion: This probably isn't an important
# contribution to our transfer function. But it's
# good that I could find a new mechanism after
# being stuck for so long.

import numpy as np
from pixell import fft, utils

crosslink = False
ndet  = 10
npix  = 100
srate = 400
pix   = np.arange(npix)
fknee = 50

def zip(tod): return tod.reshape(-1)
def unzip(x): return x.reshape(ndet,nsamp)

# Pointing matrix
if crosslink:
	nsamp = 2*npix
	P = np.zeros([ndet,nsamp,npix])
	# Forward scan
	for i in range(ndet): P[i,:npix] = np.roll(np.eye(npix),i,1)
	# Backward scan
	for i in range(ndet): P[i,npix:] = np.roll(np.eye(npix),i,1)[:,::-1]
else:
	nsamp = npix
	P = np.zeros([ndet,nsamp,nsamp])
	for i in range(ndet): P[i] = np.roll(np.eye(nsamp),i,1)
P = P.reshape(ndet*nsamp,npix)

t     = np.arange(nsamp)/srate
f     = fft.rfftfreq(nsamp, 1/srate)
nfreq = len(f)

# Noise matrix. Will have an uncorrelated
# part that's frequency-independent, and
# correlated part with a 1/f spectrum consisting of
# just a common mode
nmode = 1
#iW = np.diag(np.arange(ndet)+1) # ndet,ndet
iW = np.eye(ndet) # ndet,ndet
iE = np.zeros((nfreq,nmode,nmode)) # nfreq,nmode,nmode
with utils.nowarn():
	iE[:,0,0] = (np.maximum(f,0.01)/fknee)**3.5
	#iE[:,1,1] = (np.maximum(f,0.01)/fknee)**3.5 * 0.5
V = np.zeros((ndet,nmode))
V[:,0] = 1
#V[:,1] = np.sin(2*np.pi*np.arange(ndet)/ndet)
K  = (V.T.dot(iW).dot(V)) + iE # nfreq,nmode,nmode
iK = np.linalg.inv(K) # nfreq,nmode,nomde
iN = iW - np.einsum("fac,cb->fab",np.einsum("ab,fbc->fac",iW.dot(V),iK),V.T.dot(iW)) # nfreq,ndet,ndet

# Let's build a tod with a gain model error
signal   = np.sin(2*np.pi*pix/npix)
signal2  = np.sin(20*np.pi*pix/npix)
gainerrs = np.linspace(0.5,1.5,ndet)
#gainerrs = 1+np.sin(5*np.pi*np.arange(ndet)/ndet)
#gainerrs2= 1+np.sin(5*np.pi*np.arange(ndet)/ndet)
tod      = unzip(P.dot(signal)) *gainerrs[:,None]
tod2     = unzip(P.dot(signal2))*gainerrs[:,None]

#tod [:,:npix] *= gainerrs[:,None]; tod [:,npix:] *= gainerrs2[:,None]
#tod2[:,:npix] *= gainerrs[:,None]; tod2[:,npix:] *= gainerrs2[:,None]

# Let's try solving for the signal now. The model is d = Pm + n
rhs  = P.T.dot(zip(fft.irfft(np.einsum("fab,af->bf",iN,fft.rfft(tod)),normalize=True)))
rhs2 = P.T.dot(zip(fft.irfft(np.einsum("fab,af->bf",iN,fft.rfft(tod2)),normalize=True)))
# build iM using unit vector bashing
iM   = np.zeros((npix,npix))
for i in range(npix):
	u = np.zeros(npix)
	u[i]    = 1
	iM[i,:] = P.T.dot(zip(fft.irfft(np.einsum("fab,af->bf",iN,fft.rfft(unzip(P.dot(u)))),normalize=True)))
m = np.linalg.solve(iM,rhs)
m2 = np.linalg.solve(iM,rhs2)
np.savetxt("long.txt",  np.array([signal,m]).T, fmt="%8.6f")
np.savetxt("short.txt", np.array([signal2,m2]).T, fmt="%8.6f")
