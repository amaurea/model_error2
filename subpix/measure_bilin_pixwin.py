import numpy as np

# Some helper functions

def unitvec(n, i):
	res = np.zeros(n)
	res[i] = 1
	return res

def func_to_mat(func, n):
	return np.array([func(unitvec(n,i)) for i in range(n)]).T

def wrap(x): return np.concatenate([x,x[:1]])

def fourier_extract(fimap, oshape):
	cny, cnx = np.minimum(fimap.shape[-2:], oshape[-2:])
	hny, hnx = cny//2, cnx//2
	fomap = np.zeros(fimap.shape[:-2]+oshape[-2:], fimap.dtype)
	fomap[...,:hny,       :hnx       ] = fimap[...,:hny,       :hnx       ]
	fomap[...,:hny,       -(cnx-hnx):] = fimap[...,:hny,       -(cnx-hnx):]
	fomap[...,-(cny-hny):,:hnx       ] = fimap[...,-(cny-hny):,:hnx       ]
	fomap[...,-(cny-hny):,-(cnx-hnx):] = fimap[...,-(cny-hny):,-(cnx-hnx):]
	return fomap

npix  = 1000
nsamp = 4*npix
#npix  =  40
#nsamp = 160

# Linint pointing matrix
pix       = np.arange(nsamp, dtype=float)*npix/nsamp
pix_left  = np.floor(pix).astype(int)
rx        = pix-pix_left
ix1       = pix_left % npix
ix2       = (pix_left+1)% npix
P         = np.zeros([nsamp,npix])
P         [np.arange(nsamp),ix1] += 1-rx
P         [np.arange(nsamp),ix2] += rx

# fSignal:   d = F"a
# Response:  m = (P'P)"P'd
# fResponse: g = Fm = F(P'P)"P'F"a

iF = func_to_mat(lambda x: np.fft.irfft(x, n=nsamp), npix//2+1)*nsamp/npix
F  = func_to_mat(np.fft.rfft, npix)
R  = F.dot(np.linalg.inv(P.T.dot(P)).dot(P.T.dot(iF))).real
# R is diagonal to machine precision, as I hoped. So
# it's a simple transfer function
tfun = np.diag(R)
freq = np.fft.rfftfreq(npix)
np.savetxt("tfun_lin_1d.txt", np.array([freq,tfun]).T, fmt="%15.7e")
del iF, F, R, P

# The stuff below is just a test of the separability of 2d linint.
# It is indeed separable.
#
#nside = 40
#nscan = nside*4
#npix  = nside**2
#nsamp = nscan**2
#
## Build a bilinear pointing matrix. Here a sample with coordinates
## y,x has value
## (1-ry)*(1-rx)*val[iy,ix] + (1-ry)*rx*val[iy,ix+1] + ry*(1-rx)*val[iy+1,ix] + ry*rx*val[iy+1,ix+1]
## where iy = floor(y) and ry = y-iy, etc.
## We want the pixel centers to be the control points for the interpolation.
## To get this we need floor(pix) instead of floor(pix+0.5)
##pix_left  = np.floor(pix+0.5).astype(int)
#pix       = (np.mgrid[:nscan,:nscan]*nside/nscan).reshape(2,-1)
#pix_left  = np.floor(pix).astype(int)
#ry, rx    = pix-pix_left
#iy1,ix1   = pix_left % nside
#iy2,ix2   = (pix_left+1)% nside
#sall      = np.arange(nsamp)
#P         = np.zeros([nsamp,npix])
#P[sall,iy1*nside+ix1] += (1-ry)*(1-rx)
#P[sall,iy1*nside+ix2] += (1-ry)*rx
#P[sall,iy2*nside+ix1] += ry*(1-rx)
#P[sall,iy2*nside+ix2] += ry*rx
#
#def iF_fun(x):
#	xfull = fourier_extract(x.reshape(nside,nside),(nscan,nscan))*(nscan/nside)**2
#	return np.fft.ifft2(xfull,s=(nscan,nscan)).reshape(-1)
#def F_fun(x):
#	return np.fft.fft2(x.reshape(nside,nside)).reshape(-1)
#nf = nside**2
#iF = func_to_mat(iF_fun, nf)
#F  = func_to_mat(F_fun, npix)
#
#R     = F.dot(np.linalg.inv(P.T.dot(P)).dot(P.T.dot(iF))).real
#tfun2 = np.diag(R).reshape(nside,nside)
#np.save("moo_tfun2d.npy", tfun2)
#tfun_ext = np.concatenate([tfun,tfun[-1:0:-1]])
#tfun2_model = tfun_ext[:,None]*tfun_ext[None,:]
#np.save("moo_tfun2d_model.npy", tfun2_model)
