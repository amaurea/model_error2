# Draw a simple curve across a few pixels
import numpy as np
from pixell import utils
t  = np.linspace(np.pi/6,np.pi/2.5,100)
x  = -np.cos(t)*4
y  = np.sin(t)*4
x -= x[0]
y -= y[0] - 0.2
v  = x*y/4
v0 = utils.nint(x)*utils.nint(y)/4

np.savetxt("vals.txt", np.array([x,y,v,v0]).T, fmt="%15.7e")
