import numpy as np
# Implementation of preconditioned conjugate gradients. More general than
# the scipy version, in that it does not assume that it knows how to perform
# the dot product. This makes it possible to use this for distributed x
# vectors. It's also reentrant.

def default_M(x):     return np.copy(x)
def default_dot(a,b): return a.dot(np.conj(b))

class CG:
	"""A simple Preconditioner Conjugate gradients solver. Solves
	the equation system Ax=b."""
	def __init__(self, A, b, x0=None, M=default_M, dot=default_dot):
		"""Initialize a solver for the system Ax=b, with a starting guess of x0 (0
		if not provided). Vectors b and x0 must provide addition and multiplication,
		as well as the .copy() method, such as provided by numpy arrays. The
		preconditioner is given by M. A and M must be functors acting on vectors
		and returning vectors. The dot product may be manually specified using the
		dot argument. This is useful for MPI-parallelization, for example."""
		# Init parameters
		self.A   = A
		self.b   = b
		self.M   = M
		self.dot = dot
		if x0 is None:
			self.x = b*0
			self.r = b
		else:
			self.x  = x0.copy()
			self.r  = b-self.A(self.x)
		# Internal work variables
		n = b.size
		z = self.M(self.r)
		self.rz  = self.dot(self.r, z)
		self.rz0 = float(self.rz)
		self.p   = z
		self.i   = 0
		self.err = np.inf
	def step(self):
		"""Take a single step in the iteration. Results in .x, .i
		and .err being updated. To solve the system, call step() in
		a loop until you are satisfied with the accuracy. The result
		can then be read off from .x."""
		Ap = self.A(self.p)
		alpha = self.rz/self.dot(self.p, Ap)
		self.x += alpha*self.p
		self.r -= alpha*Ap
		z       = self.M(self.r)
		next_rz = self.dot(self.r, z)
		self.err = next_rz/self.rz0
		beta = next_rz/self.rz
		self.rz = next_rz
		self.p  = z + beta*self.p
		self.i += 1
