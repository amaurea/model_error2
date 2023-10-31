import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lknee",  type=float, default=1000)
parser.add_argument("-a", "--alpha",  type=float, default=-3)
parser.add_argument("-g", "--gainerr",type=float, default=0.1, help="Relative gain excess for second detector relative to first")
parser.add_argument("ofile")
args = parser.parse_args()
import numpy as np

l     = 10**np.linspace(np.log10(1),np.log10(10000),100)
Natm  = (l/args.lknee)**args.alpha
corr  = Natm/(1+Natm)
g1    = 1
g2    = g1*(1+args.gainerr)
tfun  = (1-corr)/(1-(2*g1*g2)/(g1**2+g2**2)*corr)
np.savetxt(args.ofile, np.array([l, tfun, 1/corr]).T, fmt="%15.7e")
