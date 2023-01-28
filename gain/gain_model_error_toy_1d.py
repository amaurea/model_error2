import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sigma", type=float, default=0.1)
parser.add_argument("-l", "--lknee", type=float, default=3000)
parser.add_argument("-a", "--alpha", type=float, default=-3.5)
parser.add_argument("ofile")
args = parser.parse_args()
import numpy as np

l     = 10**np.linspace(np.log10(1),np.log10(10000),100)
iN    = (1 + (l/args.lknee)**args.alpha)**-1
tfun  = iN/(iN + np.mean(iN)*args.sigma**2)
np.savetxt(args.ofile, np.array([l, tfun, 1/iN]).T, fmt="%15.7e")
