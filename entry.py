import sys
import viscosaur as vc
import copy
import yep
# To profile the C++ and python use yep with
# python -m yep entry.py
# and then use pperf (part of gperftools) to convert to callgrind format like:
# google-pprof --callgrind viscosaur.so entry.py.prof > output.callgrind
# I found yep at https://pypi.python.org/pypi/yep

# These two lines are required to prevent a weird openmpi bug
# that causes mpi init to fail with "undefined symbol: mca_base_param_reg_int"
# Found the fix here: http://fishercat.sr.unh.edu/trac/mrc-v3/ticket/5
from ctypes import *
mpi = CDLL('libmpi.so.0', RTLD_GLOBAL)


# Initialize the viscosaur system, including deal.ii, PETSc (or Trilinos),
# and MPI.
instance = vc.Vc(sys.argv)

# Setup a 2D poisson solver.
rhs = vc.PoissonRHS2D()
poisson = vc.Poisson2D()

# Run a poisson solve
poisson.run(rhs)

print "Poisson complete"

# Stop the profiler
yep.stop()
