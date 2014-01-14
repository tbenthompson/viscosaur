import sys
import viscosaur as vc
import copy
import yep
import defaults
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

# Set up the parameters to be used.
params = defaults.default_params()
params['abc'] = 1000

# Initialize the viscosaur system, including deal.ii, PETSc (or Trilinos),
# and MPI.
instance = vc.Vc(sys.argv)

# Setup a 2D poisson solver.
rhs = vc.SinRHS2D()
poisson = vc.Poisson2D(params)

# Run a poisson solve
abc = poisson.run(rhs)

# We must define the slip fnc outside the TLA constructor, otherwise
# it appears to get deleted (maybe by python?)
sf = vc.CosSlipFnc(params['fault_depth'])
tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'], sf)

initSzx = vc.InitSzx2D(tla)
initSzy = vc.InitSzy2D(tla)

print "Whoa"
dof_handler = poisson.get_dof_handler()
print "Whoa2"
rhs2 = vc.OneStepRHS2D(initSzx, initSzy, dof_handler)
print "Whoa3"



print "From python: Poisson complete"

# Stop the profiler
yep.stop()
