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
import ctypes
mpi = ctypes.CDLL('libmpi.so.0', ctypes.RTLD_GLOBAL)

# Set up the parameters to be used.
params = defaults.default_params()
params['abc'] = 1000

# Initialize the viscosaur system, including deal.ii, PETSc (or Trilinos),
# and MPI.
instance = vc.Vc(sys.argv)

# We must define the slip fnc outside the TLA constructor, otherwise
# it appears to get deleted (maybe by python?)
sf = vc.CosSlipFnc(params['fault_depth'])
tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'], sf)

initSzx = vc.InitSzx2D(tla)
initSzy = vc.InitSzy2D(tla)

# Setup a 2D poisson solver.
pd = vc.ProblemData2D(params)
rhs = vc.SinRHS2D()
rhs2 = vc.OneStepRHS2D(initSzx, initSzy, pd)
poisson = vc.Poisson2D(pd)

# Run a poisson solve
abc = poisson.run(rhs2)


print "Whoa"
dof_handler = poisson.get_dof_handler()
print "Whoa2"
print "Whoa3"



print "From python: Poisson complete"

# Stop the profiler
yep.stop()
