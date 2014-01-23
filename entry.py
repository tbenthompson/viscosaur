import sys
import viscosaur as vc
import copy
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

pd = vc.ProblemData2D(params)
# We must define the slip fnc outside the TLA constructor, otherwise
# it appears to get deleted (maybe by python?)
sf = vc.ConstantSlipFnc(params['fault_depth'])
tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'], sf)
initSzx = vc.SimpleInitSzx2D(tla)
initSzy = vc.SimpleInitSzy2D(tla)
soln = vc.Solution2D(pd)

def run():
    vel = vc.SimpleVelocity2D(tla)
    vel.set_t(params['time_step'])

    # Setup a 2D poisson solver.
    for i in range(params['initial_adaptive_refines']):
        strs_update = vc.Stress2D(soln, pd)
        soln.apply_init_cond(initSzx, initSzy)
        v_solver = vc.Velocity2D(soln, vel, pd)

        strs_update.step(soln)
        v_solver.step(soln)
        soln.output(i, vel)
        pd.refine_grid(soln)

run()
# one_step_strs()
print "From python: run complete"
