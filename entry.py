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

        strs_update.tentative_step(soln)
        v_solver.step(soln)
        soln.output(i, vel)
        pd.start_refine(soln)
        pd.execute_refine()

    print "Done with first time step spatial adaptation."
    strs_update = vc.Stress2D(soln, pd)
    soln.apply_init_cond(initSzx, initSzy)
    v_solver = vc.Velocity2D(soln, vel, pd)
    t = 0
    while t < params['t_max']:
        t += params['time_step']
        print("Solving for time = " + str(t / defaults.secs_in_a_year))

        vel.set_t(t)
        v_solver.update_bc(vel)

        strs_update.tentative_step(soln)
        v_solver.step(soln)
        strs_update.correction_step(soln)
        # Fix the output naming scheme
        soln.output(i, vel)

        pd.start_refine(soln)
        print "Started refinement!"
        sol_trans = soln.start_refine()
        import pdb;pdb.set_trace()
        print "Prepared solution transfer!"
        pd.execute_refine()
        print "Refinement!"
        new_soln = vc.Solution2D(pd)
        strs_update = vc.Stress2D(new_soln, pd)
        v_solver = vc.Velocity2D(new_soln, vel, pd)
        print "New Objs!"
        new_soln.post_refine(sol_trans)


run()
# one_step_strs()
print "From python: run complete"
