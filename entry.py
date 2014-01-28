import sys
import os
import tarfile
import datetime
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

# Initialize the viscosaur system, including deal.ii, PETSc (or Trilinos),
# and MPI.
instance = vc.Vc(sys.argv)
mpi_rank = instance.get_rank()
def proc0_out(info):
    if mpi_rank is 0:
        print(info)

# Set up the parameters to be used.
params = defaults.default_params()
params['initial_adaptive_refines'] = 5
params['max_grid_level'] = 9
params['time_step'] = params['t_max'] / 1000.0

# Clear the data directory if asked. The user is trusted to set the parameter
# appropriately and not delete precious data
if params['clear_data_dir'] and (mpi_rank is 0):
    folder = params['data_dir']
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e


pd = vc.ProblemData2D(params)
# We must define the slip fnc outside the TLA constructor, otherwise
# it appears to get deleted (maybe by python?)
# TODO: Use boost python custodian and ward to maintain existence of some
# objects that otherwise get garbage collected by python
sf = vc.ConstantSlipFnc(params['fault_depth'])
tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'], sf)
initSzx = vc.SimpleInitSzx2D(tla)
initSzy = vc.SimpleInitSzy2D(tla)

def run():
    soln = vc.Solution2D(pd)
    vel = vc.SimpleVelocity2D(tla)
    vel.set_t(params['time_step'])

    # Setup a 2D poisson solver.
    for i in range(params['initial_adaptive_refines']):
        strs_update = vc.Stress2D(soln, pd)
        soln.apply_init_cond(initSzx, initSzy)
        v_solver = vc.Velocity2D(soln, vel, pd)

        soln.start_timestep()
        strs_update.tentative_step(soln)
        v_solver.step(soln)
        # soln.output(i, vel)
        pd.start_refine(soln)
        pd.execute_refine()

    proc0_out("Done with first time step spatial adaptation.")
    strs_update = vc.Stress2D(soln, pd)
    soln.apply_init_cond(initSzx, initSzy)
    v_solver = vc.Velocity2D(soln, vel, pd)
    t = 0
    i = 0
    while t < params['t_max']:
        t += params['time_step']
        proc0_out("Solving for time = " + str(t / defaults.secs_in_a_year))

        vel.set_t(t)
        v_solver.update_bc(vel)

        soln.start_timestep()
        strs_update.tentative_step(soln)
        v_solver.step(soln)
        strs_update.correction_step(soln)
        # Fix the output naming scheme
        filename = "solution-" + str(i) + "."
        # soln.output(params['data_dir'], filename, vel)


        pd.start_refine(soln)
        sol_trans = soln.start_refine()
        pd.execute_refine()
        new_soln = vc.Solution2D(pd)
        strs_update = vc.Stress2D(new_soln, pd)
        v_solver = vc.Velocity2D(new_soln, vel, pd)
        new_soln.post_refine(sol_trans)
        soln = new_soln
        i += 1


run()
# one_step_strs()
if params['compress_data_dir']:
    archive_name = params['data_dir'].replace('/', '.')
    archive_name += datetime.datetime.now().strftime('%Y%m%d')
    archive_name += '.tar.gz'
    tar = tarfile.open(archive_name, "w:gz")
    tar.add(params['data_dir'], arcname = archive_name)
    tar.close()
proc0_out("From python: run complete")
