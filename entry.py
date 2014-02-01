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
params['initial_adaptive_refines'] = 13
params['max_grid_level'] = 13
params['t_max'] = 100.0 * defaults.secs_in_a_year
params['time_step'] = params['t_max'] / 4.0
params['load_mesh'] = False
params['mesh_filename'] = 'saved_mesh.msh'
params['refine_frac'] = 0.2
params['coarse_frac'] = 0.2

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


inv_visc = vc.InvViscosityTLA2D(params)
pd = vc.ProblemData2D(params, inv_visc)
# We must define the slip fnc outside the TLA constructor, otherwise
# it appears to get deleted (maybe by python?)
# TODO: Use boost python custodian and ward to maintain existence of some
# objects that otherwise get garbage collected by python
sf = vc.ConstantSlipFnc(params['fault_depth'])
tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'], sf)
init_szx = vc.SimpleInitSzx2D(tla)
init_szy = vc.SimpleInitSzy2D(tla)
init_vel = vc.SimpleVelocity2D(tla)
init_vel.set_t(0.0)
exact_vel = vc.SimpleVelocity2D(tla)

vel_bc = vc.SimpleVelocity2D(tla)

def run():
    soln = vc.Solution2D(pd)
    vel_bc.set_t(params['time_step'])

    # Setup a 2D poisson solver.
    if not params["load_mesh"]:
        for i in range(params['initial_adaptive_refines']):
            scheme = vc.FwdEuler2D(pd)
            strs_solver = vc.Stress2D(soln, pd)
            vel_solver = vc.Velocity2D(soln, vel_bc, pd, scheme)
            soln.apply_init_cond(init_szx, init_szy, init_vel)

            soln.start_timestep()
            strs_solver.tentative_step(soln, scheme)
            vel_solver.step(soln, scheme)
            soln.output(params['data_dir'], 'init_refinement_' + str(i) + '.',
                        vel_bc)
            pd.start_refine(soln.current_velocity)
            pd.execute_refine()
            soln.reinit()
        proc0_out("Done with first time step spatial adaptation.")
        pd.save_mesh("saved_mesh.msh")

    scheme = vc.FwdEuler2D(pd)
    strs_solver = vc.Stress2D(soln, pd)
    vel_solver = vc.Velocity2D(soln, vel_bc, pd, scheme)
    soln.reinit()
    soln.apply_init_cond(init_szx, init_szy, init_vel)
    t = 0
    i = 1
    while t < params['t_max']:
        t += params['time_step']
        proc0_out("\n\nSolving for time = " + \
                  str(t / defaults.secs_in_a_year) + " \n")
        vel_bc.set_t(t)
        vel_solver.update_bc(vel_bc, scheme)

        soln.start_timestep()
        strs_solver.tentative_step(soln, scheme)
        vel_solver.step(soln, scheme)
        strs_solver.correction_step(soln, scheme)

        exact_vel.set_t(t)
        filename = "solution-" + str(i) + "."
        soln.output(params['data_dir'], filename, exact_vel)


        pd.start_refine(soln.current_velocity)
        sol_trans = soln.start_refine()
        pd.execute_refine()
        scheme = vc.BDFTwo2D(pd)
        strs_solver = vc.Stress2D(soln, pd)
        vel_solver = vc.Velocity2D(soln, vel_bc, pd, scheme)
        soln.reinit()
        soln.post_refine(sol_trans)
        i += 1


#Actually run the main function
# TODO: This would be unnecessary with proper custodian and warding of boost
# python code
run()

if params['compress_data_dir']:
    archive_name = params['data_dir'].replace('/', '.')
    archive_name += datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name += '.tar.gz'
    tar = tarfile.open(archive_name, "w:gz")
    tar.add(params['data_dir'])
    tar.close()

proc0_out("From python: run complete")
