import controller
import viscosaur as vc
import defaults

# class Problem(object):
#     def get_bc(self):
#         raise Exception("Not implemented in Problem base class")
#     def get_inv_visc(self):
#         raise Exception("Not implemented in Problem base class")
#     def get_analytic(self):
#         raise Exception("Not implemented in Problem base class")
#     def get_init_strs(self):
#         raise Exception("Not implemented in Problem base class")
#     def get_init_vel(self):
#         raise Exception("Not implemented in Problem base class")
#
# class TLA(object):
#     def get_bc(self):
#         return vc.SimpleVelocity2D

def step(soln, strs_solver, vel_solver, scheme, time_step):
    soln.start_timestep()
    strs_solver.tentative_step(soln, scheme, time_step)
    vel_solver.step(soln, scheme, time_step)
    strs_solver.correction_step(soln, scheme, time_step)

def refine(pd, soln, strs_solver, vel_solver, scheme):
    pd.start_refine(soln.current_velocity)
    soln.start_refine()
    pd.execute_refine()
    soln.reinit()
    soln.post_refine(soln)
    scheme.reinit(pd);
    vel_solver.reinit(pd, soln, vel_bc, scheme)
    strs_solver.reinit(pd);

# Set up the parameters to be used.
params = defaults.default_params()
params['initial_adaptive_refines'] = 13
params['max_grid_level'] = 15
params['t_max'] = 100.0 * defaults.secs_in_a_year
params['time_step'] = params['t_max'] / 16.0
params['load_mesh'] = False
params['mesh_filename'] = 'saved_mesh.msh'
params['refine_frac'] = 0.2
params['coarse_frac'] = 0.2
params['test_output'] = True

c = controller.Controller(params)

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
init_strs = vc.SimpleInitStress2D(tla)
init_vel = vc.SimpleVelocity2D(tla)
init_vel.set_t(0.0)
exact_vel = vc.SimpleVelocity2D(tla)

vel_bc = vc.SimpleVelocity2D(tla)

sub_timesteps = 200
#sub_timesteps = 2
# Setup a 2D poisson solver.
soln = vc.Solution2D(pd)
scheme = vc.FwdEuler2D(pd)
vel_solver = vc.Velocity2D(pd, soln, vel_bc, scheme)
strs_solver = vc.Stress2D(pd)
if not params["load_mesh"]:
    time_step = params['time_step'] / sub_timesteps
    vel_bc.set_t(params['time_step'] / sub_timesteps)
    for i in range(params['initial_adaptive_refines']):
        soln.apply_init_cond(init_strs, init_vel)
        step(soln, strs_solver, vel_solver, scheme, time_step)
        soln.output(params['data_dir'], 'init_refinement_' + str(i) + '.',
                    vel_bc)
        refine(pd, soln, strs_solver, vel_solver, scheme)
    c.proc0_out("Done with first time step spatial adaptation.")
    pd.save_mesh("saved_mesh.msh")

soln.apply_init_cond(init_strs, init_vel)
t = 0
i = 1
while t < params['t_max']:
    time_step = params['time_step'] / sub_timesteps
    for sub_t in range(0, sub_timesteps):
        t += time_step
        c.proc0_out("\n\nSolving for time = " + \
                  str(t / defaults.secs_in_a_year) + " \n")
        vel_bc.set_t(t)
        vel_solver.update_bc(vel_bc, scheme)
        step(soln, strs_solver, vel_solver, scheme, time_step)
        exact_vel.set_t(t)
        filename = "solution-" + str(i) + "."
        soln.output(params['data_dir'], filename, exact_vel)
        refine(pd, soln, strs_solver, vel_solver, scheme)
    if i == 1:
        # At the end of the first time step, we switch to using a BDF2 scheme
        sub_timesteps = 1
        soln.init_multistep(init_strs, init_vel)
        scheme = vc.BDFTwo2D(pd)
    i += 1

c.proc0_out("From python: run complete")
c.kill()
