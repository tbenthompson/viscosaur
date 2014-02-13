import controller
import viscosaur as vc
import simple_solver
import defaults

# Set up the parameters to be used.
params = defaults.default_params()
params['initial_adaptive_refines'] = 9
params['max_grid_level'] = 11
params['t_max'] = 100.0 * defaults.secs_in_a_year
params['time_step'] = params['t_max'] / 8.0
params['load_mesh'] = True
params['mesh_filename'] = 'saved_mesh.msh'
params['refine_frac'] = 0.2
params['coarse_frac'] = 0.2
params['test_output'] = True
params['fe_degree'] = 2
params['first_substeps'] = 10

# Initial stress setup -- fed into an elastic half-space solution
# to determine initial conditions. In the future, I could numerically
# solve a Poisson problem to determine a solution that would allow
# slip variations and elastic modulus variations.
params['fault_slip'] = 1.0
params['fault_depth'] = 1.0e4
params['elastic_depth'] = 1.0e4

# Material parameters for a maxwell linear viscoelastic material.
params['viscosity'] = 5.0e19
params['shear_modulus'] = 3.0e10



c = controller.Controller(params)

inv_visc = vc.InvViscosityTLA2D(params)

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

simple_solver = simple_solver.SimpleSolver(params, inv_visc, vel_bc, c)
simple_solver.run(init_strs, init_vel, exact_vel)

c.proc0_out("From python: run complete")
c.kill()
