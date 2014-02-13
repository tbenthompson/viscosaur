import controller
import viscosaur as vc
import simple_solver
import defaults

# Set up the parameters to be used.
params = defaults.default_params()
params['initial_adaptive_refines'] = 30
params['max_grid_level'] = 30
params['t_max'] = 100.0 * defaults.secs_in_a_year
params['time_step'] = params['t_max'] / 30.0
params['load_mesh'] = False
params['mesh_filename'] = 'saved_mesh.msh'
params['refine_frac'] = 0.2
params['coarse_frac'] = 0.2
params['test_output'] = False
params['min_corner'] = vc.Point2D(10.0, 0.0)
params['max_corner'] = vc.Point2D(5.0e4, 4.0e4)
params['fe_degree'] = 1

# Initial stress setup -- fed into an elastic half-space solution
# to determine initial conditions. In the future, I could numerically
# solve a Poisson problem to determine a solution that would allow
# slip variations and elastic modulus variations.
params['fault_slip'] = 1.0
params['fault_depth'] = 1.0e4
params['elastic_depth'] = 1.0e4

# Parameters for a power law viscosity function
params['power_law_A'] = 2.2e-4 * 10 ** (-6 * 3.4) # (Pa^-n)/sec
params['power_law_n'] = 3.4
params['power_law_Q'] = 2.6e5 # J/mol
params['shear_modulus'] = 3.0e10 # Pa
# Far field plate rate boundary condition.
params['plate_rate'] = 0#(40.0 / 1.0e3) / secs_in_a_year  # 40 mm/yr

c = controller.Controller(params)

inv_visc = vc.InvViscosityPowerLaw2D(params)

sf = vc.CosSlipFnc(params['fault_depth'])
tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          1.0, sf)
init_strs = vc.InitStress2D(tla)
init_vel = vc.ConstantBC2D(0.0)
init_vel.set_t(0.0)
exact_vel = vc.ConstantBC2D(0.0)
vel_bc = vc.ConstantBC2D(0.0)

simple_solver = simple_solver.SimpleSolver(params, inv_visc, vel_bc, c)
simple_solver.run(init_strs, init_vel, exact_vel)

c.proc0_out("From python: run complete")
c.kill()
