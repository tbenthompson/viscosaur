import controller
import viscosaur as vc
import simple_solver
import defaults

# Set up the parameters to be used.
params = defaults.default_params()
params['initial_adaptive_refines'] = 1
params['max_grid_level'] = 8
params['t_max'] = 100.0 * defaults.secs_in_a_year
params['time_step'] = defaults.secs_in_a_year * 10.0
params['test_output'] = False
params['min_corner_x'] = 50.0
params['refine_interval'] = 5
params['output_interval'] = 5
params['load_mesh'] = True
params["mesh_filename"] = 'saved_mesh.msh'

params['fault_depth'] = 1.0e4
params['elastic_depth'] = 1.0e4
params['viscosity'] = 0.0
# Parameters for a power law viscosity function
params['power_law_n'] = 3.4 #dimensionless, it's an exponent!
params['power_law_A'] = 2.2e-4 * 10 ** (-6 * params['power_law_n'])#(Pa^-n)/sec
params['power_law_Q'] = 2.6e5 # J/mol
params['shear_modulus'] = 3.0e10 # Pa
# Far field plate rate boundary condition.
params['plate_rate'] = (40.0 / 1.0e3) / defaults.secs_in_a_year  # 40 mm/yr
params['mantle_neumann'] = True

c = controller.Controller(params)

inv_visc = vc.InvViscosityPowerLaw2D(params)

init_strs = vc.ZeroFunction2D(2)
init_vel = vc.ConstantBC2D(0.0)
init_vel.set_t(0.0)
exact_vel = vc.ConstantBC2D(0.0)
vel_bc = vc.FarFieldPlateBC2D(params['plate_rate'], params['max_corner_x'],
                              params['fault_depth'])


class ThisTestSolver(simple_solver.SimpleSolver):
    def after_timestep(self):
        if ((self.t / self.params['secs_in_a_year']) % 100) > 1.0:
            return

        # Add an earthquake with a certain amount of fault_slip to the stress.
        # In other words, this adds the stress field from a screw dislocation
        # to the existing stress field
        sf = vc.ConstantSlipFnc(params['fault_depth'])
        tla = vc.TwoLayerAnalytic(params['plate_rate'] * self.params['secs_in_a_year'] * 100,
                                  params['fault_depth'],
                                  params['shear_modulus'],
                                  params['viscosity'], sf)
        added_strs_fnc = vc.SimpleInitStress2D(tla)
        added_strs = vc.MPIVector()
        self.pd.strs_matrix_free.initialize_dof_vector(added_strs, 0)

        mfc = vc.MatrixFreeCalculation2D(self.pd, self.pd.strs_matrix_free,
                self.pd.strs_hanging_node_constraints);
        strs_op_factory = vc.StrsProjectionOpFactory2D()
        mfc.op_factory = strs_op_factory;
        mfc.apply_function(added_strs, added_strs_fnc);
        self.soln.cur_strs += added_strs

        # This restarts the time stepping to use a 1st order method again.
        self.sub_timesteps = self.params['first_substeps']
        self.local_step_index = 1
        self.scheme = vc.FwdEuler2D(self.pd)

solver = ThisTestSolver(params, inv_visc, vel_bc, c)
solver.run(init_strs, init_vel, exact_vel)

c.proc0_out("From python: run complete")
c.kill()
