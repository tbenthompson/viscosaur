import copy
import viscosaur as vc

secs_in_a_year = 3600 * 24 * 365.0
def default_params():
    defaults = dict()

    # Meshing descriptors.
    # minimum corner(x_min, y_min)
    defaults['min_corner'] = vc.Point2D(100.0 / 1.0e4, 0.0)
    # maximum corner(x_max, y_max)
    defaults['max_corner'] = vc.Point2D(2.0, 2.0)
    # How many times to isotropically refine the grid initially.
    defaults['initial_isotropic_refines'] = 6
    defaults['initial_adaptive_refines'] = 2

    # Maximum and minimum refinement levels
    defaults['max_grid_level'] = 9
    defaults['min_grid_level'] = 3

    # time stepping
    defaults['t_max'] = 100.0 * secs_in_a_year
    defaults['delta_t'] = defaults['t_max'] / 1.0

    # Adaptive meshing parameters
    # defaults['adaptive_mesh'] = False
    # defaults['load_mesh'] = False
    # defaults['just_build_adaptive'] = False
    # defaults['adapt_tol'] = 5e-5
    # defaults['save_mesh'] = False
    # defaults['all_steps_adaptive'] = True
    # defaults['mesh_file'] = 'mesh.h5'

    # Far field plate rate boundary condition.
    defaults['plate_rate'] = 0#(40.0 / 1.0e3) / secs_in_a_year  # 40 mm/yr

    # Initial stress setup -- fed into an elastic half-space solution
    # to determine initial conditions. In the future, I could numerically
    # solve a Poisson problem to determine a solution that would allow
    # slip variations and elastic modulus variations.
    defaults['fault_slip'] = 1.0
    defaults['fault_depth'] = 1.0e4
    defaults['elastic_depth'] = 1.0e4

    # Material parameters for a maxwell linear viscoelastic material.
    defaults['viscosity'] = 5.0e19
    defaults['shear_modulus'] = 3.0e10

    # from analytic_fast import integral_stress, cosine_slip_fnc, integral_velocity
    # slip_fnc = cosine_slip_fnc(defaults['fault_depth'])
    # defaults['initial_stress'] = lambda x, y: integral_stress(x, y, slip_fnc)
    # defaults['velocity'] = lambda x, y, t: integral_velocity(x, y, t, slip_fnc)

    #Boundary Conditions
    # defaults['bcs'] = 'test'

    # Where to save data?
    defaults['data_dir'] = 'test'

    defaults['plot'] = True

    # Calculate and save error
    defaults['calc_error'] = True
    return defaults

