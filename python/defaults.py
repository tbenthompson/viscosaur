import copy
import viscosaur as vc

secs_in_a_year = 3600 * 24 * 365.0


# This function creates a basic universal set of parameters for any viscosaur
# run. No parameters specific to a certain model should be included in here.
#
def default_params():
    defaults = dict()
    # Please don't change this....
    defaults['secs_in_a_year'] = secs_in_a_year

    # Meshing descriptors.
    # minimum corner(x_min, y_min)
    defaults['min_corner_x'] = 50.0
    defaults['min_corner_y'] = 0.0
    # maximum corner(x_max, y_max)
    defaults['max_corner_x'] = 5.0e4
    defaults['max_corner_y'] = 4.0e4

    # Polynomial degree of the elements.
    defaults['fe_degree'] = 2

    # TODO: Max degree for hp adaptivity (NOT IMPLEMENTED) Wait for deal.II
    # to include hp adaptivity in parallel
    defaults['max_degree'] = 10

    # Should we load a mesh or create a coarse version and refine it?
    defaults["load_mesh"] = False
    defaults["mesh_filename"] = None

    # How many times to isotropically refine the grid initially.
    defaults['initial_isotropic_refines'] = 3
    defaults['initial_adaptive_refines'] = 8

    # How often to refine once running
    defaults['refine_interval'] = 50

    # Maximum and minimum refinement levels
    defaults['max_grid_level'] = 14
    defaults['min_grid_level'] = 2

    # Refinement and coarsening percentages
    defaults['refine_frac'] = 0.2
    defaults['coarse_frac'] = 0.2

    # time stepping
    defaults['t_max'] = 100.0 * secs_in_a_year
    defaults['time_step'] = defaults['t_max'] / 100.0
    # How many little steps to take on the first step to make sure that
    # the error is not too high for the higher order BDF2 time stepper.
    defaults['first_substeps'] = 10

    # Clears the data directory at the beginning of computation.
    defaults['clear_data_dir'] = True
    # Compresses the data directory to a file of the same name in the top level
    # directory after computation is complete.
    defaults['compress_data_dir'] = True
    # Where to save data?
    defaults['data_dir'] = 'data/test'

    #Compare the output to an analytic solution?
    defaults['test_output'] = False

    # How often to output
    defaults['output_interval'] = 50

    # Should we plot?
    defaults['output'] = True

    #Should the mantle boundary conditions be neumann or dirichlet?
    defaults['mantle_neumann'] = False

    return defaults

