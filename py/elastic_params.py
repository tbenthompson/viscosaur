import numpy as np
import viscosaur as vc
import defaults
params = defaults.default_params()
params['min_corner_x'] = 0.0
params['min_corner_y'] = 0.0
params['max_corner_x'] = 2e4
params['max_corner_y'] = 2e4
params['t_max'] = 5.0
params['time_step'] = 0.001
params['test_output'] = False
params['num_threads'] = 1
params['fe_degree'] = 2
params['initial_isotropic_refines'] = 5

params['fault_depth'] = 1e4
params['viscosity'] = 5.0e30
params['shear_modulus'] = 30e9
params['inv_rho'] = 1.0 / 3000.0

inv_visc = vc.InvViscosityTLA2D(params)

class GaussStress(vc.PyFunction2D):
    def get_value(self, x, y, component):
        return np.exp(-((x ** 2) + (y ** 2)) / 1000000.0)

init_mem = vc.ZeroFunction2D(2)

init_disp = GaussStress()

bc_plate = vc.ZeroFunction2D(1)
bc_fault = vc.ZeroFunction2D(1)
