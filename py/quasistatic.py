import numpy as np
import viscosaur as vc
import defaults
params = defaults.default_params()
params['min_corner_x'] = 100.0
params['min_corner_y'] = 0.0
params['max_corner_x'] = 2e4
params['max_corner_y'] = 2e4
params['t_max'] = params['secs_in_a_year'] * 50.0
params['time_step'] = params['secs_in_a_year'] * 0.5
params['test_output'] = False
params['num_threads'] = 1
params['fe_degree'] = 2
params['initial_isotropic_refines'] = 5

params['fault_slip'] = 1.0
params['fault_depth'] = 1e4
params['viscosity'] = 5.0e19
params['shear_modulus'] = 30e9
params['inv_rho'] = 1.0 / (10 ** 23)

class GaussStress(vc.PyFunction2D):
    def __init__(self, c):
        super(GaussStress, self).__init__()
        self.c = c
    def get_value(self, x, y, component):
        return np.exp(-(((x - self.c[0]) ** 2) +
                        ((y - self.c[1]) ** 2)) / 1000000.0)

# init_mem = GaussStress((0.0, 1e4))
slip_fnc = vc.ConstantSlipFnc(params['fault_depth'])

tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'],
                          slip_fnc)

init_mem = vc.SimpleInitStress2D(tla)

init_disp = vc.ZeroFunction2D(1)
