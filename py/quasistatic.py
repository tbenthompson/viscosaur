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
params['initial_isotropic_refines'] = 4

params['fault_slip'] = 1.0
params['fault_depth'] = 1e4
params['viscosity'] = 5.0e19
params['shear_modulus'] = 30e9
params['inv_rho'] = 1.0 / (10 ** 23)
params['plate_rate'] = (40.0 / 1.0e3) / defaults.secs_in_a_year#40 mm/yr


slip_fnc = vc.CosSlipFnc(params['fault_depth'])

tla = vc.TwoLayerAnalytic(params['fault_slip'],
                          params['fault_depth'],
                          params['shear_modulus'],
                          params['viscosity'],
                          slip_fnc)

class InitMem(vc.PyFunction2D):
    def __init__(self, init_strs, inv_visc):
        super(InitMem, self).__init__()
        self.init_strs = init_strs
        self.inv_visc = inv_visc

    def get_value(self, x, y, component):
        strs = self.init_strs.value(vc.Point2D(x, y), component)
        #TODO: ONLY WORKS FOR LINEAR VISCOELASTIC!
        inv_visc = self.inv_visc.value_easy(vc.Point2D(x, y), 0.0, 0.0)
        return strs * inv_visc

inv_visc = vc.InvViscosityTLA2D(params)

init_mem = InitMem(vc.InitStress2D(tla), inv_visc)
print init_mem.get_value(10000.0, 20000.0, 0)

init_disp = vc.ZeroFunction2D(1)

class FarfieldPlateBC(vc.PyFunction2D):
    def __init__(self, rate):
        super(FarfieldPlateBC, self).__init__()
        self.rate = rate
        self.t = 0

    def set_t(self, t):
        self.t = t

    def get_value(self, x, y, component):
        return self.t * self.rate

class FaultBC(vc.PyFunction2D):
    def __init__(self, rate, fault_depth):
        super(FaultBC, self).__init__()
        self.rate = rate
        self.fault_depth = fault_depth
        self.t = 0

    def set_t(self, t):
        self.t = t

    def get_value(self, x, y, component):
        if y > self.fault_depth:
            return self.t * self.rate
        return 0.0

# bc_plate = FarfieldPlateBC(params['plate_rate'])
# bc_fault = FaultBC(params['plate_rate'], params['fault_depth'])
bc_plate = vc.ZeroFunction2D(1)
bc_fault = vc.ZeroFunction2D(1)
