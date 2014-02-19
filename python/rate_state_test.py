import numpy as np
import viscosaur as vc
import defaults
import copy
import matplotlib.pyplot as pyp

sigma = 200e6
tau = np.linspace(50e6, 200e6)
amb = 0.02
f0 = 0.7
v0 = 1e-11
v = v0 * np.exp((tau - sigma * f0) / (sigma * amb))
pyp.plot(tau, v)
pyp.show()
print v
# params = defaults.default_params()
# params['fault_slip'] = 1.0
# params['fault_depth'] = 1.0e4
# params['elastic_depth'] = 1.0e4
# params['viscosity'] = 5.0e19
# params['shear_modulus'] = 3.0e10
# params['time_step'] = params['secs_in_a_year'] * 0.01
# sf = vc.ConstantSlipFnc(params['fault_depth'])
# tla = vc.TwoLayerAnalytic(params['fault_slip'],
#                           params['fault_depth'],
#                           params['shear_modulus'],
#                           params['viscosity'], sf)
# init_strs = vc.SimpleInitStress2D(tla)
# inv_visc = vc.InvViscosityTLA2D(params)
#
# y = np.linspace(0, 30000, 100)
# v = np.array([0 + (5e-11 / 30000.0) * y_i for y_i in y])
# tau = np.array([init_strs.value(vc.Point2D(2.0, y_i), 1) for y_i in y])
# normal_strs = np.array([y_i * 9.8 * 3000 for y_i in y])
# normal_strs[0] = 1000 * 9.8 * 50
# amb = np.array([-0.02 + (y_i * (0.04 / 30000.)) for y_i in y])
# iv = np.array([inv_visc.value_easy(vc.Point2D(0, y[i]), 0.0, tau[i]) for i in range(100)])
#
# #compute new values
# tau_new = copy.copy(tau)
# v_new = copy.copy(v)
# for i in range(0, 50):
#     dvdy = np.zeros(100)
#     dvdy[1:-1] = (v_new[2:] - v_new[0:-2]) / (2 * (y[1] - y[0]))
#     # Free surface
#     dvdy[0] = 0
#     # weak at depth
#     dvdy[-1] = 0
#
#     tau_dot = (-iv * tau_new + dvdy) * params['shear_modulus']
#     v_dot = (v_new * tau_dot) / (normal_strs * amb)
#     tau_delta = params['time_step'] * tau_dot
#     v_delta = params['time_step'] * v_dot
#     tau_resid = (((tau_new - tau) - params['time_step'] * tau_dot) / tau_new)
#     v_resid = (((v_new - v) - params['time_step'] * v_dot) / v_new)
#     import pdb;pdb.set_trace()
#     print np.sum(v_resid + tau_resid)
#     tau_new = tau + tau_delta
#     v_new = v + v_delta
#
