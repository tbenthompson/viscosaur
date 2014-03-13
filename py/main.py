import controller
import viscosaur as vc
import defaults
import numpy as np

# Set up the parameters to be used.
# from elastic_params import params, init_mem, init_disp
# from attenuated_waves import params, init_mem, init_disp
from quasistatic import params, init_mem, init_disp

c = controller.Controller(params)

# import pdb;pdb.set_trace()
pd = vc.ProblemData2D(params)
soln = vc.Solution2D(pd)

soln.apply_init_cond(init_mem, init_disp);
soln.output(params['data_dir'], 'play.', init_disp)

inv_visc = vc.InvViscosityTLA2D(params)

stepper = vc.Stepper2D()
t_max = params['t_max']
t = 0
i = 0
dt = params['time_step']
while t < t_max:
    soln.start_timestep()
    inv_rho = params['inv_rho']
    stepper.step(soln, inv_visc, inv_rho, dt)
    # if i % 10 == 0:
    soln.output(params['data_dir'], 'play-' + ("%05d" % i) + '.', init_disp)
    t += dt
    i += 1

# pd.start_refine(self.soln.cur_vel)
# soln.start_refine()
# pd.execute_refine()
# soln.reinit()
# soln.post_refine(self.soln)
# scheme.reinit(self.pd);

c.proc0_out("From python: run complete")
c.kill()
