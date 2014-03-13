import controller
import viscosaur as vc
import defaults
import numpy as np

# Set up the parameters to be used.
# from elastic_params import params, init_mem, init_disp, bc_plate, bc_fault
from attenuated_waves import params, init_mem, init_disp, bc_plate, bc_fault
# from quasistatic import params, init_mem, init_disp, bc_plate, bc_fault

c = controller.Controller(params)

# import pdb;pdb.set_trace()
pd = vc.ProblemData2D(params)
soln = vc.Solution2D(pd)

# restart = 5
#
# for r in range(restart + 1):
soln.apply_init_cond(init_mem, init_disp);
soln.output(params['data_dir'], 'play.', init_disp)

inv_visc = vc.InvViscosityTLA2D(params)

stepper = vc.Stepper2D(pd)
t_max = params['t_max']
t = 0
i = 0
dt = params['time_step']
while t < t_max:
    # bc_plate.set_t(t)
    # bc_fault.set_t(t)
    soln.start_timestep()
    inv_rho = params['inv_rho']
    stepper.step(soln, inv_visc, inv_rho, dt, bc_fault, bc_plate)
    # if i % 10 == 0:
        # if r >= restart:
    soln.output(params['data_dir'],
                'play-' + ("%05d" % i) + '.',
                init_disp)
    t += dt
    i += 1

        # if r < restart:
        #     pd.start_refine(soln.cur_disp)
        #     soln.start_refine()
        #     pd.execute_refine()
        #     soln.reinit()
        #     soln.post_refine(soln)
        #     break

c.proc0_out("From python: run complete")
c.kill()
