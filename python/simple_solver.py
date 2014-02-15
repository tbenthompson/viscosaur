import viscosaur as vc

class SimpleSolver(object):
    def __init__(self, params, inv_visc, vel_bc, c):
        self.params = params
        self.inv_visc = inv_visc
        self.vel_bc = vel_bc
        self.c = c
        self.pd = vc.ProblemData2D(self.params, self.inv_visc)
        self.soln = vc.Solution2D(self.pd)
        self.scheme = vc.FwdEuler2D(self.pd)
        self.vel_solver = vc.Velocity2D(self.pd, self.soln,
                                        self.vel_bc, self.scheme)
        self.strs_solver = vc.Stress2D(self.pd)

    def step(self, time_step):
        self.soln.start_timestep()
        self.strs_solver.tentative_step(self.soln, self.scheme, time_step)
        self.vel_solver.step(self.soln, self.scheme, time_step)
        self.strs_solver.correction_step(self.soln, self.scheme, time_step)

    def refine(self):
        self.pd.start_refine(self.soln.current_velocity)
        self.soln.start_refine()
        self.pd.execute_refine()
        self.soln.reinit()
        self.soln.post_refine(self.soln)
        self.scheme.reinit(self.pd);
        self.vel_solver.reinit(self.pd, self.soln, self.vel_bc, self.scheme)
        self.strs_solver.reinit(self.pd);

    def run(self, init_strs, init_vel, exact_vel):
        sub_timesteps = self.params['first_substeps']
        if not self.params["load_mesh"]:
            time_step = self.params['time_step'] / sub_timesteps
            self.vel_bc.set_t(time_step)
            exact_vel.set_t(time_step)
            for i in range(self.params['initial_adaptive_refines']):
                self.soln.apply_init_cond(init_strs, init_vel)
                self.step(time_step)
                if self.params['output']:
                    self.soln.output(self.params['data_dir'], 'init_refinement_' +
                                str(i) + '.', exact_vel)
                self.refine()
            self.c.proc0_out("Done with first time step spatial adaptation.")
            self.pd.save_mesh("saved_mesh.msh")

        self.soln.apply_init_cond(init_strs, init_vel)
        t = 0
        i = 1
        while t < self.params['t_max']:
            time_step = self.params['time_step'] / sub_timesteps
            for sub_t in range(0, sub_timesteps):
                t += time_step
                self.c.proc0_out("\n\nSolving for time = " + \
                          str(t / self.params['secs_in_a_year']) + " \n")
                self.vel_bc.set_t(t)
                self.vel_solver.update_bc(self.vel_bc, self.scheme)
                self.step(time_step)
                exact_vel.set_t(t)
                filename = "solution_" + str(i) + "."
                if self.params['output']:
                    self.soln.output(self.params['data_dir'], filename, exact_vel)
                # self.refine()
            if i == 1:
                # At the end of the first time step, we switch to using a BDF2 scheme
                sub_timesteps = 1
                self.soln.init_multistep(init_strs, init_vel)
                self.scheme = vc.BDFTwo2D(self.pd)
            i += 1
