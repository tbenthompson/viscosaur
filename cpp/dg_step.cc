#include "dg_step.h"
#include "solution.h"
#include "problem_data.h"

#include <deal.II/numerics/vector_tools.h>


namespace viscosaur
{
    using namespace dealii;

    template <int dim>
    Stepper<dim>::Stepper(ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
    }

    template <int dim>
    void
    Stepper<dim>::step(Solution<dim> &soln,
                       InvViscosity<dim> &iv,
                       double inv_rho,
                       double dt,
                       Function<dim>& bc_fault,
                       Function<dim>& bc_plate)
    {
        TimerOutput::Scope t(pd->computing_timer, "dg_step");

        disp_constraints = pd->create_disp_constraints();
        mem_constraints = pd->create_mem_constraints();
        mem_constraints.close();

        //Boundary 0 is the fault, locked? or rate-state?, sliding at depth?
        //Boundary 1 is the far-field plate
        //Boundary 2 is the surface, should be stress free
        //Boundary 3 is the deep
        VectorTools::interpolate_boundary_values(pd->disp_dof_handler,
                0, bc_fault, disp_constraints);
        VectorTools::interpolate_boundary_values(pd->disp_dof_handler,
                1, bc_plate, disp_constraints);
        // VectorTools::interpolate_boundary_values(pd->vel_dof_handler,
        //         2, *encapsulated_bc, constraints);
        // VectorTools::interpolate_boundary_values(pd->vel_dof_handler,
        //         3, *encapsulated_bc, constraints);
        disp_constraints.close();

        std::vector<dealii::parallel::distributed::Vector<double>* > 
            sources(3);
        sources[0] = &soln.old_disp;
        sources[1] = &soln.old_old_disp;
        sources[2] = &soln.old_mem;
        
        EvalData<dim> data;
        double mu = bp::extract<double>(pd->parameters["shear_modulus"]);
        data.mu = make_vectorized_array(mu);
        data.inv_rho = make_vectorized_array(inv_rho);
        data.C = sqrt(mu / inv_rho);
        data.iv = &iv;
        data.dt = dt;

        //Calculate the time derivative that results from the FEM
        //semidiscretized equations. I could easily move this into a function
        //that runs inside a fancy time integrator allowing nice things like
        //high order runge kutta with adaptive time stepping
        //
        //Steps
        //1. Create the matrix free calculation 
        parallel::distributed::Vector<double> dsdt;
        //2. Initialize the derivative result variable and set it to zero
        pd->mem_matrix_free.initialize_dof_vector(dsdt);
        dsdt = 0;

        //3. Create the matrix free calculator. This object controls the mass
        // matrix and the hanging node constraints. It also directs the actual
        // calculations.
        MatrixFreeCalculation<dim> mfc(*pd, pd->mem_matrix_free, 
                mem_constraints, false);
        // 4. Set which operator to use.
        // The op factory construct allows the use
        // of a variable finite element degree. It produces an operator 
        // corresponding to the degree of element necessary
        MemEvalDerivFactory<dim> mem_op_factory;
        mfc.op_factory = &mem_op_factory;
        // //5. Finally, run the actual calculation.
        mfc.apply(dsdt, sources, &data);

        // Do it all again for the displacement equation instead of the stress.
        soln.cur_disp = 0;
        MatrixFreeCalculation<dim> mfc2(*pd, pd->disp_matrix_free, 
                disp_constraints, true);
        DispEvalDerivFactory<dim> disp_op_factory;
        mfc2.op_factory = &disp_op_factory;
        mfc2.apply(soln.cur_disp, sources, &data);


        //Forward euler is not the best...
        soln.cur_mem = 0;
        soln.cur_mem += dsdt;
        soln.cur_mem *= dt;
        soln.cur_mem += soln.old_mem;
    }

    template class Stepper<2>;
    template class Stepper<3>;
}
