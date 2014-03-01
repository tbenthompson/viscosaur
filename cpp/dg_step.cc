#include "dg_step.h"
#include "problem_data.h"
#include "solution.h"

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>


namespace viscosaur
{
    using namespace dealii;

    template <int dim>
    DGStep<dim>::DGStep(ProblemData<dim> &pd,
                        double dt)
    {
        this->dt = dt; 

        MeshWorker::IntegrationInfoBox<dim>& info_box = rhs_integrator.info_box;
        MeshWorker::DoFInfo<dim>& dof_info = rhs_integrator.dof_info;
        MeshWorker::Assembler::ResidualSimple<Vector<double> >&
          assembler = rhs_integrator.assembler;

        const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
        info_box.cell_quadrature = pd.quadrature;
        info_box.face_quadrature = pd.face_quad;
        info_box.boundary_quadrature = pd.face_quad;

        // Add solution vector to info_box
        NamedData< Vector<double>* > solution_data;
        solution_data.add (&solution, "solution");
        info_box.cell_selector.add     ("solution", true, false, false);
        info_box.boundary_selector.add ("solution", true, false, false);
        info_box.face_selector.add     ("solution", true, false, false);

        info_box.initialize_update_flags ();
        info_box.add_update_flags_all      (update_quadrature_points);
        info_box.add_update_flags_cell     (update_gradients);
        info_box.add_update_flags_boundary (update_values);
        info_box.add_update_flags_face     (update_values);

        info_box.initialize (fe, mapping, solution_data);

        // Attach rhs vector to assembler
        NamedData<Vector<double>* > rhs;
        rhs.add (&right_hand_side, "RHS");
        assembler.initialize (rhs);
    }

    template <int dim>
    void
    DGStep<dim>::step(ProblemData<dim> &pd,
                      Solution<dim> &soln,
                      InvViscosity<dim> &iv,
                      double inv_rho)
    {
        TimerOutput::Scope t(pd.computing_timer, "dg_step");

        std::vector<dealii::parallel::distributed::Vector<double>* > 
            sources(2);
        sources[0] = &soln.old_vel;
        sources[1] = &soln.old_strs;
        
        EvalData<dim> data;
        double mu = bp::extract<double>(pd.parameters["shear_modulus"]);
        data.mu = make_vectorized_array(mu);
        data.inv_rho = make_vectorized_array(inv_rho);
        data.C = sqrt(mu / inv_rho);
        data.iv = &iv;

        //Calculate the time derivative that results from the DG-FEM
        //semidiscretized equations. I could easily move this into a function
        //that runs inside a fancy time integrator allowing nice things like
        //high order runge kutta with adaptive time stepping
        //
        //Steps
        //1. Create the matrix free calculation 
        parallel::distributed::Vector<double> dsdt;
        //2. Initialize the derivative result variable and set it to zero
        pd.strs_matrix_free.initialize_dof_vector(dsdt);
        dsdt = 0;

        //3. Create the matrix free calculator. This object controls the mass
        // matrix and the hanging node constraints. It also directs the actual
        // calculations.
        MatrixFreeCalculation<dim> mfc(pd, pd.strs_matrix_free, 
                pd.strs_hanging_node_constraints, false);
        //4. Set which operator to use. The op factory construct allows the use
        // of a variable finite element degree. It produces an operator 
        // corresponding to the degree of element necessary
        StrsEvalDerivFactory<dim> strs_op_factory;
        mfc.op_factory = &strs_op_factory;
        //5. Finally, run the actual calculation.
        mfc.apply(dsdt, sources, &data);

        // Do it all again for the velocity equation instead of the stress.
        parallel::distributed::Vector<double> dvdt;
        pd.vel_matrix_free.initialize_dof_vector(dvdt);
        dvdt = 0;
        MatrixFreeCalculation<dim> mfc2(pd, pd.vel_matrix_free, 
                pd.vel_hanging_node_constraints, true);
        VelEvalDerivFactory<dim> vel_op_factory;
        mfc2.op_factory = &vel_op_factory;
        mfc2.apply(dvdt, sources, &data);


        //Forward euler is not the best...
        soln.cur_strs = 0;
        soln.cur_vel = 0;
        soln.cur_strs += dsdt;
        soln.cur_vel += dvdt;
        soln.cur_strs *= dt;
        soln.cur_vel *= dt;
        soln.cur_strs += soln.old_strs;
        soln.cur_vel += soln.old_vel;
    }

    template class DGStep<2>;
    template class DGStep<3>;
}
