#include "poisson.h"
#include "problem_data.h"
#include "solution.h"
#include "analytic.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/utilities.h>
#include <deal.II/lac/sparsity_tools.h>

#include <Python.h>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>


namespace viscosaur
{
    using namespace dealii;
    namespace bp = boost::python;

    template <int dim>
    InvViscosity<dim>::InvViscosity(ProblemData<dim> &p_pd){
        layer_depth = 
            bp::extract<double>(p_pd.parameters["fault_depth"]);
        inv_viscosity = 1.0 /
            bp::extract<double>(p_pd.parameters["viscosity"]);
    }

    template <int dim>
    Velocity<dim>::Velocity(Solution<dim> &soln,
                          dealii::Function<dim> &bc,
                          ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        pd->pcout << "Setting up the Velocity solver." << std::endl;
        setup_system(bc, soln);
        assemble_matrix(soln);
        pd->pcout << "   Number of active cells:       "
            << pd->triangulation.n_global_active_cells()
            << std::endl
            << "   Number of degrees of freedom: "
            << pd->dof_handler.n_dofs()
            << std::endl;
    }



    template <int dim>
    void Velocity<dim>::setup_system(Function<dim> &bc, Solution<dim> &soln)
    {
        //The theme in this function is that only the locally relevant or 
        //locally owned dofs will be made known to any given processor.
        TimerOutput::Scope t(pd->computing_timer, "setup");


        // Constrain hanging nodes and boundary conditions.
        hanging_node_constraints = *pd->create_constraints();
        hanging_node_constraints.close();

        constraints = *pd->create_constraints();
        VectorTools::interpolate_boundary_values(pd->dof_handler,
                0, bc, constraints);
        VectorTools::interpolate_boundary_values(pd->dof_handler,
                1, bc, constraints);
        VectorTools::interpolate_boundary_values(pd->dof_handler,
                3, bc, constraints);
        constraints.close();

        CompressedSimpleSparsityPattern* csp = 
            pd->create_sparsity_pattern(constraints);
        // Initialize the matrix, rhs and solution vectors.
        // Ax = b, 
        // where system_rhs is b, system_matrix is A, locally_relevant_solution 
        // is A
        system_rhs.reinit(pd->locally_owned_dofs, pd->mpi_comm);
        system_rhs = 0;
        system_matrix.reinit(pd->locally_owned_dofs,
                             pd->locally_owned_dofs,
                             *csp,
                             pd->mpi_comm);
        //GET RID OF MANUAL POINTER HANDLING!
        delete csp;
    }



    template <int dim>
    void 
    Velocity<dim>::
    assemble_matrix(Solution<dim> &soln) 
    { 
        TimerOutput::Scope t(pd->computing_timer, "assem_mat");
        FEValues<dim> fe_values(pd->fe, pd->quadrature, 
                                update_values | update_gradients | 
                                update_quadrature_points | update_JxW_values); 

        const unsigned int   dofs_per_cell = pd->fe.dofs_per_cell;
        const unsigned int   n_q_points    = pd->quadrature.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
            cell = pd->dof_handler.begin_active(),
            endc = pd->dof_handler.end();


        std::vector<Tensor<1, dim> > grad(dofs_per_cell);
        double JxW;
        for (; cell!=endc; ++cell)
        {
            if (!cell->is_locally_owned())
            {
                continue;
            }
            // TimerOutput::Scope t(pd->computing_timer, "assem.cell_one");
            // TimerOutput::Scope t2(pd->computing_timer, "cell_construction");
            cell_matrix = 0;

            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                JxW = fe_values.JxW(q);
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    grad[i] = fe_values.shape_grad(i, q);
                }
                // This pair of loops is symmetric. I cut the assembly
                // cost in half by taking advantage of this.
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j <= i; ++j)
                    {
                        // The main matrix entries are the integral of product of 
                        // the gradient of the shape functions. We also must accnt
                        // for the mapping between the element and the unit 
                        // element and the size of the element
                        cell_matrix(i,j) += grad[i] * grad[j] * JxW;
                    } 
                } 

            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    cell_matrix(i, j) = cell_matrix(j, i);
            constraints.distribute_local_to_global(cell_matrix,
                                                   local_dof_indices,
                                                   system_matrix);
        }
        system_matrix.compress(VectorOperation::add);
    }

    template <int dim>
    void 
    Velocity<dim>::
    assemble_rhs(Solution<dim> &soln) 
    { 
        TimerOutput::Scope t(pd->computing_timer, "assem_rhs");
        FEValues<dim> fe_values(pd->fe, pd->quadrature, 
                                update_values | update_gradients | 
                                update_quadrature_points | update_JxW_values); 

        const unsigned int   dofs_per_cell = pd->fe.dofs_per_cell;
        const unsigned int   n_q_points    = pd->quadrature.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
            cell = pd->dof_handler.begin_active(),
            endc = pd->dof_handler.end();

        double value;
        const double shear_modulus = 
            bp::extract<double>(pd->parameters["shear_modulus"]);
        const double time_step = 
            bp::extract<double>(pd->parameters["time_step"]);
        const double factor = 1.0 / (shear_modulus * time_step);

        std::vector<Tensor<1, dim> > Szxgrad(n_q_points);
        std::vector<Tensor<1, dim> > Szygrad(n_q_points);
        std::vector<Tensor<1, dim> > grad(dofs_per_cell);
        std::vector<double> val(dofs_per_cell);
        double JxW;
        for (; cell!=endc; ++cell)
        {
            if (!cell->is_locally_owned())
            {
                continue;
            }
            // TimerOutput::Scope t(pd->computing_timer, "assem.cell_one");
            // TimerOutput::Scope t2(pd->computing_timer, "cell_construction");
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            fe_values.get_function_gradients(soln.tent_szx, Szxgrad);
            fe_values.get_function_gradients(soln.tent_szy, Szygrad);


            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                JxW = fe_values.JxW(q);
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    if (constraints.
                        is_inhomogeneously_constrained(local_dof_indices[i]))
                    {
                        grad[i] = fe_values.shape_grad(i, q);
                    }
                    val[i] = fe_values.shape_value(i, q);
                }
                // This pair of loops is symmetric. I cut the assembly
                // cost in half by taking advantage of this.
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {

                    if (constraints.
                        is_inhomogeneously_constrained(local_dof_indices[i]))
                    {
                        for (unsigned int j = 0; j <= i; ++j)
                        {
                            // The main matrix entries are the integral of product of 
                            // the gradient of the shape functions. We also must accnt
                            // for the mapping between the element and the unit 
                            // element and the size of the element
                            cell_matrix(i,j) += grad[i] * grad[j] * JxW;
                        } 
                    }
                    cell_rhs(i) += factor * JxW * 
                        (Szxgrad[q][0] + Szygrad[q][1]) * val[i];
                } 

            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                  cell_matrix(i, j) = cell_matrix(j, i);
            constraints.distribute_local_to_global(cell_rhs,
                                                      local_dof_indices,
                                                      system_rhs,
                                                      cell_matrix);
        }
        system_rhs.compress(VectorOperation::add);
    }



    template <int dim>
    void Velocity<dim>::solve(Solution<dim> &soln)
    {
        TimerOutput::Scope t(pd->computing_timer, "solve");
        LA::MPI::Vector
            completely_distributed_solution(pd->locally_owned_dofs, 
                                             pd->mpi_comm);

        const double tol = 1e-8 * system_rhs.l2_norm();
        SolverControl solver_control(pd->dof_handler.n_dofs(), tol);

        LA::SolverCG solver(solver_control, pd->mpi_comm);
        LA::MPI::PreconditionAMG preconditioner;

        LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
        data.symmetric_operator = true;
#else
#endif
        preconditioner.initialize(system_matrix, data);

        solver.solve(system_matrix, 
                     completely_distributed_solution, 
                     system_rhs,
                     preconditioner);

        pd->pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

        constraints.distribute(completely_distributed_solution);

        soln.cur_vel = completely_distributed_solution;
    }


    template <int dim>
    void Velocity<dim>::step(Solution<dim> &soln)
    {
        assemble_rhs(soln);
        solve(soln);
    }

    // ESSENTIAL: explicity define the template types we will use.
    // Otherwise, the template definition needs to go in the header file, which
    // is ugly!
    template class Velocity<2>;
    template class Velocity<3>;
    template class InvViscosity<2>;
    template class InvViscosity<3>;
}
