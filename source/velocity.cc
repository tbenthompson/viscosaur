#include "velocity.h"
#include "problem_data.h"
#include "solution.h"
#include "scheme.h"
#include "boundary_cond.h"

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
    Velocity<dim>::Velocity(ProblemData<dim> &p_pd, 
                          Solution<dim> &soln,
                          BoundaryCond<dim> &bc,
                          Scheme<dim> &sch)
    {
        reinit(p_pd, soln, bc, sch);
    }

    template <int dim>
    Velocity<dim>::~Velocity()
    {
        // std::cout << "Destruction of Velocity." << std::endl;
    }

    template <int dim>
    void
    Velocity<dim>::
    reinit(ProblemData<dim> &p_pd,
           Solution<dim> &soln,
           BoundaryCond<dim> &bc,
           Scheme<dim> &sch)
    {
        pd = &p_pd;
        pd->pcout << "Setting up the Velocity solver." << std::endl;
        setup_system(bc, soln, sch);
        // assemble_matrix(soln);
        pd->pcout << "   Number of active cells:       "
            << pd->triangulation.n_global_active_cells()
            << std::endl
            << "   Number of degrees of freedom: "
            << pd->vel_dof_handler.n_dofs()
            << std::endl;
    }


    template <int dim>
    void Velocity<dim>::setup_system(BoundaryCond<dim> &bc, Solution<dim> &soln,
            Scheme<dim> &sch)
    {
        //The theme in this function is that only the locally relevant or 
        //locally owned dofs will be made known to any given processor.
        TimerOutput::Scope t(pd->computing_timer, "setup");

        update_bc(bc, sch);
        CompressedSimpleSparsityPattern* csp = 
            pd->create_vel_sparsity_pattern(constraints);
        // Initialize the matrix, rhs and solution vectors.
        // Ax = b, 
        // where system_rhs is b, system_matrix is A, 
        // is A
        system_rhs.reinit(pd->vel_locally_owned_dofs, pd->mpi_comm);
        system_rhs = 0;
        system_matrix.reinit(pd->vel_locally_owned_dofs,
                             pd->vel_locally_owned_dofs,
                             *csp,
                             pd->mpi_comm);
        //GET RID OF MANUAL POINTER HANDLING!
        delete csp;
    }

    template <int dim>
    void Velocity<dim>::update_bc(BoundaryCond<dim> &bc, Scheme<dim> &sch)
    {
        constraints = *pd->create_vel_constraints();
        Function<dim>* encapsulated_bc = sch.handle_bc(bc);
        VectorTools::interpolate_boundary_values(pd->vel_dof_handler,
                0, *encapsulated_bc, constraints);
        VectorTools::interpolate_boundary_values(pd->vel_dof_handler,
                1, *encapsulated_bc, constraints);
        // VectorTools::interpolate_boundary_values(pd->vel_dof_handler,
        //         2, *encapsulated_bc, constraints);
        VectorTools::interpolate_boundary_values(pd->vel_dof_handler,
                3, *encapsulated_bc, constraints);
        constraints.close();
    }


    template <int dim>
    void 
    Velocity<dim>::
    assemble_matrix(Solution<dim> &soln, Scheme<dim> &sch, double time_step)
    { 
        TimerOutput::Scope t(pd->computing_timer, "assem_mat");
        FEValues<dim> vel_fe_values(pd->vel_fe, pd->quadrature, 
                                update_values | update_gradients | 
                                update_JxW_values); 

        FEValues<dim> strs_fe_values(pd->strs_fe, pd->quadrature, 
                                update_values); 

        FEFaceValues<dim> fe_face_values(pd->vel_fe, pd->face_quad,
                                  update_values | update_quadrature_points |
                                  update_normal_vectors | update_JxW_values);

        const FEValuesExtractors::Vector strses(0);

        const unsigned int   dofs_per_cell = pd->vel_fe.dofs_per_cell;
        const unsigned int   n_q_points    = pd->quadrature.size();
        const unsigned int n_face_q_points = pd->face_quad.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
            cell = pd->vel_dof_handler.begin_active(),
            endc = pd->vel_dof_handler.end();

        double value;
        const double shear_modulus = 
            bp::extract<double>(pd->parameters["shear_modulus"]);
        const double factor = sch.poisson_rhs_factor() / 
            (shear_modulus * time_step);

        std::vector<Tensor<1, dim> > strs_val(n_q_points);
        std::vector<Tensor<1, dim> > grad(dofs_per_cell);
        std::vector<double> val(dofs_per_cell);
        double JxW;
        for (; cell!=endc; ++cell)
        {
            typename DoFHandler<dim>::active_cell_iterator
                  cell_strs (&pd->triangulation,
                    cell->level(),
                    cell->index(),
                    &pd->strs_dof_handler);
            if (!cell->is_locally_owned())
            {
                continue;
            }
            // TimerOutput::Scope t(pd->computing_timer, "assem.cell_one");
            // TimerOutput::Scope t2(pd->computing_timer, "cell_construction");
            cell_matrix = 0;
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            vel_fe_values.reinit(cell);

            strs_fe_values.reinit(cell_strs);
            strs_fe_values[strses].get_function_values(soln.tent_strs, 
                                                            strs_val);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                JxW = vel_fe_values.JxW(q);
                for (unsigned int i=0; i < dofs_per_cell; ++i)
                {
                    grad[i] = vel_fe_values.shape_grad(i, q);
                    val[i] = vel_fe_values.shape_value(i, q);
                }
                // This pair of loops is symmetric. I cut the assembly
                // cost in half by taking advantage of this.
                for (unsigned int i=0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j <= i; ++j)
                    {
                    // The main matrix entries are the integral of product of 
                    // the gradient of the shape functions. We also must accnt
                        // for the mapping between the element and the unit 
                        // element and the size of the element
                        cell_matrix(i,j) += grad[i] * grad[j] * JxW;
                    } 
                    cell_rhs(i) -= factor * strs_val[q] * grad[i] * JxW; 
                } 

            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    cell_matrix(i, j) = cell_matrix(j, i);


            // Compute the contribution of the face terms resulting from the
            // integration by parts of the divergence on the rhs
            for (unsigned int face = 0; 
                    face < GeometryInfo<dim>::faces_per_cell; 
                    ++face)
            {
                if (!cell->face(face)->at_boundary())
                {
                    continue;
                }
                fe_face_values.reinit (cell, face);
                for (unsigned int q= 0; q< n_face_q_points; ++q)
                {
                    const double surf_rhs_value
                        = factor * strs_val[q] * 
                            fe_face_values.normal_vector(q);
                    for (unsigned int i=0; i < dofs_per_cell; ++i)
                    {
                        cell_rhs(i) += (surf_rhs_value *
                            fe_face_values.shape_value(i, q) *
                            fe_face_values.JxW(q));
                    }
                }
            }

            constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix, system_rhs);
        }
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }

    template <int dim>
    void 
    Velocity<dim>::
    assemble_rhs(Solution<dim> &soln, Scheme<dim> &sch, double time_step)
    { 
        // I should separate matrix and rhs construction so they are 
        // independent. But, wait for the moment, not necessary!
        assemble_matrix(soln, sch, time_step);
    }



    template <int dim>
    void Velocity<dim>::solve(Solution<dim> &soln, Scheme<dim> &sch)
    {
        TimerOutput::Scope t(pd->computing_timer, "solve");
        LA::MPI::Vector
            completely_distributed_solution(pd->vel_locally_owned_dofs,
                                             pd->mpi_comm);

        const double tol = 1e-8 * system_rhs.l2_norm();
        SolverControl solver_control(pd->vel_dof_handler.n_dofs(), tol);

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

        sch.handle_poisson_soln(soln, completely_distributed_solution);
    }


    template <int dim>
    void
    Velocity<dim>::
    step(Solution<dim> &soln, Scheme<dim> &sch, double time_step)
    {
        assemble_rhs(soln, sch, time_step);
        solve(soln, sch);
    }

    // ESSENTIAL: explicity define the template types we will use.
    // Otherwise, the template definition needs to go in the header file, which
    // is ugly!
    template class Velocity<2>;
    template class Velocity<3>;
}
