#include "poisson.h"
#include "problem_data.h"
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
#include <deal.II/numerics/data_out.h>

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
    Poisson<dim>::Poisson(dealii::Function<dim> &p_init_cond_Szx,
                          dealii::Function<dim> &p_init_cond_Szy,
                          ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        init_cond_Szx = &p_init_cond_Szx;
        init_cond_Szy = &p_init_cond_Szy;
        pd->pcout << "Setting up the Poisson solver." << std::endl;
        // start_assembly();
        // DataOut<dim> data_out;
        // data_out.attach_dof_handler(dof_handler);
        // data_out.add_data_vector(Szx, "Szx"); 
        // data_out.add_data_vector(Szy, "Szy"); 
        // data_out.build_patches();

        // std::ofstream output("abcdef.vtk");
        // data_out.write_vtk(output);
    }



    template <int dim>
    void Poisson<dim>::setup_system(Function<dim> &bc)
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
        locally_relevant_solution.reinit(pd->locally_owned_dofs,
                pd->locally_relevant_dofs, pd->mpi_comm);
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
    Poisson<dim>::
    assemble_system() 
    { 
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
        LA::MPI::Vector Szx;
        LA::MPI::Vector Szy;
        LA::MPI::Vector inv_visc;
        InvViscosity<dim> fnc(*pd);
        {
            // TimerOutput::Scope t(pd->computing_timer, "assem.interpolation");

            Szx.reinit(pd->locally_owned_dofs, pd->mpi_comm);
            VectorTools::project(pd->dof_handler, 
                                 hanging_node_constraints,
                                 pd->quadrature,
                                 *init_cond_Szx,
                                 Szx);
            // Szx.compress(VectorOperation::add);

            Szy.reinit(pd->locally_owned_dofs, pd->mpi_comm);
            VectorTools::project(pd->dof_handler, 
                                 hanging_node_constraints,
                                 pd->quadrature,
                                 *init_cond_Szy,
                                 Szy);
            // Szy.compress(VectorOperation::add);

            inv_visc.reinit(pd->locally_owned_dofs, pd->mpi_comm);
            VectorTools::project(pd->dof_handler, 
                                 hanging_node_constraints,
                                 pd->quadrature,
                                 fnc,
                                 inv_visc);
            inv_visc.compress(VectorOperation::add);
            inv_visc *= -shear_modulus * time_step;
            inv_visc.add(1.0);
        }
        Szx.scale(inv_visc);
        Szy.scale(inv_visc);

        std::vector<Tensor<1, dim> > grad(dofs_per_cell);
        std::vector<double> val(dofs_per_cell);
        std::vector<Tensor<1, dim> > Szxgrad(n_q_points);
        std::vector<Tensor<1, dim> > Szygrad(n_q_points);
        double JxW;
        TimerOutput::Scope t(pd->computing_timer, "assem");
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
            fe_values.get_function_gradients(Szx, Szxgrad);
            fe_values.get_function_gradients(Szy, Szygrad);


            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                JxW = fe_values.JxW(q);
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    grad[i] = fe_values.shape_grad(i, q);
                    val[i] = fe_values.shape_value(i, q);
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
                    cell_rhs(i) += factor * JxW * 
                        (Szxgrad[q][0] + Szygrad[q][1]) * val[i];
                } 

            }
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                  cell_matrix(i, j) = cell_matrix(j, i);
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
        }
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }




    template <int dim>
    void Poisson<dim>::solve ()
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

        locally_relevant_solution = completely_distributed_solution;
    }






    template <int dim>
    std::string Poisson<dim>::output_filename(const unsigned int cycle,
            const unsigned int subdomain) const
    {
        std::string filename = "solution-" +
                Utilities::int_to_string(cycle, 2) +
                "." +
                Utilities::int_to_string(subdomain, 4);
        return filename;
    }

    template <int dim>
    void Poisson<dim>::output_results (const unsigned int cycle,
                                       Function<dim> &exact) const
    {
        LA::MPI::Vector vel;

        vel.reinit(pd->locally_owned_dofs, pd->mpi_comm);
        VectorTools::interpolate(pd->dof_handler, exact, vel);
        vel.compress(VectorOperation::add);

        Vector<double> local_errors(
                pd->triangulation.n_active_cells());
        VectorTools::integrate_difference(
                pd->dof_handler,
                locally_relevant_solution,
                exact,
                local_errors,
                QGauss<dim>(3),
                VectorTools::L2_norm);

        const double total_local_error = local_errors.l2_norm();
        const double total_global_error = std::sqrt (
                dealii::Utilities::MPI::sum (
                    total_local_error * 
                    total_local_error, pd->mpi_comm));
        pd->pcout << "Total exact error: " << total_global_error << std::endl;

        DataOut<dim> data_out;
        data_out.attach_dof_handler(pd->dof_handler);
        data_out.add_data_vector(locally_relevant_solution, "u");
        data_out.add_data_vector(vel, "vel");
        data_out.add_data_vector(local_errors, "error");

        Vector<float> subdomain(pd->triangulation.n_active_cells());
        unsigned int this_subd = pd->triangulation.locally_owned_subdomain();
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = this_subd;
        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::string this_f = output_filename(cycle, this_subd);
        std::ofstream output(("data/" + this_f + ".vtu").c_str());
        data_out.write_vtu(output);

        if (Utilities::MPI::this_mpi_process(pd->mpi_comm) == 0)
        {
            // Build the list of filenames to store in the master pvtu file
            std::vector<std::string> filenames;
            for (unsigned int i=0;
                    i<Utilities::MPI::n_mpi_processes(pd->mpi_comm);
                    ++i)
            {
                std::string f = output_filename(cycle, i);
                filenames.push_back (f + ".vtu");
            }

            std::ofstream master_output(("data/" + this_f + ".pvtu").c_str());
            // Write the master pvtu record. Load this pvtu file if you want
            // to view all the data at once. Use a tool like visit or paraview.
            data_out.write_pvtu_record(master_output, filenames);
        }
    }




    template <int dim>
    LA::MPI::Vector Poisson<dim>::run(Function<dim> &bc)
    {
        const unsigned int n_cycles =
            bp::extract<int>(pd->parameters["initial_adaptive_refines"]);
        for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
        {
            if (cycle != 0)
            {
                pd->refine_grid(locally_relevant_solution);
                //
                // Build the matrices
            }
            setup_system(bc);
            pd->pcout << "Cycle " << cycle << ':' << std::endl;

            assemble_system();

            pd->pcout << "   Number of active cells:       "
                << pd->triangulation.n_global_active_cells()
                << std::endl
                << "   Number of degrees of freedom: "
                << pd->dof_handler.n_dofs()
                << std::endl;

            // Solve the linear system.
            solve();

            if (Utilities::MPI::n_mpi_processes(pd->mpi_comm) <= 32)
            {
                TimerOutput::Scope t(pd->computing_timer, "output");
                output_results(cycle, bc);
            }

            pd->pcout << std::endl;
            pd->computing_timer.print_summary ();
            pd->computing_timer.reset ();
        }
        return locally_relevant_solution;
    }

    // ESSENTIAL: explicity define the template types we will use.
    // Otherwise, the template definition needs to go in the header file, which
    // is ugly!
    template class Poisson<2>;
    template class Poisson<3>;
}
