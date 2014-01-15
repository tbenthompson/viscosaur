#include "poisson.h"
#include "problem_data.h"
#include "one_step_rhs.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>

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
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <Python.h>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>


namespace viscosaur
{
    namespace bp = boost::python;

    template<int dim>
    DoFHandler<dim>* Poisson<dim>::get_dof_handler()
    {
        return &pd->dof_handler;
    }

    template <int dim>
    Poisson<dim>::Poisson (ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        pd->pcout << "Setting up the Poisson solver." << std::endl;
        setup_system();
    }



    template <int dim>
    Poisson<dim>::~Poisson ()
    {
        pd->dof_handler.clear();
    }



    template <int dim>
    void Poisson<dim>::setup_system ()
    {
        //The theme in this function is that only the locally relevant or 
        //locally owned dofs will be made known to any given processor.
        TimerOutput::Scope t(pd->computing_timer, "setup");


        // Constrain hanging nodes and boundary conditions.
        constraints = *pd->create_constraints();
        VectorTools::interpolate_boundary_values(pd->dof_handler,
                0,
                ZeroFunction<dim>(),
                constraints);
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
    void Poisson<dim>::fill_cell_matrix(FullMatrix<double> &cell_matrix,
                                        FEValues<dim> &fe_values,
                                        const unsigned int n_q_points,
                                        const unsigned int dofs_per_cell)
    {
        for (unsigned int q_point=0; q_point < n_q_points; ++q_point)
        {
            // This pair of loops is symmetric. I could cut the assembly
            // cost in half by taking advantage of this.
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    // The main matrix entries are the integral of product of 
                    // the gradient of the shape functions. We also must accnt
                    // for the mapping between the element and the unit 
                    // element and the size of the element
                    // Note: for a 12th order method, this = 0 in 88% of
                    // cases. Efficiency improvment?
                    cell_matrix(i,j) += (fe_values.shape_grad(i, q_point) *
                            fe_values.shape_grad(j, q_point) * 
                            fe_values.JxW(q_point)); 
                } 
            } 
        } 
    } 
    
    template <int dim>
    void 
    Poisson<dim>::
    assemble_system(PoissonRHS<dim>* rhs) 
    { 
        TimerOutput::Scope t(pd->computing_timer, "assembly");
        const unsigned int fe_d = bp::extract<int>(pd->parameters["fe_degree"]);
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

        rhs->start_assembly();

        for (; cell!=endc; ++cell)
        {
            if (!cell->is_locally_owned())
            {
                continue;
            }
            TimerOutput::Scope t2(pd->computing_timer, "cell_construction");
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);
            rhs->fill_cell_rhs(cell_rhs, fe_values, n_q_points, dofs_per_cell);
            fill_cell_matrix(cell_matrix, fe_values, n_q_points, dofs_per_cell);

            cell->get_dof_indices(local_dof_indices);
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
            completely_distributed_solution (pd->locally_owned_dofs, 
                                             pd->mpi_comm);

        SolverControl solver_control (pd->dof_handler.n_dofs(), 1e-12);

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
    void Poisson<dim>::output_results (const unsigned int cycle) const
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(pd->dof_handler);
        data_out.add_data_vector(locally_relevant_solution, "u");

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
    LA::MPI::Vector Poisson<dim>::run (PoissonRHS<dim>* rhs)
    {
        const unsigned int n_cycles = 5;
        for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
        {

            if (cycle != 0)
            {
                pd->refine_grid(locally_relevant_solution);
                setup_system();
                //
                // Build the matrices
            }
            pd->pcout << "Cycle " << cycle << ':' << std::endl;

            assemble_system(rhs);

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
                output_results (cycle);
            }

            pd->pcout << std::endl;
            pd->computing_timer.print_summary ();
            pd->computing_timer.reset ();
        }
        return locally_relevant_solution;
    }

    //Explicitly define the two types of templates we will use
    template class Poisson<2>;
    template class Poisson<3>;
}
