// All includes are in core.h 
// This may be bad coding practice but it's much cleaner to deal with the
// file this way.
#include "core.h"

namespace viscosaur
{
    template <int dim>
    Controller<dim>::Controller ()
    :
        mpi_communicator (MPI_COMM_WORLD),
        triangulation (mpi_communicator,
                typename Triangulation<dim>::MeshSmoothing
                (Triangulation<dim>::smoothing_on_refinement |
                 Triangulation<dim>::smoothing_on_coarsening)),
        dof_handler (triangulation),
        fe (2),
        pcout (std::cout,
                (Utilities::MPI::this_mpi_process(mpi_communicator)
                 == 0)),
        computing_timer (pcout,
                TimerOutput::summary,
                TimerOutput::wall_times)
    {
    }



    template <int dim>
    Controller<dim>::~Controller ()
    {
        dof_handler.clear ();
    }



    template <int dim>
    void Controller<dim>::setup_system ()
    {
        TimerOutput::Scope t(computing_timer, "setup");

        dof_handler.distribute_dofs (fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs ();
        DoFTools::extract_locally_relevant_dofs (dof_handler,
                locally_relevant_dofs);

        locally_relevant_solution.reinit (locally_owned_dofs,
                locally_relevant_dofs, mpi_communicator);
        system_rhs.reinit (locally_owned_dofs, mpi_communicator);

        system_rhs = 0;

        constraints.clear ();
        constraints.reinit (locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
        VectorTools::interpolate_boundary_values (dof_handler,
                0,
                ZeroFunction<dim>(),
                constraints);
        constraints.close ();

        CompressedSimpleSparsityPattern csp (locally_relevant_dofs);

        DoFTools::make_sparsity_pattern (dof_handler, csp,
                constraints, false);
        SparsityTools::distribute_sparsity_pattern (csp,
                dof_handler.n_locally_owned_dofs_per_processor(),
                mpi_communicator,
                locally_relevant_dofs);

        system_matrix.reinit (locally_owned_dofs,
                locally_owned_dofs,
                csp,
                mpi_communicator);
    }




    template <int dim>
    void Controller<dim>::assemble_system ()
    {
        TimerOutput::Scope t(computing_timer, "assembly");

        const QGauss<dim>  quadrature_formula(3);

        FEValues<dim> fe_values (fe, quadrature_formula,
                update_values    |  update_gradients |
                update_quadrature_points |
                update_JxW_values);

        const unsigned int   dofs_per_cell = fe.dofs_per_cell;
        const unsigned int   n_q_points    = quadrature_formula.size();

        FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       cell_rhs (dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
                 endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            if (cell->is_locally_owned())
            {
                cell_matrix = 0;
                cell_rhs = 0;

                fe_values.reinit (cell);

                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                    const double
                        rhs_value
                        = (fe_values.quadrature_point(q_point)[1]
                                >
                                0.5+0.25*std::sin(4.0 * numbers::PI *
                                    fe_values.quadrature_point(q_point)[0])
                                ? 1 : -1);

                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                            cell_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                                    fe_values.shape_grad(j,q_point) *
                                    fe_values.JxW(q_point));

                        cell_rhs(i) += (rhs_value *
                                fe_values.shape_value(i,q_point) *
                                fe_values.JxW(q_point));
                    }
                }

                cell->get_dof_indices (local_dof_indices);
                constraints.distribute_local_to_global (cell_matrix,
                        cell_rhs,
                        local_dof_indices,
                        system_matrix,
                        system_rhs);
            }
        }

        system_matrix.compress (VectorOperation::add);
        system_rhs.compress (VectorOperation::add);
    }




    template <int dim>
    void Controller<dim>::solve ()
    {
        TimerOutput::Scope t(computing_timer, "solve");
        LA::MPI::Vector
            completely_distributed_solution (locally_owned_dofs, mpi_communicator);

        SolverControl solver_control (dof_handler.n_dofs(), 1e-12);

        LA::SolverCG solver(solver_control, mpi_communicator);
        LA::MPI::PreconditionAMG preconditioner;

        LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
        data.symmetric_operator = true;
#else
#endif
        preconditioner.initialize(system_matrix, data);

        solver.solve (system_matrix, completely_distributed_solution, system_rhs,
                preconditioner);

        pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

        constraints.distribute (completely_distributed_solution);

        locally_relevant_solution = completely_distributed_solution;
    }




    template <int dim>
    void Controller<dim>::refine_grid ()
    {
        TimerOutput::Scope t(computing_timer, "refine");

        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate (dof_handler,
                QGauss<dim-1>(3),
                typename FunctionMap<dim>::type(),
                locally_relevant_solution,
                estimated_error_per_cell);
        parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_number (triangulation,
                    estimated_error_per_cell,
                    0.3, 0.03);
        triangulation.execute_coarsening_and_refinement ();
    }




    template <int dim>
    void Controller<dim>::output_results (const unsigned int cycle) const
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (locally_relevant_solution, "u");

        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");

        data_out.build_patches ();

        const std::string filename = ("solution-" +
                Utilities::int_to_string (cycle, 2) +
                "." +
                Utilities::int_to_string
                (triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0;
                    i<Utilities::MPI::n_mpi_processes(mpi_communicator);
                    ++i)
                filenames.push_back ("solution-" +
                        Utilities::int_to_string (cycle, 2) +
                        "." +
                        Utilities::int_to_string (i, 4) +
                        ".vtu");

            std::ofstream master_output ((filename + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }



    template <int dim>
    void Controller<dim>::init_mesh ()
    {
        GridGenerator::hyper_cube (triangulation);
        triangulation.refine_global (5);
    }

    template <int dim>
    void Controller<dim>::run ()
    {
        const unsigned int n_cycles = 10;
        for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
        {
            pcout << "Cycle " << cycle << ':' << std::endl;

            if (cycle == 0)
            {
                init_mesh ();
            }
            else
                refine_grid ();

            setup_system ();

            pcout << "   Number of active cells:       "
                << triangulation.n_global_active_cells()
                << std::endl
                << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

            assemble_system ();
            solve ();

            if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
            {
                TimerOutput::Scope t(computing_timer, "output");
                // output_results (cycle);
            }

            pcout << std::endl;
            computing_timer.print_summary ();
            computing_timer.reset ();
        }

    }
}


double constant_slip(double z)
{
    if (z > 10000)
    {
        return 0.0;
    }
    return 1.0;
}


int main(int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace viscosaur;


        std::vector<std::vector<double> > v;
        double x_min = 1.0;
        double x_max = 10000.0;
        double y_min = 0.0;
        double y_max = 20000.0;
        double fault_slip = 1.0;
        double fault_depth = 10000.0;
        double shear_modulus = 30.0e9;
        double viscosity = 5.0e19;
        
        TwoLayerAnalytic* tla = new TwoLayerAnalytic(fault_slip,
                fault_depth, shear_modulus, viscosity, constant_slip);
        // boost::array<double, 2> fff = abc->simple_stress(1.0, 10000.0);
        // std::cout << def << "   " << abc->simple_velocity(1000.0, 10000.0, 1.0) << std::endl;
        // std::cout << fff[0] << "    " << fff[1] << std::endl;
        // fff = abc->integral_stress(1.0, 10000.0);
        // std::cout << fff[0] << "    " << fff[1] << std::endl;
        boost::array<boost::array<double, 50>, 50> vels;
        for (int i = 0; i < 50; i++) 
        {
            for (int j = 0; j < 50; j++) 
            {
                 vels[i][j] = tla->integral_velocity(10.0 + 500.0 * i, 0.0 + 500 * j, 0.0);
                 std::cout << vels[i][j] << std::endl;
            }
        }
        delete tla;
        // return 1;

        // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        // deallog.depth_console (0);

        // {
        //     Controller<2> laplace_problem_2d;
        //     laplace_problem_2d.run ();
        // }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }

    return 0;
}
