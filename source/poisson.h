#ifndef __viscosaur_poisson_h
#define __viscosaur_poisson_h
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// On my current machine, Trilinos linear algebra seems to be
// about twice as fast as PETSc. This is probably an artifact of some 
// configurations, so flip this flag to try out PETSc (assuming it's
// installed and deal.II is configured to use it). 
// However, the python bindings are not set up for PETSc, so new 
// bindings will need to be made. On the other hand, petsc4py
// might be able to do the job.
// #define USE_PETSC_LA

namespace LA
{
#ifdef USE_PETSC_LA
    using namespace dealii::LinearAlgebraPETSc;
#else
    using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
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

#include <fstream>
#include <iostream>
#include <memory>

#include <Python.h>
#include <boost/python/dict.hpp>


#include "analytic.h"
#include "one_step_rhs.h"

// TODO: Clean up this header file. Pimpl it.
namespace viscosaur
{
    using namespace dealii;
    namespace bp = boost::python;


    /* Currently, this must remain a compile constant.
     * Using higher order polynomials results in a problem with very slow 
     * assembly. I suspect the fe_values.shape_grad call is not caching 
     * the values properly
     */
    const unsigned int fe_degree = 2;

    template <int dim>
    struct ProblemData
    {
        bp::dict              parameters;
        MPI_Comm              mpi_comm;
        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim>       dof_handler;
        FE_Q<dim>             fe;
        IndexSet              locally_owned_dofs;
        IndexSet              locally_relevant_dofs;
        ConstraintMatrix      constraints;
        ConditionalOStream    pcout;
        TimerOutput           computing_timer;

        ProblemData(bp::dict params):
            mpi_comm(MPI_COMM_WORLD),
            triangulation(mpi_comm,
                    typename Triangulation<dim>::MeshSmoothing
                    (Triangulation<dim>::smoothing_on_refinement |
                     Triangulation<dim>::smoothing_on_coarsening)),
            dof_handler (triangulation),
            fe (QGaussLobatto<1>(fe_degree + 1)),
            pcout (std::cout,
                    (Utilities::MPI::this_mpi_process(mpi_comm)
                     == 0)),
            computing_timer (pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),
            parameters (params)
        {}
    };

    /*
     * The Poisson Solver. Most of this code is extracted from tutorial 40
     * on the deal.ii website. Currently located at
     * http://www.dealii.org/8.1.0/doxygen/deal.II/step_40.html 
     *
     * Add some documentation...
     *
     * Note that this entire class is defined in the header. This is required
     * for a templated class. C++11 may have "fixed" this. Check?
     */
    template <int dim>
    class Poisson
    {
        public:
            Poisson (bp::dict params);
            ~Poisson ();

            LA::MPI::Vector run (PoissonRHS<dim>* rhs);
            DoFHandler<dim>* get_dof_handler();

        private:
            void setup_system ();

            /* Assembly functions.
             */
            void fill_cell_matrix(
                    FullMatrix<double> &cell_matrix,
                    FEValues<dim> &fe_values,
                    const unsigned int n_q_points,
                    const unsigned int dofs_per_cell);
            void assemble_system (PoissonRHS<dim>* rhs);

            void solve ();
            void refine_grid ();
            std::string output_filename(const unsigned int cycle,
                                        const unsigned int subdomain) const;
            void output_results (const unsigned int cycle) const;
            void init_mesh ();
            
            ProblemData<dim> pd;
            LA::MPI::SparseMatrix system_matrix;
            LA::MPI::Vector       locally_relevant_solution;
            LA::MPI::Vector       system_rhs;
    };

    template<int dim>
    DoFHandler<dim>* Poisson<dim>::get_dof_handler()
    {
        return &pd.dof_handler;
    }

    template <int dim>
    Poisson<dim>::Poisson (bp::dict params):
        pd(params)
    {
        pd.pcout << "Setting up the Poisson solver." << std::endl;
        init_mesh();
        setup_system();
    }



    template <int dim>
    Poisson<dim>::~Poisson ()
    {
        pd.dof_handler.clear();
    }



    template <int dim>
    void Poisson<dim>::setup_system ()
    {
        //The theme in this function is that only the locally relevant or 
        //locally owned dofs will be made known to any given processor.
        TimerOutput::Scope t(pd.computing_timer, "setup");

        // Spread dofs 
        pd.dof_handler.distribute_dofs (pd.fe);

        // Set the dofs that this process will actually solve.
        pd.locally_owned_dofs = pd.dof_handler.locally_owned_dofs ();

        // Set the dofs that this process will need to perform solving
        DoFTools::extract_locally_relevant_dofs(pd.dof_handler,
                pd.locally_relevant_dofs);

        locally_relevant_solution.reinit (pd.locally_owned_dofs,
                pd.locally_relevant_dofs, pd.mpi_comm);
        system_rhs.reinit (pd.locally_owned_dofs, pd.mpi_comm);

        system_rhs = 0;

        // Constrain hanging nodes and boundary conditions.
        pd.constraints.clear();
        pd.constraints.reinit(pd.locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints (pd.dof_handler, pd.constraints);
        VectorTools::interpolate_boundary_values (pd.dof_handler,
                0,
                ZeroFunction<dim>(),
                pd.constraints);
        pd.constraints.close();

        CompressedSimpleSparsityPattern csp (pd.locally_relevant_dofs);

        DoFTools::make_sparsity_pattern (pd.dof_handler, csp,
                pd.constraints, false);
        SparsityTools::distribute_sparsity_pattern (csp,
                pd.dof_handler.n_locally_owned_dofs_per_processor(),
                pd.mpi_comm,
                pd.locally_relevant_dofs);

        system_matrix.reinit (pd.locally_owned_dofs,
                pd.locally_owned_dofs,
                csp,
                pd.mpi_comm);
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
                    // Note: for a 12th order method, this = 0 in 88% of
                    // cases.
                    cell_matrix(i,j) += (fe_values.shape_grad(i, q_point) *
                            fe_values.shape_grad(j, q_point) *
                            fe_values.JxW(q_point));
                }
            }
        }
    }

    template <int dim>
    void Poisson<dim>::assemble_system(PoissonRHS<dim>* rhs)
    {
        TimerOutput::Scope t(pd.computing_timer, "assembly");

        const QGaussLobatto<dim>  quadrature_formula(fe_degree + 1);

        FEValues<dim> fe_values (pd.fe, quadrature_formula,
                update_values    |  update_gradients |
                update_quadrature_points |
                update_JxW_values);

        const unsigned int   dofs_per_cell = pd.fe.dofs_per_cell;
        const unsigned int   n_q_points    = quadrature_formula.size();

        FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       cell_rhs (dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator
            cell = pd.dof_handler.begin_active(),
                 endc = pd.dof_handler.end();

        rhs->start_assembly();

        for (; cell!=endc; ++cell)
        {
            if (!cell->is_locally_owned())
            {
                continue;
            }
            TimerOutput::Scope t2(pd.computing_timer, "cell_construction");
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);
            rhs->fill_cell_rhs(cell_rhs, fe_values, n_q_points, dofs_per_cell);
            fill_cell_matrix(cell_matrix, fe_values, n_q_points, dofs_per_cell);

            cell->get_dof_indices(local_dof_indices);
            pd.constraints.distribute_local_to_global(cell_matrix,
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
        TimerOutput::Scope t(pd.computing_timer, "solve");
        LA::MPI::Vector
            completely_distributed_solution (pd.locally_owned_dofs, 
                                             pd.mpi_comm);

        SolverControl solver_control (pd.dof_handler.n_dofs(), 1e-12);

        LA::SolverCG solver(solver_control, pd.mpi_comm);
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

        pd.pcout << "   Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

        pd.constraints.distribute(completely_distributed_solution);

        locally_relevant_solution = completely_distributed_solution;
    }




    template <int dim>
    void Poisson<dim>::refine_grid ()
    {
        TimerOutput::Scope t(pd.computing_timer, "refine");

        Vector<float> estimated_error_per_cell (pd.triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate (pd.dof_handler,
                QGauss<dim-1>(fe_degree + 1),
                typename FunctionMap<dim>::type(),
                locally_relevant_solution,
                estimated_error_per_cell);

        //Print the local L2 error estimate.
        double l2_error = estimated_error_per_cell.l2_norm();
        std::cout << "Processor: " + 
            Utilities::int_to_string(
                    Utilities::MPI::this_mpi_process(pd.mpi_comm), 4) + 
            "  with error: " << l2_error <<
            std::endl;

        parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_number (pd.triangulation,
                    estimated_error_per_cell,
                    0.5, 0.3);

        // Don't overrefine or underrefine.
        const unsigned int max_grid_level = 
            bp::extract<int>(pd.parameters["max_grid_level"]);
        const unsigned int min_grid_level = 
            bp::extract<int>(pd.parameters["min_grid_level"]);
        if (pd.triangulation.n_levels() > max_grid_level)
            for (typename Triangulation<dim>::active_cell_iterator
                 cell = pd.triangulation.begin_active(max_grid_level);
                 cell != pd.triangulation.end(); ++cell)
                cell->clear_refine_flag ();
        for (typename Triangulation<dim>::active_cell_iterator
             cell = pd.triangulation.begin_active(min_grid_level);
             cell != pd.triangulation.end_active(min_grid_level); ++cell)
            cell->clear_coarsen_flag ();

        pd.triangulation.execute_coarsening_and_refinement();
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
        data_out.attach_dof_handler(pd.dof_handler);
        data_out.add_data_vector(locally_relevant_solution, "u");

        Vector<float> subdomain(pd.triangulation.n_active_cells());
        unsigned int this_subd = pd.triangulation.locally_owned_subdomain();
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = this_subd;
        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::string this_f = output_filename(cycle, this_subd);
        std::ofstream output(("data/" + this_f + ".vtu").c_str());
        data_out.write_vtu(output);

        if (Utilities::MPI::this_mpi_process(pd.mpi_comm) == 0)
        {
            // Build the list of filenames to store in the master pvtu file
            std::vector<std::string> filenames;
            for (unsigned int i=0;
                    i<Utilities::MPI::n_mpi_processes(pd.mpi_comm);
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
    void Poisson<dim>::init_mesh ()
    {
        Point<dim> min = bp::extract<Point<dim> >(pd.parameters["min_corner"]);
        Point<dim> max = bp::extract<Point<dim> >(pd.parameters["max_corner"]);
        std::vector<unsigned int> n_subdivisions;
        n_subdivisions.push_back(1);
        n_subdivisions.push_back(1);
        if (dim == 3)
        {
            n_subdivisions.push_back(1);
        }
        pd.pcout << "Creating rectangle with min corner: ("
            << min(0) << ", " << min(1) << ") and max corner: ("
            << max(0) << ", " << max(1) << ")" << std::endl;

        // Create a rectangular grid with corners at min and max
        // the "true" specifies that each side should have a 
        // different boundary indicator.
        // Only 1 subdivision in each direction to start
        GridGenerator::subdivided_hyper_rectangle(pd.triangulation,
                                                  n_subdivisions,
                                                  min,
                                                  max,
                                                  true);
        // GridGenerator::hyper_cube (pd.triangulation);
        // Isotropically refine a few times.
        int initial_isotropic_refines = bp::extract<int>
            (pd.parameters["initial_isotropic_refines"]);
        pd.triangulation.refine_global(initial_isotropic_refines);
    }

    template <int dim>
    LA::MPI::Vector Poisson<dim>::run (PoissonRHS<dim>* rhs)
    {
        const unsigned int n_cycles = 5;
        for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
        {
            pd.pcout << "Cycle " << cycle << ':' << std::endl;

            if (cycle != 0)
            {
                refine_grid();
                setup_system();
                // Build the matrices
                assemble_system(rhs);
            }


            pd.pcout << "   Number of active cells:       "
                << pd.triangulation.n_global_active_cells()
                << std::endl
                << "   Number of degrees of freedom: "
                << pd.dof_handler.n_dofs()
                << std::endl;


            // Solve the linear system.
            solve();

            if (Utilities::MPI::n_mpi_processes(pd.mpi_comm) <= 32)
            {
                TimerOutput::Scope t(pd.computing_timer, "output");
                output_results (cycle);
            }

            pd.pcout << std::endl;
            pd.computing_timer.print_summary ();
            pd.computing_timer.reset ();
        }
        return locally_relevant_solution;
    }
}
#endif
