#include "problem_data.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/grid_refinement.h>

#include <boost/python/extract.hpp>
#include <boost/filesystem.hpp>

namespace viscosaur
{
    using namespace dealii;
    namespace bp = boost::python;
    template <int dim>
    ProblemData<dim>::ProblemData(bp::dict &params):
        mpi_comm(MPI_COMM_WORLD),
        triangulation(mpi_comm,
                typename Triangulation<dim>::MeshSmoothing
                (Triangulation<dim>::smoothing_on_refinement |
                 Triangulation<dim>::smoothing_on_coarsening)),
        parameters(params),
        quadrature(bp::extract<int>(parameters["fe_degree"]) + 1),
        one_d_quad(bp::extract<int>(parameters["fe_degree"]) + 1),
        face_quad(bp::extract<int>(parameters["fe_degree"]) + 1),
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
        computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times),
        disp_dof_handler(triangulation),
        disp_fe(QGaussLobatto<1>(
                    bp::extract<int>(parameters["fe_degree"]) + 1)),
        mem_dof_handler(triangulation),
        mem_fe(FE_Q<dim>(QGaussLobatto<1>(
                        bp::extract<int>(parameters["fe_degree"]) + 1)), dim)
    {
        TimerOutput::Scope t(computing_timer, "setup_problem_data");
        const std::string filename = bp::extract<std::string>(
                parameters["mesh_filename"]);
        bool should_load_mesh = 
            bp::extract<bool>(parameters["load_mesh"]);
        if (should_load_mesh)
        {
            bool file_exists = boost::filesystem::exists(filename);
            if (!file_exists)
            {
                should_load_mesh = false;
                pcout << "load_mesh = True, but the mesh file does not"
                    << " exist, so mesh generation will proceed as if"
                    << " load_mesh = False. params[\"load_mesh\"] = False now."
                    << std::endl;
                params["load_mesh"] = false;
            }
        }

        //Even if we load the mesh, we need the basic coarse version to start
        generate_mesh();
        if (should_load_mesh)
        {
            load_mesh();
        } else
        {
            initial_refinement();
        }
        init_dofs();
    }

    template <int dim>
    ProblemData<dim>::~ProblemData()
    {
        disp_dof_handler.clear();
        mem_dof_handler.clear();
        // std::cout << "Destruction of Problem Data." << std::endl;
    }

    template <int dim>
    void
    ProblemData<dim>::
    init_dofs()
    {
        // Collect dofs from the finite elements for the whole triangulation
        // Internally, this also does load balancing and makes sure that each
        // process has a chunk of the dofs. The process will not be aware of 
        // any dofs outside of the owned dofs and the ghost adjacent dofs.
        disp_dof_handler.distribute_dofs(disp_fe);
        mem_dof_handler.distribute_dofs(mem_fe);

        // Check the dealii faq for more information on dof handling in mpi 
        // processes. Set the dofs that this process will actually solve.
        disp_locally_owned_dofs = disp_dof_handler.locally_owned_dofs();
        mem_locally_owned_dofs = mem_dof_handler.locally_owned_dofs();

        // Set the dofs that this process will need to perform solving
        DoFTools::extract_locally_relevant_dofs(disp_dof_handler,
                disp_locally_relevant_dofs);
        DoFTools::extract_locally_relevant_dofs(mem_dof_handler,
                mem_locally_relevant_dofs);

        // Create a constraints matrix that just contains the hanging node 
        // constraints. We will copy this matrix later when we need to add 
        // other constraints like boundary conditions.
        disp_hanging_node_constraints.clear();
        disp_hanging_node_constraints.reinit(disp_locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(disp_dof_handler, 
                                        disp_hanging_node_constraints);
        disp_hanging_node_constraints.close();

        mem_hanging_node_constraints.clear();
        mem_hanging_node_constraints.reinit(mem_locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(mem_dof_handler, 
                                        mem_hanging_node_constraints);
        mem_hanging_node_constraints.close();

        // Also initialize the matrix free objects for any explicit operations
        // we may wish to perform.
        typename MatrixFree<dim>::AdditionalData additional_data_mem;
        additional_data_mem.mapping_update_flags = (update_values |
                                          update_JxW_values |
                                          update_quadrature_points |
                                          update_gradients);
        additional_data_mem.mpi_communicator = mpi_comm;

        typename MatrixFree<dim>::AdditionalData additional_data_disp;
        additional_data_disp.mapping_update_flags = update_gradients |
                                                   update_quadrature_points;
        additional_data_disp.mpi_communicator = mpi_comm;

        mem_matrix_free.reinit(mem_dof_handler, 
                                mem_hanging_node_constraints,
                                one_d_quad, 
                                additional_data_mem);
        disp_matrix_free.reinit(disp_dof_handler, 
                                disp_hanging_node_constraints,
                                one_d_quad, 
                                additional_data_disp);
    }

    template <int dim>
    void ProblemData<dim>::save_mesh(const std::string &filename)
    {
        triangulation.save(filename.c_str());
    }

    template <int dim>
    void ProblemData<dim>::load_mesh()
    {
        const std::string filename = bp::extract<std::string>(
                parameters["mesh_filename"]);
        triangulation.load(filename.c_str());
    }

    template <int dim>
    void ProblemData<dim>::generate_mesh()
    {
        double min_x = bp::extract<double>(parameters["min_corner_x"]);
        double min_y = bp::extract<double>(parameters["min_corner_y"]);
        double max_x = bp::extract<double>(parameters["max_corner_x"]);
        double max_y = bp::extract<double>(parameters["max_corner_y"]);
        Point<dim> min(min_x, min_y);
        Point<dim> max(max_x, max_y);
        std::vector<unsigned int> n_subdivisions;
        
        n_subdivisions.push_back(1);
        n_subdivisions.push_back(1);
        if (dim == 3)
        {
            n_subdivisions.push_back(1);
        }
        pcout << "Creating rectangle with min corner: ("
            << min(0) << ", " << min(1) << ") and max corner: ("
            << max(0) << ", " << max(1) << ")" << std::endl;

        // Create a rectangular grid with corners at min and max
        // the "true" specifies that each side should have a 
        // different boundary indicator.
        // Only 1 subdivision in each direction to start
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  n_subdivisions,
                                                  min,
                                                  max,
                                                  true);
    }

    template <int dim>
    void ProblemData<dim>::initial_refinement()
    {
        // Isotropically refine a few times.
        int initial_isotropic_refines = bp::extract<int>
            (parameters["initial_isotropic_refines"]);
        triangulation.refine_global(initial_isotropic_refines);
    }



    /* Begins the refinement process. Call should be succeeded by calls
     * to the refinement functions in Solution and then by execute_refine.
     * The refinement_measure parameter must be a scalar field. This 
     * constraint could be relaxed easily with a small amount of coding
     */
    template <int dim>
    void 
    ProblemData<dim>::
    start_refine(parallel::distributed::Vector<double> &refinement_measure)
    {
        TimerOutput::Scope t(computing_timer, "refine");

        // Calculate the error estimate from the kelly et. al. paper.
        // TODO: Look up this paper and read about the error estimator.
        // TODO: Refactor this out into another module -- error analysis
        Vector<float> estimated_error_per_cell
            (triangulation.n_active_cells());
        const unsigned int fe_d = bp::extract<int>(parameters["fe_degree"]);
        KellyErrorEstimator<dim>::estimate(disp_dof_handler,
                face_quad,
                typename FunctionMap<dim>::type(),
                refinement_measure,
                estimated_error_per_cell);

        //Print the local L2 error estimate.
        double l2_error = estimated_error_per_cell.l2_norm();
        double l2_soln = refinement_measure.l2_norm();
        double percent_error = l2_error / l2_soln;
        std::cout << "Processor: " + 
            Utilities::int_to_string(
                    Utilities::MPI::this_mpi_process(mpi_comm), 4) + 
            "  with estimated percent error: " << percent_error << std::endl
            << " and estimated absolute error: " << l2_error << std::endl;

        const unsigned int max_grid_level = 
            bp::extract<int>(parameters["max_grid_level"]);
        const unsigned int min_grid_level = 
            bp::extract<int>(parameters["min_grid_level"]);
        const double refine_frac = 
            bp::extract<double>(parameters["refine_frac"]);
        const double coarse_frac = 
            bp::extract<double>(parameters["coarse_frac"]);

        parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_number (triangulation,
                    estimated_error_per_cell,
                    refine_frac, coarse_frac);

        // Don't overrefine or underrefine. Clear all underrefined or 
        // overrefinings
        if (triangulation.n_levels() > max_grid_level)
            for (typename Triangulation<dim>::active_cell_iterator
                 cell = triangulation.begin_active(max_grid_level);
                 cell != triangulation.end(); ++cell)
                cell->clear_refine_flag ();
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active(min_grid_level);
             cell != triangulation.end_active(min_grid_level); ++cell)
            cell->clear_coarsen_flag ();
    }

    template <int dim>
    void
    ProblemData<dim>::
    execute_refine()
    {
        TimerOutput::Scope t(computing_timer, "refine");
        //Actually perform the grid adaption.
        triangulation.execute_coarsening_and_refinement();

        //After the triangulation is modified, 
        //we need to reinitialize the dofs.
        init_dofs();
    }

    // ESSENTIAL: explicity define the template types we will use.
    // Otherwise, 
    // the template definition needs to go in the header file, which
    // is ugly!
    template class ProblemData<2>;
    template class ProblemData<3>;
}
