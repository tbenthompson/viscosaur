/* This file is based on the step 48 tutorial from the deal.II documentation.
 */
#include "stress.h"
#include "stress_op.h"
#include "problem_data.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <iomanip>


namespace viscosaur
{
    using namespace dealii;

    template <int dim>
    Stress<dim>::Stress(ProblemData<dim> &p_pd):
    {
        pd = &p_pd;
        init();
    }

    template <int dim>
    void Stress<dim>::init()
    {
        constraints = *pd->create_constraints();

        typename MatrixFree<dim>::AdditionalData additional_data;
        additional_data.mpi_communicator = MPI_COMM_WORLD;
        additional_data.tasks_parallel_scheme =
            MatrixFree<dim>::AdditionalData::partition_partition;

        matrix_free_data.reinit(pd->dof_handler, constraints,
                                pd->quadrature, additional_data);
        matrix_free_data.initialize_dof_vector(solution);
        old_solution.reinit(solution);
        old_old_solution.reinit(solution);
    }
    // // Applying constraints to the solution that was computed without paying
    // attention to constraints
        // constraints.distribute (solution);
    //I think this allows some of the fancy operators like "itegrate_difference"
    // to operate on a distributed vector
        // solution.update_ghost_values();

    //Calculate CFL number! Useful for the future
        // const double local_min_cell_diameter =
        //     triangulation.last()->diameter()/std::sqrt(dim);
        // const double global_min_cell_diameter
        //     = -Utilities::MPI::max(-local_min_cell_diameter, MPI_COMM_WORLD);
        // time_step = cfl_number * global_min_cell_diameter;
        // time_step = (final_time-time)/(int((final_time-time)/time_step));
    template <int dim>
    void
    Stress<dim>::run ()
    {
        init();
        pcout << "   Time step size: " << time_step << ", finest cell: "
              << global_min_cell_diameter << std::endl << std::endl;

        VectorTools::interpolate (pd->dof_handler,
            ExactSolution<dim> (1, time),
            solution);
        VectorTools::interpolate (pd->dof_handler,
            ExactSolution<dim> (1, time-time_step),
            old_solution);
        output_results (0);

        std::vector<parallel::distributed::Vector<double>*> previous_solutions;
        previous_solutions.push_back(&old_solution);
        previous_solutions.push_back(&old_old_solution);

        StressOp<dim,fe_degree> operation (matrix_free_data,
        time_step);
        unsigned int timestep_number = 1;

        Timer timer;
        double wtime = 0;
        double output_time = 0;
        for (time += time_step; time<=final_time; time+=time_step, 
                                                  ++timestep_number)
        {
            old_old_solution.swap (old_solution);
            old_solution.swap (solution);
            operation.apply (solution, previous_solutions);
        }
    }
}
