/* This file is based on the step 48 tutorial from the deal.II documentation.
 */
#include "stress.h"
#include "stress_op.h"
#include "problem_data.h"
#include "poisson.h"
#include "solution.h"

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

#include <fstream>
#include <iostream>
#include <iomanip>

#include <boost/python/extract.hpp>

namespace viscosaur
{
    using namespace dealii;
    namespace bp = boost::python;

    template <int dim>
    Stress<dim>::Stress(Solution<dim> &soln,
                        ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        time_step = bp::extract<double>(pd->parameters["time_step"]);

        constraints = *pd->create_constraints();
        constraints.close();

        typename MatrixFree<dim>::AdditionalData additional_data;
        additional_data.mapping_update_flags = (update_gradients |
                                          update_JxW_values |
                                          update_quadrature_points);
        additional_data.mpi_communicator = MPI_COMM_WORLD;
        additional_data.tasks_parallel_scheme =
            MatrixFree<dim>::AdditionalData::partition_partition;

        //Needs to be one-dimensional
        QGaussLobatto<1> quadrature (fe_degree+1);
        matrix_free_szx.reinit(pd->dof_handler, constraints,
                                quadrature, additional_data);
        matrix_free_szy.reinit(pd->dof_handler, constraints,
                                quadrature, additional_data);
        matrix_free_szx.initialize_dof_vector(soln.cur_szx);
        matrix_free_szy.initialize_dof_vector(soln.cur_szy);
        soln.old_szx.reinit(soln.cur_szx);
        soln.old_szy.reinit(soln.cur_szy);
        soln.tent_szx.reinit(soln.cur_szx);
        soln.tent_szy.reinit(soln.cur_szy);

        InvViscosity<dim>* inv_visc = new InvViscosity<dim>(*pd);
        //Make a vector of stress ops, so that degree can be flexible.
        //First check if the initialization takes a substantial amount of time.
        op_szx = new StressOp<dim, fe_degree>(matrix_free_szx, time_step, 
                                              *pd, *inv_visc);
        op_szy = new StressOp<dim, fe_degree>(matrix_free_szy, time_step, 
                                              *pd, *inv_visc);
    }

    template <int dim>
    Stress<dim>::~Stress()
    {
        delete op_szx;
        delete op_szy;
    }

    template <int dim>
    void
    Stress<dim>::step(Solution<dim> &soln)
    {
        time += time_step;
        timestep_number++;

        //Flip the solns to retain the old soln.
        soln.old_szx.swap(soln.cur_szx);
        soln.old_szy.swap(soln.cur_szy);

        //One time step of the relevant operation
        op_szx->apply(soln.tent_szx, soln.old_szx);
        op_szy->apply(soln.tent_szy, soln.old_szy);

        //Apply constraints to set constrained DoFs to their correct value
        constraints.distribute(soln.tent_szx);
        constraints.distribute(soln.tent_szy);
    }

    template class Stress<2>;
    template class Stress<3>;
}
