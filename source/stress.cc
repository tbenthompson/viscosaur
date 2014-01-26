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
        TimerOutput::Scope t(pd->computing_timer, "setup_stress");
        time_step = bp::extract<double>(pd->parameters["time_step"]);

        constraints = *pd->create_constraints();
        constraints.close();

        typename MatrixFree<dim>::AdditionalData additional_data;
        additional_data.mapping_update_flags = (update_gradients |
                                          update_JxW_values |
                                          update_quadrature_points);
        additional_data.mpi_communicator = pd->mpi_comm;

        //Needs to be one-dimensional
        QGaussLobatto<1> quadrature (fe_degree+1);
        matrix_free.reinit(pd->dof_handler, constraints,
                                quadrature, additional_data);
        matrix_free.initialize_dof_vector(soln.cur_szx);
        soln.cur_szy.reinit(soln.cur_szx);
        soln.old_szx.reinit(soln.cur_szx);
        soln.old_szy.reinit(soln.cur_szx);
        soln.tent_szx.reinit(soln.cur_szx);
        soln.tent_szy.reinit(soln.cur_szx);
        soln.cur_vel_for_strs.reinit(soln.cur_szx);

        InvViscosity<dim>* inv_visc = new InvViscosity<dim>(*pd);
        //Make a vector of stress ops, so that degree can be flexible.
        //First check if the initialization takes a substantial amount of time.
        t_step = new TentativeOp<dim, fe_degree>(matrix_free, time_step, 
                                              *pd, *inv_visc);
        c_step = new CorrectionOp<dim, fe_degree>(matrix_free, time_step, 
                                              *pd, *inv_visc);
    }

    template <int dim>
    Stress<dim>::~Stress()
    {
        delete t_step;
        delete c_step;
    }

    template <int dim>
    void
    Stress<dim>::
    generic_step(parallel::distributed::Vector<double> &input,
                 parallel::distributed::Vector<double> &output,
                 Solution<dim> &soln,
                 unsigned int component,
                 StressOp<dim, fe_degree> &op)
    {
        TimerOutput::Scope t(pd->computing_timer, "stress_step");

        //One time step of the relevant operation
        op.apply(output, input, soln, component);

        //Apply constraints to set constrained DoFs to their correct value
        constraints.distribute(output);

        //Spread ghost values across processors
        output.update_ghost_values();
    }

    template <int dim>
    void
    Stress<dim>::tentative_step(Solution<dim> &soln)
    {
        //Move the time forward
        time += time_step;
        timestep_number++;

        //Flip the solns to retain the old soln.
        soln.old_szx.swap(soln.cur_szx);
        soln.old_szy.swap(soln.cur_szy);

        //Take a step using the first step operator
        generic_step(soln.old_szx, soln.tent_szx, soln, 0, *t_step);
        generic_step(soln.old_szy, soln.tent_szy, soln, 1, *t_step);
    }

    template <int dim>
    void
    Stress<dim>::correction_step(Solution<dim> &soln)
    {
        generic_step(soln.tent_szx, soln.cur_szx, soln, 0, *c_step);
        generic_step(soln.tent_szy, soln.cur_szy, soln, 1, *c_step);
    }

    template class Stress<2>;
    template class Stress<3>;
}
