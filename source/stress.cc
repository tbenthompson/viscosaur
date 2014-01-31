/* This file is based on the step 48 tutorial from the deal.II documentation.
 */
#include "stress.h"
#include "scheme.h"
#include "stress_op.h"
#include "problem_data.h"
#include "inv_visc.h"
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

        constraints = *pd->create_constraints();
        constraints.close();


        //Make a vector of stress ops, so that degree can be flexible.
        //First check if the initialization takes a substantial amount of time.
        //TODO:Separate this out to the python layer. Maybe provide a "factory"
        //type system that produces 1st,2nd,3rd,4th order versions?
        // t_step = new TentativeOp2<dim, fe_degree>(matrix_free, time_step, 
        //                                       *pd, *inv_visc);
        // c_step = new CorrectionOp2<dim, fe_degree>(matrix_free, time_step, 
        //                                       *pd, *inv_visc);
    }

    template <int dim>
    Stress<dim>::~Stress()
    {
    }

    template <int dim>
    void
    Stress<dim>::
    generic_step(parallel::distributed::Vector<double> &input,
                 parallel::distributed::Vector<double> &output,
                 Solution<dim> &soln,
                 unsigned int component,
                 StressOp<dim, FE_DEGREE> &op)
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
    Stress<dim>::tentative_step(Solution<dim> &soln, 
                                Scheme<dim> &scheme)
    {
        StressOp<dim, FE_DEGREE>* stepper = scheme.get_tentative_stepper();
        generic_step(soln.old_szx, soln.tent_szx, soln, 0, *stepper);
        generic_step(soln.old_szy, soln.tent_szy, soln, 1, *stepper);
    }

    template <int dim>
    void
    Stress<dim>::correction_step(Solution<dim> &soln,
                                 Scheme<dim>& scheme)
    {
        StressOp<dim, FE_DEGREE>* stepper = scheme.get_correction_stepper();
        generic_step(soln.tent_szx, soln.cur_szx, soln, 0, *stepper);
        generic_step(soln.tent_szy, soln.cur_szy, soln, 1, *stepper);
    }

    template class Stress<2>;
    template class Stress<3>;
}
