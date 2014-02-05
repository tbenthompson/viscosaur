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
    Stress<dim>::Stress(ProblemData<dim> &p_pd)
    {
        reinit(p_pd);
    }

    template <int dim>
    Stress<dim>::~Stress()
    {
    }
    
    template <int dim>
    void
    Stress<dim>::
    reinit(ProblemData<dim> &p_pd)
    {
        pd = &p_pd; 

        //TODO: This is the only object contained by this class. Can we get
        //rid of this class completely?
        constraints = *pd->create_strs_constraints();
        constraints.close();
    }

    template <int dim>
    void
    Stress<dim>::
    generic_step(parallel::distributed::Vector<double> &input,
                 parallel::distributed::Vector<double> &output,
                 Solution<dim> &soln,
                 unsigned int component,
                 double time_step,
                 StressOp<dim, FE_DEGREE> &op)
    {
        TimerOutput::Scope t(pd->computing_timer, "stress_step");

        //One time step of the relevant operation
        op.apply(output, input, soln, component, time_step);

        //Apply constraints to set constrained DoFs to their correct value
        constraints.distribute(output);

        //Spread ghost values across processors
        output.update_ghost_values();
    }

    template <int dim>
    void
    Stress<dim>::tentative_step(Solution<dim> &soln, 
                                Scheme<dim> &scheme,
                                 double time_step)
    {
        StressOp<dim, FE_DEGREE>* stepper = scheme.get_tentative_stepper();
        generic_step(soln.old_strs, soln.tent_strs, soln, 0, time_step, *stepper);
    }

    template <int dim>
    void
    Stress<dim>::correction_step(Solution<dim> &soln,
                                 Scheme<dim>& scheme,
                                 double time_step)
    {
        StressOp<dim, FE_DEGREE>* stepper = scheme.get_correction_stepper();
        generic_step(soln.tent_strs, soln.cur_strs, soln, 0, time_step, *stepper);
    }

    template class Stress<2>;
    template class Stress<3>;
}
