/* This file is based on the step 48 tutorial from the deal.II documentation.
 */
#include "stress.h"
#include "scheme.h"
#include "op_factory.h"
#include "problem_data.h"
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
        dealii::TimerOutput::Scope t(p_pd.computing_timer, "setup_stress");
        MatrixFreeCalculation<dim>::reinit(p_pd, p_pd.strs_matrix_free,
                *p_pd.create_strs_constraints());
    }

    template <int dim>
    void
    Stress<dim>::tentative_step(Solution<dim> &soln, 
                                Scheme<dim> &scheme,
                                 double time_step)
    {
        TimerOutput::Scope t(this->pd->computing_timer, "stress_step");
        std::vector<dealii::parallel::distributed::Vector<double> > sources(4);
        sources[0] = soln.old_strs;
        sources[1] = soln.old_old_strs;
        sources[2] = soln.cur_vel;
        sources[3] = soln.old_vel;
        this->op_factory = scheme.get_tentative_step_factory();
        this->apply(soln.tent_strs, sources, &time_step);
    }

    template <int dim>
    void
    Stress<dim>::correction_step(Solution<dim> &soln,
                                 Scheme<dim>& scheme,
                                 double time_step)
    {
        TimerOutput::Scope t(this->pd->computing_timer, "stress_step");
        std::vector<dealii::parallel::distributed::Vector<double> > sources(4);
        sources[0] = soln.tent_strs;
        sources[1] = soln.old_old_strs;
        sources[2] = soln.cur_vel;
        sources[3] = soln.old_vel;
        this->op_factory = scheme.get_correction_step_factory();
        this->apply(soln.cur_strs, sources, &time_step);
    }

    template class Stress<2>;
    template class Stress<3>;
}
