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
        dealii::TimerOutput::Scope t(pd->computing_timer, "setup_stress");

        //TODO: This is the only object contained by this class. Can we get
        //rid of this class completely?
        constraints = *pd->create_strs_constraints();
        constraints.close();

        compute_mass_matrix();
    }

    template <int dim>
    void
    Stress<dim>::
    compute_mass_matrix()
    {
        // Integrate and invert the diagonal mass matrix resulting from the
        // Gauss Lobatto Lagrange elements and quadrature match up.
        pd->strs_matrix_free.initialize_dof_vector(inv_mass_matrix);

        this->op_factory = new MassMatrixOpFactory<dim>();
        
        // Use the massmatrixoperator 
        std::vector<dealii::parallel::distributed::Vector <double> >
            empty_source(0);
        local_apply(pd->strs_matrix_free, 
                    inv_mass_matrix, 
                    empty_source,
                    std::pair<unsigned int, unsigned int>(0, 
                        pd->strs_matrix_free.n_macro_cells()));

        delete this->op_factory;

        inv_mass_matrix.compress(dealii::VectorOperation::add);
        for (unsigned int k=0; k < inv_mass_matrix.local_size(); ++k)
        {
            if (inv_mass_matrix.local_element(k)>1e-15)
            {
                inv_mass_matrix.local_element(k) = 
                    1. / inv_mass_matrix.local_element(k);
            }
            else
            {
                inv_mass_matrix.local_element(k) = 0;
            }
        }
    }

    template <int dim>
    void 
    Stress<dim>::
    apply(dealii::parallel::distributed::Vector<double> &dst,
          const dealii::parallel::distributed::Vector<double> &src,
          Solution<dim> &soln,
          const double time_step)
    {
        dst = 0;

        std::vector<dealii::parallel::distributed::Vector<double> >
            sources(4);
        sources[0] = src;
        sources[1] = soln.old_old_strs;
        sources[2] = soln.cur_vel;
        sources[3] = soln.old_vel;
        
        this->time_step = time_step;

        pd->strs_matrix_free.cell_loop(&Stress<dim>::local_apply,
                this, dst, sources);

        dst.scale(inv_mass_matrix);
        //
        //Apply constraints to set constrained DoFs to their correct value
        constraints.distribute(dst);

        //Spread ghost values across processors
        dst.update_ghost_values();
    }

    template <int dim> 
    void
    Stress<dim>::
    local_apply(const dealii::MatrixFree<dim> &data,
                dealii::parallel::distributed::Vector<double> &dst,
                const std::vector<
                    dealii::parallel::distributed::Vector <double> > &src,
                const std::pair<unsigned int, unsigned int> &cell_range)
    {
        // Ask MatrixFree for cell_range for different
        // orders
        const unsigned int max_degree =
            bp::extract<int>(pd->parameters["max_degree"]);
        std::pair<unsigned int, unsigned int> subrange_deg; 
        for(unsigned int deg = 0; deg < max_degree; deg++)
        {
            subrange_deg = data.create_cell_subrange_hp(cell_range, deg); 
            if (subrange_deg.second > subrange_deg.first) 
                this->op_factory->call(deg, *pd, dst, src, subrange_deg, time_step);
        }
    }

    template <int dim>
    void
    Stress<dim>::tentative_step(Solution<dim> &soln, 
                                Scheme<dim> &scheme,
                                 double time_step)
    {
        TimerOutput::Scope t(pd->computing_timer, "stress_step");
        this->op_factory = scheme.get_tentative_step_factory();
        apply(soln.tent_strs, soln.old_strs, soln, time_step);
    }

    template <int dim>
    void
    Stress<dim>::correction_step(Solution<dim> &soln,
                                 Scheme<dim>& scheme,
                                 double time_step)
    {
        TimerOutput::Scope t(pd->computing_timer, "stress_step");
        this->op_factory = scheme.get_correction_step_factory();
        apply(soln.cur_strs, soln.tent_strs, soln, time_step);
    }

    template class Stress<2>;
    template class Stress<3>;
}
