#include "stress_op.h"

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
    template <int dim, int fe_degree>
    StressOp<dim,fe_degree>::
    StressOp(const MatrixFree<dim,double> &data_in, const double time_step):
        data(data_in),
        delta_t_sqr(make_vectorized_array(time_step * time_step))
    {
        VectorizedArray<double> one = make_vectorized_array(1.);

        data.initialize_dof_vector(inv_mass_matrix);

        FEEvaluationGL<dim,fe_degree> fe_eval(data);
        const unsigned int n_q_points = fe_eval.n_q_points;

        for (unsigned int cell=0; cell < data.n_macro_cells(); ++cell)
        {
            fe_eval.reinit(cell);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                fe_eval.submit_value(one, q);
            }
            fe_eval.integrate(true, false);
            fe_eval.distribute_local_to_global(inv_mass_matrix);
        }

        inv_mass_matrix.compress(VectorOperation::add);
        for (unsigned int k=0; k<inv_mass_matrix.local_size(); ++k)
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

    template <int dim, int fe_degree>
    void 
    StressOp<dim, fe_degree>::
    local_apply (const MatrixFree<dim> &data,
                 parallel::distributed::Vector<double> &dst,
                 const std::vector<parallel::distributed::Vector<double>*> &src,
                 const std::pair<unsigned int,unsigned int> &cell_range) const
    {
        AssertDimension(src.size(), 2);
        FEEvaluationGL<dim,fe_degree> current(data), old(data);
        for (unsigned int cell = cell_range.first;
             cell < cell_range.second; 
             ++cell)
        {
            current.reinit(cell);
            old.reinit(cell);

            current.read_dof_values(*src[0]);
            old.read_dof_values(*src[1]);

            current.evaluate(true, true, false);
            old.evaluate(true, false, false);

            for (unsigned int q=0; q<current.n_q_points; ++q)
            {
                const VectorizedArray<double> current_value = 
                        current.get_value(q);
                const VectorizedArray<double> old_value = 
                        old.get_value(q);

                // Here's where you modify the time stepping!
                current.submit_value (2.*current_value - old_value -
                          delta_t_sqr * std::sin(current_value),q);
                current.submit_gradient (- delta_t_sqr *
                             current.get_gradient(q), q);
            }

            current.integrate(true,true);
            current.distribute_local_to_global(dst);
        }
    }

    template <int dim, int fe_degree>
    void 
    StressOp<dim, fe_degree>::
    apply(parallel::distributed::Vector<double> &dst,
          const std::vector<parallel::distributed::Vector<double>*> &src) const
    {
        dst = 0;
        data.cell_loop(&StressOp<dim,fe_degree>::local_apply,
                       this, dst, src);
        dst.scale(inv_mass_matrix);
    }

    template <2, 2> class StressOp;
    template <2, 3> class StressOp;
    template <2, 3> class StressOp;
    template <2, 4> class StressOp;
    template <2, 5> class StressOp;
    template <2, 6> class StressOp;
    template <2, 7> class StressOp;
    template <2, 8> class StressOp;
    template <2, 9> class StressOp;
    template <2, 10> class StressOp;
}
