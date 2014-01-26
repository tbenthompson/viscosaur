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
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#include <boost/python/extract.hpp>

namespace viscosaur
{
    using namespace dealii;
    namespace bp = boost::python;

    template <int dim, int fe_degree>
    void
    StressOp<dim, fe_degree>::
    init(const MatrixFree<dim,double> &data_in, 
             const double p_time_step,
             ProblemData<dim> &p_pd,
             InvViscosity<dim> &p_inv_visc)
    {
        data = &data_in;
        pd = &p_pd;
        VectorizedArray<double> one = make_vectorized_array(1.);
        data->initialize_dof_vector(inv_mass_matrix);

        FEEvaluationGL<dim,fe_degree> fe_eval(*data);
        const unsigned int n_q_points = fe_eval.n_q_points;

        for (unsigned int cell=0; cell < data->n_macro_cells(); ++cell)
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

        shear_modulus = bp::extract<double>
            (pd->parameters["shear_modulus"]);
        time_step = p_time_step;
        inv_visc = &p_inv_visc;
    }

    template <int dim, int fe_degree>
    void 
    StressOp<dim, fe_degree>::
    apply(parallel::distributed::Vector<double> &dst,
          const parallel::distributed::Vector<double> &src,
          Solution<dim> &soln,
          const unsigned int comp)
    {
        dst = 0;
        this->soln = &soln;
        this->component = comp;
        data->cell_loop(&StressOp<dim,fe_degree>::local_apply,
                       this, dst, src);
        dst.scale(inv_mass_matrix);
    }


    template <int dim, int fe_degree>
    void 
    TentativeOp<dim, fe_degree>::
    local_apply (const MatrixFree<dim> &data,
                 parallel::distributed::Vector<double> &output,
                 const parallel::distributed::Vector<double> &input,
                 const std::pair<unsigned int,unsigned int> &cell_range) const
    {
        FEEvaluationGL<dim,fe_degree> current(data);
        VectorizedArray<double> one = make_vectorized_array(1.);
        for (unsigned int cell = cell_range.first;
             cell < cell_range.second; 
             ++cell)
        {
            current.reinit(cell);

            current.read_dof_values(input);

            current.evaluate(true, false, false);

            for (unsigned int q=0; q < current.n_q_points; ++q)
            {
                const VectorizedArray<double> current_value = 
                        current.get_value(q);
                //
                // Here's where you modify the time stepping!
                const VectorizedArray<double> factor = 
                    this->shear_modulus * this->time_step * 
                    this->inv_visc->value(current.quadrature_point(q), 0);
                current.submit_value((one - factor) * current_value, q);
            }

            current.integrate(true, false);
            current.distribute_local_to_global(output);
        }
    }

    template <int dim, int fe_degree>
    void 
    CorrectionOp<dim, fe_degree>::
    local_apply (const MatrixFree<dim> &data,
                 parallel::distributed::Vector<double> &output,
                 const parallel::distributed::Vector<double> &input,
                 const std::pair<unsigned int,unsigned int> &cell_range) const
    {
        FEEvaluationGL<dim, fe_degree> current(data);
        FEEvaluationGL<dim, fe_degree> vel(data);
        VectorizedArray<double> mu_dt = 
            make_vectorized_array(this->shear_modulus * this->time_step);
        for (unsigned int cell = cell_range.first;
             cell < cell_range.second; 
             ++cell)
        {
            current.reinit(cell);
            vel.reinit(cell);
            current.read_dof_values(input);
            vel.read_dof_values_plain(this->soln->cur_vel_for_strs);
            current.evaluate(true, false, false);
            vel.evaluate(false, true, false);

            for (unsigned int q=0; q < current.n_q_points; ++q)
            {
                const VectorizedArray<double> current_value = 
                        current.get_value(q);

                const dealii::Tensor<1, dim, 
                      dealii::VectorizedArray<double> > grad_vel = 
                        vel.get_gradient(q);

                // Here's where you modify the time stepping!
                current.submit_value(current_value + 
                        mu_dt * grad_vel[this->component], q);
            }

            current.integrate(true, false);
            current.distribute_local_to_global(output);
        }
    }
    template class StressOp<2, 1>;
    template class StressOp<2, 2>;
    template class StressOp<2, 3>;
    template class StressOp<2, 4>;
    template class StressOp<2, 5>;
    template class StressOp<2, 6>;
    template class StressOp<2, 7>;
    template class StressOp<2, 8>;
    template class StressOp<2, 9>;
    template class StressOp<2, 10>;
    template class StressOp<3, 1>;
    template class StressOp<3, 2>;
    template class StressOp<3, 3>;
    template class StressOp<3, 4>;
    template class StressOp<3, 5>;
    template class StressOp<3, 6>;
    template class StressOp<3, 7>;
    template class StressOp<3, 8>;
    template class StressOp<3, 9>;
    template class StressOp<3, 10>;
    template class TentativeOp<2, 1>;
    template class TentativeOp<2, 2>;
    template class TentativeOp<2, 3>;
    template class TentativeOp<2, 4>;
    template class TentativeOp<2, 5>;
    template class TentativeOp<2, 6>;
    template class TentativeOp<2, 7>;
    template class TentativeOp<2, 8>;
    template class TentativeOp<2, 9>;
    template class TentativeOp<2, 10>;
    template class TentativeOp<3, 1>;
    template class TentativeOp<3, 2>;
    template class TentativeOp<3, 3>;
    template class TentativeOp<3, 4>;
    template class TentativeOp<3, 5>;
    template class TentativeOp<3, 6>;
    template class TentativeOp<3, 7>;
    template class TentativeOp<3, 8>;
    template class TentativeOp<3, 9>;
    template class TentativeOp<3, 10>;


    template class CorrectionOp<2, 1>;
    template class CorrectionOp<2, 2>;
    template class CorrectionOp<2, 3>;
    template class CorrectionOp<2, 4>;
    template class CorrectionOp<2, 5>;
    template class CorrectionOp<2, 6>;
    template class CorrectionOp<2, 7>;
    template class CorrectionOp<2, 8>;
    template class CorrectionOp<2, 9>;
    template class CorrectionOp<2, 10>;
    template class CorrectionOp<3, 1>;
    template class CorrectionOp<3, 2>;
    template class CorrectionOp<3, 3>;
    template class CorrectionOp<3, 4>;
    template class CorrectionOp<3, 5>;
    template class CorrectionOp<3, 6>;
    template class CorrectionOp<3, 7>;
    template class CorrectionOp<3, 8>;
    template class CorrectionOp<3, 9>;
    template class CorrectionOp<3, 10>;
}
