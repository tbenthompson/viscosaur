#ifndef __viscosaur_stress_op_h
#define __viscosaur_stress_op_h
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/parallel_vector.h>
#include <boost/python/extract.hpp>
#include <boost/any.hpp>
#include "solution.h"
#include "problem_data.h"
#include "inv_visc.h"
#include "op_factory.h"

namespace viscosaur
{
    namespace bp = boost::python;

    template <int dim, int fe_degree>
    class StressOp
    {
        public:
            void hp_local_apply(ProblemData<dim> &pd, 
                 dealii::parallel::distributed::Vector<double> &dst,
                 const std::vector<
                        dealii::parallel::distributed::Vector<double> > &src,
                 const std::pair<unsigned int, unsigned int> &cell_range,
                 boost::any data);

            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree, dim> 
                    &cur_eval, const unsigned int q) = 0;

            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > cur_strs;
            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > old_strs;
            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > cur_grad_vel;
            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > old_grad_vel;
            ProblemData<dim>* pd;
            dealii::VectorizedArray<double> mu_dt;
            dealii::VectorizedArray<double> one;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > tensor_one;
    };

    template <int dim, int fe_degree>
    void 
    StressOp<dim, fe_degree>::
    hp_local_apply(ProblemData<dim> &pd,
                   dealii::parallel::distributed::Vector<double> &dst,
                   const std::vector<
                       dealii::parallel::distributed::Vector<double> > &src,
                   const std::pair<unsigned int,unsigned int> &cell_range,
                   boost::any data)
    {
        double time_step = *boost::any_cast<double*>(data);
        this->pd = &pd;
        const double shear_modulus = 
            bp::extract<double>(pd.parameters["shear_modulus"]);
        this->mu_dt = dealii::make_vectorized_array(shear_modulus * time_step);
        this->one = dealii::make_vectorized_array(1.0);
        for(int d = 0; d < dim; d++) 
        {
            for(unsigned int array_el = 0; array_el < 
                    this->tensor_one[0].n_array_elements; array_el++)
            {
                this->tensor_one[d][array_el] = 1.0;
            }
        }

        dealii::FEEvaluationGL<dim, fe_degree, dim> cur_eval(pd.strs_matrix_free);
        dealii::FEEvaluationGL<dim, fe_degree, dim> old_eval(pd.strs_matrix_free);
        dealii::FEEvaluationGL<dim, fe_degree> cur_vel(pd.vel_matrix_free);
        dealii::FEEvaluationGL<dim, fe_degree> old_vel(pd.vel_matrix_free);
        for (unsigned int cell = cell_range.first;
             cell < cell_range.second; 
             ++cell)
        {
            cur_eval.reinit(cell);
            old_eval.reinit(cell);
            cur_vel.reinit(cell);
            old_vel.reinit(cell);

            cur_eval.read_dof_values(src[0]);
            old_eval.read_dof_values(src[1]);
            cur_vel.read_dof_values_plain(src[2]);
            old_vel.read_dof_values_plain(src[3]);

            cur_eval.evaluate(true, false, false);
            old_eval.evaluate(true, false, false);
            cur_vel.evaluate(false, true, false);
            old_vel.evaluate(false, true, false);

            for (unsigned int q=0; q < cur_eval.n_q_points; ++q)
            {
                this->cur_strs = cur_eval.get_value(q);
                this->old_strs = old_eval.get_value(q);
                this->cur_grad_vel = cur_vel.get_gradient(q);
                this->old_grad_vel = old_vel.get_gradient(q);
                // Here's where you modify the time stepping!
                eval(cur_eval, q);
            }

            cur_eval.integrate(true, false);
            cur_eval.distribute_local_to_global(dst);
        }
    }
}
#endif
