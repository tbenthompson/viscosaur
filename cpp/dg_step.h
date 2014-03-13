#ifndef __viscosaur_dg_step_h
#define __viscosaur_dg_step_h
#include "matrix_free_calculation.h"
#include "op_factory.h"
#include "inv_visc.h"
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
namespace viscosaur
{
    template <int dim> class ProblemData;
    template <int dim> class Solution;

    template <int dim>
    struct EvalData
    {
        dealii::VectorizedArray<double> mu;
        dealii::VectorizedArray<double> inv_rho;
        double C;
        double dt;
        InvViscosity<dim>* iv;
    };

    template <int dim, int fe_degree>
    class MemEvalDeriv
    {
        public:
        void hp_local_apply(ProblemData<dim> &pd, 
             dealii::parallel::distributed::Vector<double> &dst,
             const std::vector<
                    dealii::parallel::distributed::Vector<double>* > &src,
             const std::pair<unsigned int, unsigned int> &cell_range,
             boost::any data)
        {
            // Data structures for the cell assembly
            EvalData<dim>* d = boost::any_cast<EvalData<dim>*>(data);
            dealii::FEEvaluationGL<dim, fe_degree> 
                disp_eval(pd.disp_matrix_free);
            dealii::FEEvaluationGL<dim, fe_degree, dim> 
                mem_eval(pd.mem_matrix_free);
            const unsigned int n_q_points = mem_eval.n_q_points;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > 
                old_mem;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > 
                grad_disp;
            dealii::VectorizedArray<double> iv;

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                disp_eval.reinit(cell);
                mem_eval.reinit(cell);
                disp_eval.read_dof_values(*src[0]);
                mem_eval.read_dof_values(*src[2]);
                disp_eval.evaluate(false, true, false);
                mem_eval.evaluate(true, false, false);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    //Eval deriv here
                    old_mem = mem_eval.get_value(q);
                    grad_disp = disp_eval.get_gradient(q);
                    iv = d->iv->value(mem_eval.quadrature_point(q),
                                grad_disp);
                    mem_eval.submit_value(d->mu * iv * (grad_disp 
                                          - old_mem), q);
                }
                mem_eval.integrate(true, false);
                mem_eval.distribute_local_to_global(dst);
            }
        }
    };

    template <int dim, int fe_degree>
    class DispEvalDeriv
    {
        public:
        void hp_local_apply(ProblemData<dim> &pd, 
             dealii::parallel::distributed::Vector<double> &dst,
             const std::vector<
                    dealii::parallel::distributed::Vector<double>* > &src,
             const std::pair<unsigned int, unsigned int> &cell_range,
             boost::any data)
        {
            //Data structures for the cell assembly
            EvalData<dim>* d = boost::any_cast<EvalData<dim>*>(data);
            const dealii::VectorizedArray<double> delta_t_sqr =
                dealii::make_vectorized_array(d->dt);

            dealii::FEEvaluationGL<dim, fe_degree>
                cur_disp_eval(pd.disp_matrix_free);
            dealii::FEEvaluationGL<dim, fe_degree>
                old_disp_eval(pd.disp_matrix_free);
            dealii::FEEvaluationGL<dim, fe_degree, dim>
                mem_eval(pd.mem_matrix_free);
            const unsigned int n_q_points = cur_disp_eval.n_q_points;
            dealii::VectorizedArray<double> old_disp;
            dealii::VectorizedArray<double> cur_disp;
            dealii::VectorizedArray<double> div_mem;

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                cur_disp_eval.reinit(cell);
                old_disp_eval.reinit(cell);
                mem_eval.reinit(cell);
                cur_disp_eval.read_dof_values(*src[0]);
                old_disp_eval.read_dof_values(*src[1]);
                mem_eval.read_dof_values(*src[2]);
                cur_disp_eval.evaluate(true, true, false);
                old_disp_eval.evaluate(true, false, false);
                mem_eval.evaluate(false, true, false);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    cur_disp = cur_disp_eval.get_value(q);
                    old_disp = old_disp_eval.get_value(q);
                    div_mem = mem_eval.get_divergence(q);
                    cur_disp_eval.submit_value(2.0 * cur_disp - 
                        old_disp -
                        d->inv_rho * d->mu * delta_t_sqr *
                        div_mem, q);
                    cur_disp_eval.submit_gradient(
                        -d->inv_rho * d->mu * delta_t_sqr *
                        cur_disp_eval.get_gradient(q), q);
                }
                cur_disp_eval.integrate(true, true);
                cur_disp_eval.distribute_local_to_global(dst);
            }
        }
    };

    //This line creates ProjectionOpFactory
    VISCOSAUR_OP_FACTORY(MemEvalDeriv);
    VISCOSAUR_OP_FACTORY(DispEvalDeriv);

    template <int dim> class ProblemData;

    template <int dim>
    class Stepper
    {
        public:
            Stepper(ProblemData<dim> &p_pd);
            void step(Solution<dim> &soln,
                      InvViscosity<dim> &iv,
                      double inv_rho,
                      double dt,
                      dealii::Function<dim>& bc_fault,
                      dealii::Function<dim>& bc_plate);

            ProblemData<dim>* pd;
            dealii::ConstraintMatrix mem_constraints;
            dealii::ConstraintMatrix disp_constraints;
    };
}
#endif
