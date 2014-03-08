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
        InvViscosity<dim>* iv;
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
            // Data structures for the cell assembly
            EvalData<dim>* d = boost::any_cast<EvalData<dim>*>(data);
            dealii::FEEvaluationGL<dim, fe_degree> disp_eval(pd.vel_matrix_free);
            const unsigned int n_q_points = vel_eval.n_q_points;
            dealii::VectorizedArray<double> laplacian_disp;

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                disp_eval.reinit(cell);
                disp_eval.read_dof_values(*src[1]);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    disp_eval.submit_value(d->mu * d->inv_rho * 
                            disp_eval.get_laplacian(q), q);
                }
                disp_eval.integrate(true, false);
                disp_eval.distribute_local_to_global(dst);
            }
        }
    };

    template <int dim, int fe_degree>
    class VelEvalDeriv
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
            dealii::FEEvaluationGL<dim, fe_degree> vel_eval(pd.vel_matrix_free);
            dealii::FEEvaluationGL<dim, fe_degree> disp_eval(pd.vel_matrix_free);
            const unsigned int n_q_points = vel_eval.n_q_points;
            dealii::VectorizedArray<double> strs_div;

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                vel_eval.reinit (cell);
                strs_eval.reinit (cell);
                strs_eval.read_dof_values(*src[1]);
                strs_eval.evaluate(false, true, false);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    //Eval deriv here
                    strs_div = strs_eval.get_divergence(q);
                    vel_eval.submit_value(-d->inv_rho * strs_div, q);
                }
                vel_eval.integrate(true, false);
                vel_eval.distribute_local_to_global(dst);
            }
        }
    };

    //This line creates ProjectionOpFactory
    VISCOSAUR_OP_FACTORY(StrsEvalDeriv);
    VISCOSAUR_OP_FACTORY(VelEvalDeriv);

    template <int dim>
    class Stepper
    {
        public:
            Stepper(double dt);
            void step(ProblemData<dim> &pd,
                      Solution<dim> &soln,
                      InvViscosity<dim> &iv,
                      double inv_rho);
            double dt;
    };
}
#endif
