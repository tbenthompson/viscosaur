#ifndef __viscosaur_fwd_euler_h
#define __viscosaur_fwd_euler_h

#include "scheme.h"
#include <deal.II/base/function.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace viscosaur
{
    template <int dim, int fe_degree>
    class FwdEulerTentOp: public StressOp<dim, fe_degree>
    {
        public:
            FwdEulerTentOp(ProblemData<dim> &p_pd)
            {this->init(p_pd);}
            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);
    };

    template <int dim, int fe_degree>
    class FwdEulerCorrOp: public StressOp<dim, fe_degree>
    {
        public:
            FwdEulerCorrOp(ProblemData<dim> &p_pd)
            {this->init(p_pd);}
            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);
    };

    template <int dim>
    class FwdEuler: public Scheme<dim>
    {
        public:
            FwdEuler(ProblemData<dim> &p_pd):
                Scheme<dim>(p_pd)
            {
                //init tent_op                
                this->tent_op = new FwdEulerTentOp<dim, FE_DEGREE>(p_pd);
                //init corr_op                
                this->corr_op = new FwdEulerCorrOp<dim, FE_DEGREE>(p_pd);
            }

            virtual double poisson_rhs_factor() const
            {
                return 1.0;
            }
            virtual void handle_poisson_soln(Solution<dim> &soln,
                dealii::PETScWrappers::MPI::Vector& poisson_soln) const
            {
                soln.poisson_soln.reinit(soln.cur_vel);
                soln.poisson_soln = poisson_soln;
                soln.cur_vel = soln.poisson_soln; 
            }

            virtual BoundaryCond<dim>* handle_bc(BoundaryCond<dim> &bc)
                const
            {
                return &bc; 
            }
    };

    template <int dim, int fe_degree>
    void 
    FwdEulerTentOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
          dealii::VectorizedArray<double> &cur_val,
          dealii::Tensor<1, dim,
            dealii::VectorizedArray<double> > &grad_vel,
         unsigned int q)
    {
        const dealii::VectorizedArray<double> factor = 
            this->mu_dt * this->pd->inv_visc->value(fe_eval.quadrature_point(q), cur_val);
        fe_eval.submit_value((this->one - factor) * cur_val, q);
    }

    template <int dim, int fe_degree>
    void 
    FwdEulerCorrOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
          dealii::VectorizedArray<double> &cur_val,
          dealii::Tensor<1, dim,
            dealii::VectorizedArray<double> > &grad_vel,
         unsigned int q)
    {
        fe_eval.submit_value(cur_val + 
                this->mu_dt * grad_vel[this->component], q);
    }

}
#endif
