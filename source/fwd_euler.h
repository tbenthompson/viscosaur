#ifndef __viscosaur_fwd_euler_h
#define __viscosaur_fwd_euler_h

#include "scheme.h"
#include <deal.II/base/function.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "stress_op.h"

namespace viscosaur
{
    template <int dim, int fe_degree>
    class FwdEulerTentOp: public StressOp<dim, fe_degree>
    {
        public:
            virtual void eval(
                    dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
                    const unsigned int q);
    };

    template <int dim, int fe_degree>
    class FwdEulerCorrOp: public StressOp<dim, fe_degree>
    {
        public:
            virtual void eval(
                    dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
                    const unsigned int q);
    };

    //This line create FwdEulerTentOpFactory
    VISCOSAUR_OP_FACTORY(FwdEulerTentOp);

    //This line create FwdEulerCorrOpFactory
    VISCOSAUR_OP_FACTORY(FwdEulerCorrOp);

    template <int dim>
    class FwdEuler: public Scheme<dim>
    {
        public:
            FwdEuler(ProblemData<dim> &p_pd)
            {
                reinit(p_pd);
            }

            virtual void reinit(ProblemData<dim> &p_pd)
            {
                Scheme<dim>::reinit(p_pd);
                //init tent_op                
                this->tent_op_factory = new FwdEulerTentOpFactory<dim>();
                //init corr_op                
                this->corr_op_factory = new FwdEulerCorrOpFactory<dim>();
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
    eval(dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
         const unsigned int q)
    {
        const dealii::VectorizedArray<double> factor = 
            this->mu_dt * this->pd->inv_visc->value(
                    cur_eval.quadrature_point(q), this->cur_strs);
        const dealii::Tensor<1, dim, dealii::VectorizedArray<double> > out =
            (this->one - factor) * this->cur_strs;
        cur_eval.submit_value(out, q);
    }

    template <int dim, int fe_degree>
    void 
    FwdEulerCorrOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
         const unsigned int q)
    {
        cur_eval.submit_value(this->cur_strs + this->mu_dt * this->cur_grad_vel, q);
    }
}
#endif
