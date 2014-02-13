#ifndef __viscosaur_bdf2_h
#define __viscosaur_bdf2_h
#include "boundary_cond.h"
#include "stress_op.h"
#include "scheme.h"

namespace viscosaur
{

    template <int dim, int fe_degree>
    class BDFTwoTentOp: public StressOp<dim, fe_degree>
    {
        public:
            virtual void eval(
                    dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
                    const unsigned int q);
    };

    template <int dim, int fe_degree>
    class BDFTwoCorrOp: public StressOp<dim, fe_degree>
    {
        public:
            virtual void eval(
                    dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
                    const unsigned int q);
 
    };

    template <int dim>
    class BDFTwoBC: public BoundaryCond<dim>
    {
        public:
            BDFTwoBC(BoundaryCond<dim> &p_bc, double p_time_step):
                BoundaryCond<dim>()
            {
                bc = &p_bc;
                time_step = p_time_step;
                this->t_ = bc->t_;
            }

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                //Compute v_{n+1} - v_n

                // Set the bc time.
                bc->set_t(this->t_);
                const double val1 = bc->value(p, component);
                // Back one time step.
                bc->set_t(this->t_ - time_step);
                const double val2 = bc->value(p, component);
                // Reset the bc time.
                bc->set_t(this->t_);
                return val1 - val2;
            }
            double time_step;
            BoundaryCond<dim>* bc;
    };

    //This line create BDFTwoTentOpFactory
    VISCOSAUR_OP_FACTORY(BDFTwoTentOp);

    //This line create BDFTwoCorrOpFactory
    VISCOSAUR_OP_FACTORY(BDFTwoCorrOp);

    template <int dim>
    class BDFTwo: public Scheme<dim>
    {
        public:
            BDFTwo(ProblemData<dim> &p_pd)
            {
                reinit(p_pd);
            }

            virtual void reinit(ProblemData<dim> &p_pd)
            {
                Scheme<dim>::reinit(p_pd);
                //init tent_op                
                this->tent_op_factory = new BDFTwoTentOpFactory<dim>();
                //init corr_op                
                this->corr_op_factory = new BDFTwoCorrOpFactory<dim>();
            }

            virtual void get_rhs_grad_terms(
                    dealii::FEValues<dim> &vel_fe_values,
                    Solution<dim> &soln,
                    std::vector<dealii::Tensor<1, dim> >& retval)
            {
                vel_fe_values.get_function_gradients(soln.old_vel, retval);
            }
            

            virtual double poisson_rhs_factor() const
            {
                return 1.5;
            }

            virtual void handle_poisson_soln(Solution<dim> &soln,
                dealii::PETScWrappers::MPI::Vector& poisson_soln) const
            {
                //Does the same thing as FwdEuler, but didn't in the past. 
                //Might use this again in the future so keep it around.
                soln.poisson_soln.reinit(soln.cur_vel);
                soln.poisson_soln = poisson_soln;
                soln.cur_vel = soln.poisson_soln;
            }

            virtual BoundaryCond<dim>* handle_bc(BoundaryCond<dim> &bc)
                const
            {
                //Does the same thing as FwdEuler, but didn't in the past. 
                //Might use this again in the future so keep it around.
                return &bc;
            }
    };



    template <int dim, int fe_degree>
    void
    BDFTwoTentOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
         const unsigned int q)
    {
        dealii::VectorizedArray<double> one_pt_five =
            dealii::make_vectorized_array(1.5);
        dealii::VectorizedArray<double> two = 
            dealii::make_vectorized_array(2.0);
        dealii::VectorizedArray<double> pt_five =
            dealii::make_vectorized_array(0.5);

        dealii::Point<dim, dealii::VectorizedArray<double> > p = 
            cur_eval.quadrature_point(q);

        dealii::Tensor<1, dim, dealii::VectorizedArray<double> > f_val, 
            fp_val, term;
        dealii::VectorizedArray<double> iv, ivd;
        dealii::Tensor<1, dim, dealii::VectorizedArray<double> > guess
            = this->cur_strs;

        const unsigned int iter_max = 30;
        const double abs_tol = 1e-6;
        double rel_tol = 0;
        for(int d = 0; d < dim; d++) 
        {
            for(unsigned int array_el = 0; array_el < 
                    guess[0].n_array_elements; array_el++)
            {
                rel_tol += 1e-9 * abs(guess[d][array_el]);
            }
        }
        rel_tol /= dim * guess[0].n_array_elements;


        for(unsigned int iter = 0; iter < iter_max; iter++)
        {
            //Replace this with a generic fnc call.
            iv = this->pd->inv_visc->value(p, guess);
            //Calculate the current value of the function.
            f_val = (one_pt_five * guess - two * this->cur_strs + 
                        pt_five * this->old_strs) + 
                this->mu_dt * iv * guess -
                this->mu_dt * this->old_grad_vel;

            // std::cout << "Guess " << iter << ":" << guess[0][0] << std::endl << 
            //     "     with residual: " << f_val[0][0] << std::endl <<
            //     "     and tolerance: " << rel_tol << std::endl <<
            //     "     and cur: " << guess[0][0] << std::endl <<
            //     "     and old: " << this->old_strs[0][0] << std::endl;
            double residual = 0; 
            for(int d = 0; d < dim; d++) 
            {
                for(unsigned int array_el = 0; array_el < 
                        guess[0].n_array_elements; array_el++)
                {
                    residual += abs(f_val[d][array_el]);
                }
            }
            if ((residual < rel_tol) || (residual < abs_tol))
            {
                // std::cout << "Convergence!" << std::endl;
                break;
            }
            //Replace this with a generic fnc call.
            //Calculate the current derivative of the function.
            for(int d = 0; d < dim; d++)
            {
                ivd = this->pd->inv_visc->strs_deriv(p, guess, d);
                term[d] = guess[d] * ivd;
            }
            fp_val = one_pt_five * this->tensor_one + 
                this->mu_dt * (iv * this->tensor_one + term);
            for(int d = 0; d < dim; d++) 
            {
                for(unsigned int array_el = 0; array_el < 
                        guess[0].n_array_elements; array_el++)
                {
                    guess[d][array_el] -= f_val[d][array_el] /
                        fp_val[d][array_el];
                }
            }
        }
        cur_eval.submit_value(guess, q);
    }



    template <int dim, int fe_degree>
    void
    BDFTwoCorrOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree, dim> &cur_eval,
         const unsigned int q)
    {
        cur_eval.submit_value(this->cur_strs + (2.0 / 3.0) * this->mu_dt * 
            (this->cur_grad_vel - this->old_grad_vel), q);
    }
}
#endif
