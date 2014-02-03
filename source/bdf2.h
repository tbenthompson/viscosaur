#ifndef __viscosaur_bdf2_h
#define __viscosaur_bdf2_h
#include "boundary_cond.h"
#include "stress_op.h"

namespace viscosaur
{

    template <int dim, int fe_degree>
    class BDFTwoTentOp: public StressOp<dim, fe_degree>
    {
        public:
            BDFTwoTentOp(ProblemData<dim> &p_pd)
            {this->init(p_pd);}

            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);
    };

    template <int dim, int fe_degree>
    class BDFTwoCorrOp: public StressOp<dim, fe_degree>
    {
        public:
            BDFTwoCorrOp(ProblemData<dim> &p_pd)
            {this->init(p_pd);}

            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);

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
                this->tent_op = new BDFTwoTentOp<dim, FE_DEGREE>(p_pd);
                //init corr_op                
                this->corr_op = new BDFTwoCorrOp<dim, FE_DEGREE>(p_pd);
            }

            virtual double poisson_rhs_factor() const
            {
                return 1.5;
            }

            virtual void handle_poisson_soln(Solution<dim> &soln,
                dealii::PETScWrappers::MPI::Vector& poisson_soln) const
            {
                soln.poisson_soln.reinit(soln.cur_vel);
                soln.poisson_soln = poisson_soln;
                soln.cur_vel = soln.old_vel; 
                soln.cur_vel += soln.poisson_soln; 
            }

            virtual BoundaryCond<dim>* handle_bc(BoundaryCond<dim> &bc)
                const
            {
                const double time_step = 
                    bp::extract<double>(this->pd->parameters["time_step"]);
                return new BDFTwoBC<dim>(bc, time_step);
            }
    };



    template <int dim, int fe_degree>
    void 
    BDFTwoTentOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
          dealii::VectorizedArray<double> &cur_val,
          dealii::Tensor<1, dim,
            dealii::VectorizedArray<double> > &grad_vel,
         unsigned int q)
    {
        dealii::VectorizedArray<double> output;
        const unsigned int iter_max = 50;
        dealii::Point<dim, double> p;
        double f_val, fp_val, iv, ivd;
        const double abs_tol = 1e-6;
        for(unsigned int array_el = 0; array_el < 
                cur_val.n_array_elements; array_el++)
        {
            double guess = cur_val[array_el];
            const double rel_tol = 1e-9 * abs(cur_val[array_el]);
            for(int d = 0; d < dim; d++) 
            {
                p[d] = fe_eval.quadrature_point(q)[d][array_el];
            }
            for(unsigned int iter = 0; iter < iter_max; iter++)
            {
                //Replace this with a generic fnc call.
                iv = this->pd->inv_visc->value(p, guess);
                //Calculate the current value of the function.
                f_val = (1.5 * guess - 2 * cur_val[array_el] + 
                            0.5 * this->old_val[array_el]) + 
                    this->mu_dt[array_el] * iv * guess -
                    this->mu_dt[array_el] * 
                        this->old_grad_vel[this->component][array_el];

                // std::cout << "Guess " << iter << ":" << guess << std::endl << 
                //     "     with residual: " << f_val << std::endl <<
                //     "     and tolerance: " << rel_tol << std::endl <<
                //     "     and cur: " << cur_val[array_el] << std::endl <<
                //     "     and old: " << this->old_val[array_el] << std::endl;
                if ((abs(f_val) < rel_tol) || (abs(f_val) < abs_tol))
                {
                    // std::cout << "Convergence!" << std::endl;
                    break;
                }
                //Replace this with a generic fnc call.
                //Calculate the current derivative of the function.
                ivd = this->pd->inv_visc->strs_deriv(p, guess);
                fp_val = 1.5 + 
                    this->mu_dt[array_el] * (iv + guess * ivd);
                guess -= f_val / fp_val;
            }
            output[array_el] = guess;
        }
        fe_eval.submit_value(output, q);
    }



    template <int dim, int fe_degree>
    void 
    BDFTwoCorrOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
          dealii::VectorizedArray<double> &cur_val,
          dealii::Tensor<1, dim,
            dealii::VectorizedArray<double> > &grad_vel,
         unsigned int q)

    {
        fe_eval.submit_value(cur_val + (2.0 / 3.0) * 
                this->mu_dt * (grad_vel[this->component] -
                    this->old_grad_vel[this->component]), q);
    }
}
#endif
