#ifndef __viscosaur_bdf2_h
#define __viscosaur_bdf2_h
namespace viscosaur
{
    template <int dim>
    class BDF2: public Scheme<dim>
    {

    };

    template <int dim, int fe_degree>
    class TentativeOp2: public StressOp<dim, fe_degree>
    {
        public:
            TentativeOp2(ProblemData<dim> &p_pd)
            {this->init(p_pd);}

            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);
    };

    template <int dim, int fe_degree>
    class CorrectionOp2: public StressOp<dim, fe_degree>
    {
        public:
            CorrectionOp2(ProblemData<dim> &p_pd)
            {this->init(p_pd);}

            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);

    };

    template <int dim, int fe_degree>
    void 
    TentativeOp2<dim, fe_degree>::
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
                iv = this->inv_visc->value(p, guess);
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
                ivd = this->inv_visc->strs_deriv(p, guess);
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
    CorrectionOp2<dim, fe_degree>::
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
#endif;
