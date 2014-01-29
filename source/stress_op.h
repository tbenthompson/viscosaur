#ifndef __viscosaur_stress_op_h
#define __viscosaur_stress_op_h
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/parallel_vector.h>
#include <boost/python/extract.hpp>
#include "solution.h"
#include "problem_data.h"
#include "velocity.h"

namespace viscosaur
{
    namespace bp = boost::python;

    template <int dim, int fe_degree>
    class StressOp
    {
        public:
            StressOp() {}
            StressOp(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc)
            {init(data_in, p_time_step, p_pd, p_inv_visc);}
            void init(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc); 

            /* The main function of the class computes one time step. Call the
             * local_apply function for every cell. Then, uses the inverse mass
             * matrix to compute the solution. Because of the Gauss Lobatto 
             * lagrange interpolation, the mass matrix is diagonal and can
             * be easily inverted.
             */
            void apply(dealii::parallel::distributed::Vector<double> &dst, 
                const dealii::parallel::distributed::Vector<double> &src,
                Solution<dim> &soln,
                const unsigned int comp);

            const dealii::MatrixFree<dim,double>* data;
            dealii::parallel::distributed::Vector<double> inv_mass_matrix;
            ProblemData<dim>* pd;
            Solution<dim>* soln;
            unsigned int component;
            dealii::VectorizedArray<double> old_val;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> >
                old_grad_vel;
            double time_step;
            double shear_modulus;
            InvViscosity<dim>* inv_visc;
            dealii::VectorizedArray<double> mu_dt;
            dealii::VectorizedArray<double> one;

            /* The partner in crime of the "apply" function above. This computes
             * one time step for one cell. What a messy declaration!
             */
            void local_apply(const dealii::MatrixFree<dim,double> &data,
                             dealii::parallel::distributed::Vector<double> &dst,
                             const dealii::parallel::distributed::Vector
                                    <double> &src,
                             const std::pair<unsigned int,
                                             unsigned int> &cell_range);
            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q) = 0;
    };

    template <int dim, int fe_degree>
    class TentativeOp: public StressOp<dim, fe_degree>
    {
        public:
            TentativeOp(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc)
            {this->init(data_in, p_time_step, p_pd, p_inv_visc);}
            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);
    };

    template <int dim, int fe_degree>
    class TentativeOp2: public StressOp<dim, fe_degree>
    {
        public:
            TentativeOp2(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc)
            {this->init(data_in, p_time_step, p_pd, p_inv_visc);}
            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);
    };

    template <int dim, int fe_degree>
    class CorrectionOp: public StressOp<dim, fe_degree>
    {
        public:
            CorrectionOp(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc)
            {this->init(data_in, p_time_step, p_pd, p_inv_visc);}
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
            CorrectionOp2(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc)
            {this->init(data_in, p_time_step, p_pd, p_inv_visc);}
            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
                              dealii::VectorizedArray<double> &cur_val,
                              dealii::Tensor<1, dim,
                                dealii::VectorizedArray<double> > &grad_vel,
                                unsigned int q);

    };

    template <int dim, int fe_degree>
    void
    StressOp<dim, fe_degree>::
    init(const dealii::MatrixFree<dim,double> &data_in, 
             const double p_time_step,
             ProblemData<dim> &p_pd,
             InvViscosity<dim> &p_inv_visc)
    {
        data = &data_in;
        pd = &p_pd;
        one = dealii::make_vectorized_array(1.);
        data->initialize_dof_vector(inv_mass_matrix);

        dealii::FEEvaluationGL<dim,fe_degree> fe_eval(*data);
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

        inv_mass_matrix.compress(dealii::VectorOperation::add);
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
        mu_dt = dealii::make_vectorized_array(
                this->shear_modulus * this->time_step);
        inv_visc = &p_inv_visc;
    }

    template <int dim, int fe_degree>
    void 
    StressOp<dim, fe_degree>::
    apply(dealii::parallel::distributed::Vector<double> &dst,
          const dealii::parallel::distributed::Vector<double> &src,
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
    StressOp<dim, fe_degree>::
    local_apply (const dealii::MatrixFree<dim> &data,
                 dealii::parallel::distributed::Vector<double> &output,
                 const dealii::parallel::distributed::Vector<double> &input,
                 const std::pair<unsigned int,unsigned int> &cell_range)
    {
        dealii::FEEvaluationGL<dim, fe_degree> current(data);
        dealii::FEEvaluationGL<dim, fe_degree> old(current);
        dealii::FEEvaluationGL<dim, fe_degree> old_vel(current);
        dealii::FEEvaluationGL<dim, fe_degree> vel(current);
        for (unsigned int cell = cell_range.first;
             cell < cell_range.second; 
             ++cell)
        {
            current.reinit(cell);
            old.reinit(cell);
            old_vel.reinit(cell);
            vel.reinit(cell);
            current.read_dof_values(input);
            if (this->component == 0)
            {
                old.read_dof_values(this->soln->old_old_szx);
            } else
            {
                old.read_dof_values(this->soln->old_old_szy);
            }
            vel.read_dof_values_plain(this->soln->cur_vel_for_strs);
            old_vel.read_dof_values_plain(this->soln->old_vel_for_strs);
            current.evaluate(true, false, false);
            old.evaluate(true, false, false);
            old_vel.evaluate(false, true, false);
            vel.evaluate(false, true, false);

            for (unsigned int q=0; q < current.n_q_points; ++q)
            {
                dealii::VectorizedArray<double> current_value = 
                        current.get_value(q);
                this->old_val = old.get_value(q);

                dealii::Tensor<1, dim, 
                      dealii::VectorizedArray<double> > grad_vel = 
                        vel.get_gradient(q);
                this->old_grad_vel = old_vel.get_gradient(q);

                // Here's where you modify the time stepping!
                eval(current, current_value, grad_vel, q);
            }

            current.integrate(true, false);
            current.distribute_local_to_global(output);
        }
    }

    template <int dim, int fe_degree>
    void 
    TentativeOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
          dealii::VectorizedArray<double> &cur_val,
          dealii::Tensor<1, dim,
            dealii::VectorizedArray<double> > &grad_vel,
         unsigned int q)
    {
        const dealii::VectorizedArray<double> factor = 
            this->mu_dt * this->inv_visc->value(fe_eval.quadrature_point(q), cur_val);
        fe_eval.submit_value((this->one - factor) * cur_val, q);
    }



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
    CorrectionOp<dim, fe_degree>::
    eval(dealii::FEEvaluationGL<dim, fe_degree> &fe_eval,
          dealii::VectorizedArray<double> &cur_val,
          dealii::Tensor<1, dim,
            dealii::VectorizedArray<double> > &grad_vel,
         unsigned int q)
    {
        fe_eval.submit_value(cur_val + 
                this->mu_dt * grad_vel[this->component], q);
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
#endif
