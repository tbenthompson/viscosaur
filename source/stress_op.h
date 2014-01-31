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
#include "solution.h"
#include "problem_data.h"
#include "inv_visc.h"

namespace viscosaur
{
    namespace bp = boost::python;

    template <int dim, int fe_degree>
    class StressOp
    {
        public:
            StressOp() {}

            StressOp(ProblemData<dim> &p_pd)
            {init(p_pd);}

            /* Pre-build some of the structures necessary for efficient 
             * updating of the stress.
             */
            void init(ProblemData<dim> &p_pd);

            /* Compute and invert the diagonal mass matrix produced by the 
             * GLL quadrature and interpolation.
             */
            void compute_mass_matrix();

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

            ProblemData<dim>* pd;
            dealii::parallel::distributed::Vector<double> inv_mass_matrix;
            dealii::VectorizedArray<double> mu_dt;
            dealii::VectorizedArray<double> one;

            Solution<dim>* soln;
            unsigned int component;
            dealii::VectorizedArray<double> old_val;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> >
                old_grad_vel;

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
    void
    StressOp<dim, fe_degree>::
    init(ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        dealii::TimerOutput::Scope t(pd->computing_timer, "setup_stress");
        one = dealii::make_vectorized_array(1.);
        const double shear_modulus = 
            bp::extract<double>(pd->parameters["shear_modulus"]);
        const double time_step = 
            bp::extract<double>(pd->parameters["time_step"]);
        mu_dt = dealii::make_vectorized_array(shear_modulus * time_step);
        compute_mass_matrix();
    }


    template <int dim, int fe_degree>
    void
    StressOp<dim, fe_degree>::
    compute_mass_matrix()
    {
        pd->matrix_free.initialize_dof_vector(inv_mass_matrix);
        dealii::FEEvaluationGL<dim,fe_degree> fe_eval(pd->matrix_free);
        const unsigned int n_q_points = fe_eval.n_q_points;

        for(unsigned int cell=0; 
            cell < pd->matrix_free.n_macro_cells();
            ++cell)
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
        pd->matrix_free.cell_loop(&StressOp<dim,fe_degree>::local_apply,
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
}
#endif
