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
                const unsigned int comp,
                const double time_step);

            /* The partner in crime of the "apply" function above. This computes
             * one time step for one cell. What a messy declaration!
             */
            void local_apply(const dealii::MatrixFree<dim> &data,
                             dealii::parallel::distributed::Vector<double> &dst,
                             const dealii::parallel::distributed::Vector
                                    <double> &src,
                             const std::pair<unsigned int,
                                             unsigned int> &cell_range);

            virtual void eval(dealii::FEEvaluationGL<dim, fe_degree, dim> 
                    &cur_eval, const unsigned int q) = 0;

            ProblemData<dim>* pd;
            Solution<dim>* soln;
            dealii::parallel::distributed::Vector<double> inv_mass_matrix;
            dealii::VectorizedArray<double> mu_dt;
            dealii::VectorizedArray<double> one;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > tensor_one;

            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > cur_strs;
            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > old_strs;
            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > cur_grad_vel;
            dealii::Tensor<1, dim, 
                dealii::VectorizedArray<double> > old_grad_vel;
    };




    template <int dim, int fe_degree>
    void
    StressOp<dim, fe_degree>::
    init(ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        dealii::TimerOutput::Scope t(pd->computing_timer, "setup_stress");
        one = dealii::make_vectorized_array(1.0);
        for(int d = 0; d < dim; d++) 
        {
            for(unsigned int array_el = 0; array_el < 
                    tensor_one[0].n_array_elements; array_el++)
            {
                tensor_one[d][array_el] = 1.0;
            }
        }
        compute_mass_matrix();
    }


    template <int dim, int fe_degree>
    void
    StressOp<dim, fe_degree>::
    compute_mass_matrix()
    {
        // Integrate and invert the diagonal mass matrix resulting from the
        // Gauss Lobatto Lagrange elements and quadrature match up.
        pd->strs_matrix_free.initialize_dof_vector(inv_mass_matrix);
        dealii::FEEvaluationGL<dim, fe_degree, dim> fe_eval(pd->strs_matrix_free);
        const unsigned int n_q_points = fe_eval.n_q_points;

        for(unsigned int cell=0; 
            cell < pd->strs_matrix_free.n_macro_cells();
            ++cell)
        {
            fe_eval.reinit(cell);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                fe_eval.submit_value(tensor_one, q);
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
          const unsigned int comp,
          const double time_step)
    {
        const double shear_modulus = 
            bp::extract<double>(pd->parameters["shear_modulus"]);
        this->mu_dt = dealii::make_vectorized_array(shear_modulus * time_step);
        dst = 0;
        this->soln = &soln;
        pd->strs_matrix_free.cell_loop(&StressOp<dim,fe_degree>::local_apply,
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
        dealii::FEEvaluationGL<dim, fe_degree, dim> cur_eval(data);
        dealii::FEEvaluationGL<dim, fe_degree, dim> old_eval(cur_eval);
        dealii::FEEvaluationGL<dim, fe_degree> cur_vel(pd->vel_matrix_free);
        dealii::FEEvaluationGL<dim, fe_degree> old_vel(cur_vel);
        for (unsigned int cell = cell_range.first;
             cell < cell_range.second; 
             ++cell)
        {
            cur_eval.reinit(cell);
            old_eval.reinit(cell);
            cur_vel.reinit(cell);
            old_vel.reinit(cell);

            cur_eval.read_dof_values(input);
            old_eval.read_dof_values(this->soln->old_old_strs);
            cur_vel.read_dof_values_plain(this->soln->cur_vel_for_strs);
            old_vel.read_dof_values_plain(this->soln->old_vel_for_strs);

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
            cur_eval.distribute_local_to_global(output);
        }
    }
}
#endif
