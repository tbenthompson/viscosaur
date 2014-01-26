#ifndef __viscosaur_stress_h
#define __viscosaur_stress_h

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>


namespace viscosaur
{
    template <int dim> class ProblemData; 
    template <int dim> class Solution; 
    template <int dim, int fe_degree> class StressOp;

    const unsigned int fe_degree = 2;
    
    template <int dim>
    class Stress
    {
        public:
            Stress(Solution<dim> &soln,
                   ProblemData<dim> &p_pd);
            ~Stress();
            void generic_step(
                 dealii::parallel::distributed::Vector<double> &input,
                 dealii::parallel::distributed::Vector<double> &output,
                 Solution<dim> &soln,
                 unsigned int component,
                 StressOp<dim, fe_degree> &op);
            void tentative_step(Solution<dim> &soln);
            void correction_step(Solution<dim> &soln);
        private:
            dealii::ConstraintMatrix     constraints;
            dealii::MatrixFree<dim,double> matrix_free;
            ProblemData<dim>* pd;
            StressOp<dim, fe_degree>* t_step;
            StressOp<dim, fe_degree>* c_step;

            double time;
            double time_step;
            unsigned int timestep_number;
    };
}
#endif
