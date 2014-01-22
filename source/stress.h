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
            void step(Solution<dim> &soln);

        private:
            void init();

            dealii::ConstraintMatrix     constraints;
            dealii::MatrixFree<dim,double> matrix_free_szx;
            dealii::MatrixFree<dim,double> matrix_free_szy;
            ProblemData<dim>* pd;
            StressOp<dim, fe_degree>* op_szx;
            StressOp<dim, fe_degree>* op_szy;

            double time;
            double time_step;
            unsigned int timestep_number;
    };
}
#endif
