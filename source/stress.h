#ifndef __viscosaur_stress_h
#define __viscosaur_stress_h

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>


namespace viscosaur
{
    template <int dim> class ProblemData; 
    template <int dim, int fe_degree> class StressOp;

    const unsigned int fe_degree = 2;
    
    template <int dim>
    class Stress
    {
        public:
            Stress(dealii::Function<dim> &init_szx, 
                   dealii::Function<dim> &init_szy, 
                   ProblemData<dim> &p_pd);
            ~Stress();
            void step();

        private:
            void init();

            dealii::ConstraintMatrix     constraints;
            dealii::MatrixFree<dim,double> matrix_free_szx;
            dealii::MatrixFree<dim,double> matrix_free_szy;
            dealii::parallel::distributed::Vector<double> szx;
            dealii::parallel::distributed::Vector<double> szy;
            dealii::parallel::distributed::Vector<double> old_szx;
            dealii::parallel::distributed::Vector<double> old_szy;
            ProblemData<dim>* pd;
            StressOp<dim, fe_degree>* op_szx;
            StressOp<dim, fe_degree>* op_szy;

            double time;
            double time_step;
            unsigned int timestep_number;
    };
}
#endif
