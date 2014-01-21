#ifndef __viscosaur_stress_h
#define __viscosaur_stress_h

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace viscosaur
{
    template <int dim> class ProblemData; 
    
    template <int dim>
    class Stress
    {
        public:
            Stress(ProblemData<dim> &p_pd);
            void run();

        private:
            void init();
            void output_results(const unsigned int timestep_number);

            dealii::ConstraintMatrix     constraints;
            dealii::MatrixFree<dim,double> matrix_free_data;
            dealii::parallel::distributed::Vector<double> solution;
            dealii::parallel::distributed::Vector<double> old_solution;
            dealii::parallel::distributed::Vector<double> old_old_solution;
            ProblemData<dim>* pd;

            const unsigned int n_global_refinements;
            double time, time_step;
            const double final_time;
            const double cfl_number;
            const unsigned int output_timestep_skip;
    };
}
#endif
