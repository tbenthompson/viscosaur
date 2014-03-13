#ifndef __viscosaur_solution_h
#define __viscosaur_solution_h

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/distributed/solution_transfer.h>

namespace dealii
{
    template <int dim> class Function;
}
namespace viscosaur
{
    template <int dim> class ProblemData;

    template <int dim>
    class Solution
    {
        public:
            Solution(ProblemData<dim> &p_pd);

            void reinit();

            void apply_init_cond(dealii::Function<dim> &init_mem,
                                 dealii::Function<dim> &init_disp);

            void output(std::string data_dir,
                        std::string filename,
                        dealii::Function<dim> &disp);

            void start_timestep();

            void start_refine();

            void post_refine(Solution<dim> &soln);

            dealii::parallel::distributed::Vector<double> cur_mem;
            dealii::parallel::distributed::Vector<double> old_mem;

            dealii::parallel::distributed::Vector<double> cur_disp;
            dealii::parallel::distributed::Vector<double> old_disp;
            dealii::parallel::distributed::Vector<double> old_old_disp;

            ProblemData<dim>* pd;
            std::vector<dealii::parallel::distributed::SolutionTransfer<dim, 
                    dealii::parallel::distributed::Vector<double> >* > 
                        sol_trans;
    };
}
#endif
