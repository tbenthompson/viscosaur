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

            void apply_init_cond(dealii::Function<dim> &init_szx,
                                 dealii::Function<dim> &init_szy,
                                 dealii::Function<dim> &init_vel);
            void init_multistep(dealii::Function<dim> &init_szx,
                                dealii::Function<dim> &init_szy,
                                dealii::Function<dim> &init_vel);

            void output(std::string data_dir,
                        std::string filename,
                        dealii::Function<dim> &vel) const;

            void start_timestep();

            dealii::parallel::distributed::SolutionTransfer<dim, 
                dealii::parallel::distributed::Vector<double> >*
                    start_refine();

            void post_refine(
                    dealii::parallel::distributed::SolutionTransfer<dim, 
                        dealii::parallel::distributed::Vector<double> >*
                    sol_trans);

            dealii::parallel::distributed::Vector<double> cur_szx;
            dealii::parallel::distributed::Vector<double> cur_szy;
            dealii::parallel::distributed::Vector<double> tent_szx;
            dealii::parallel::distributed::Vector<double> tent_szy;
            dealii::parallel::distributed::Vector<double> old_szx;
            dealii::parallel::distributed::Vector<double> old_szy;
            dealii::parallel::distributed::Vector<double> old_old_szx;
            dealii::parallel::distributed::Vector<double> old_old_szy;

            dealii::parallel::distributed::Vector<double> cur_vel;
            dealii::parallel::distributed::Vector<double> cur_vel_for_strs;
            dealii::parallel::distributed::Vector<double> poisson_soln;
            dealii::parallel::distributed::Vector<double> old_vel;
            dealii::parallel::distributed::Vector<double> old_vel_for_strs;

            ProblemData<dim>* pd;
    };
}
#endif
