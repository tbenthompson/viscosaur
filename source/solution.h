#ifndef __viscosaur_solution_h
#define __viscosaur_solution_h

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include "linear_algebra.h"

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

            void apply_init_cond(dealii::Function<dim> &init_szx,
                     dealii::Function<dim> &init_szy);

            std::string output_filename(const unsigned int cycle,
                                        const unsigned int subdomain) const;
            void output(const unsigned int cycle,
                        dealii::Function<dim> &vel) const;

            dealii::parallel::distributed::Vector<double> cur_szx;
            dealii::parallel::distributed::Vector<double> cur_szy;
            dealii::parallel::distributed::Vector<double> tent_szx;
            dealii::parallel::distributed::Vector<double> tent_szy;
            dealii::parallel::distributed::Vector<double> old_szx;
            dealii::parallel::distributed::Vector<double> old_szy;

            dealii::parallel::distributed::Vector<double> cur_vel;
            dealii::parallel::distributed::Vector<double> cur_vel_for_strs;
            dealii::parallel::distributed::Vector<double> old_vel;

            ProblemData<dim>* pd;
    };
}
#endif
