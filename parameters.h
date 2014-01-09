#ifndef __viscosaur_parameters_h
#define __viscosaur_parameters_h

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
namespace viscosaur
{
    using namespace dealii;

    template <int dim>
    class Parameters
    {
        public:
            Parameters();
            void declare_parameters (ParameterHandler &prm);
            void parse_parameters (ParameterHandler &prm);
            
            Point<dim> min_corner;
            Point<dim> max_corner;

            unsigned int initial_global_refinement;
            unsigned int initial_adaptive_refinement;
            double refinement_fraction;
            double coarsening_fraction;

    };
}

#endif
