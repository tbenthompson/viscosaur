#ifndef __viscosaur_stress_h
#define __viscosaur_stress_h

#include "matrix_free_calculation.h"

namespace viscosaur
{
    template <int dim> class ProblemData; 
    template <int dim> class Solution; 
    template <int dim> class Scheme;
    template <int dim> class OpFactory;

    template <int dim>
    class Stress: public MatrixFreeCalculation<dim>
    {
        public:
            Stress(ProblemData<dim> &p_pd);
            ~Stress();

            /* Pre-build some of the structures necessary for efficient 
             * updating of the stress.
             */
            void reinit(ProblemData<dim> &p_pd);

            void tentative_step(Solution<dim> &soln, Scheme<dim> &scheme,
                    double time_step);
            void correction_step(Solution<dim> &soln, Scheme<dim> &scheme,
                    double time_step);
    };
}
#endif
