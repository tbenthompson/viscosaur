#ifndef __viscosaur_init_cond_h
#define __viscosaur_init_cond_h
#include <deal.II/base/function.h>
namespace viscosaur
{
    template <int dim>
    class GaussStress: public dealii::Function<dim>
    {
        public:
            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                return exp(-((p[0] * p[0]) + (p[1] * p[1])));
            }
    };
}
#endif
