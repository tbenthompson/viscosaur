
#ifndef __viscosaur_boundary_cond_h
#define __viscosaur_boundary_cond_h
#include <deal.II/base/function.h>

namespace viscosaur
{
    template <int dim>
    class BoundaryCond: public dealii::Function<dim>
    {
        public:
            BoundaryCond():
                dealii::Function<dim>(1)
            {
                t_ = 0;
            }

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const = 0;

            void set_t(const double p_t)
            {
                t_ = p_t;
            }

            double t_;
    };

    template <int dim>
    class ConstantBC: public BoundaryCond<dim>
    {
        public:
            ConstantBC(const double val):
                BoundaryCond<dim>()
            {
                retval = val; 
            }

            virtual double value(const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                return retval;
            }

            double retval;
    };
}

#endif
