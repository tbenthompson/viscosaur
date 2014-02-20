
#ifndef __viscosaur_boundary_cond_h
#define __viscosaur_boundary_cond_h
#include "tla.h"
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

    template <int dim>
    class FarFieldPlateBC: public BoundaryCond<dim>
    {
        public:
            FarFieldPlateBC(const double val, const double max_x,
                            const double f_depth, TLA::SlipFnc slip_fnc):
                BoundaryCond<dim>()
            {
                farfield = val; 
                distance = max_x;
                fault_depth = f_depth;
            }

            virtual double value(const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                //Bottom of fault and the far field both slip at a slow rate
                if(p[0] < distance)
                {
                    if(p[1] >= fault_depth)
                    {
                        return farfield;
                    }
                }
                else
                {
                    return farfield;
                }
                return 0;
            }

            double farfield;
            double distance;
            double fault_depth;
    };
}

#endif
