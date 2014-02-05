#ifndef __viscosaur_analytic_h
#define __viscosaur_analytic_h
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <gsl/gsl_integration.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/utilities.h>
#include "linear_algebra.h"
#include "problem_data.h"
#include "boundary_cond.h"

namespace boost 
{
    template<class T, std::size_t N> class array;
}

namespace viscosaur
{

    class SlipFnc
    {
        public:
            SlipFnc(double D);
            virtual double call(double z) = 0;
        protected:
            double D;
    };

    class ConstantSlipFnc: public SlipFnc
    {
        public:
            ConstantSlipFnc(double D): SlipFnc(D) {}
            double call(double z);
    };

    class CosSlipFnc: public SlipFnc
    {
        public:
            CosSlipFnc(double D): SlipFnc(D) {}
            double call(double z);
    };

    class TwoLayerAnalytic
    {
        public:
            TwoLayerAnalytic(double fault_slip,
                             double fault_depth,
                             double shear_modulus,
                             double viscosity,
                             SlipFnc &slip_fnc);
            ~TwoLayerAnalytic();

            double simple_velocity(double x, double y, double t);
            double simple_Szx(double x, double y) const;
            double simple_Szy(double x, double y) const;
            double integral_velocity(double x, double y, double t); 
            double integral_Szx(double x, double y) const;
            double integral_Szy(double x, double y) const;

        private:
            TwoLayerAnalytic(const TwoLayerAnalytic&) {}
            double fault_slip;
            double fault_depth;
            double shear_modulus;
            double viscosity;
            SlipFnc* slip_fnc;
            int images;

            gsl_integration_workspace* integration;
    };
    //For efficiency, I could move the TLA behavior into these classes.
    //or at least write separate functions for Szx and Szy
    template <int dim>
    class InitStress: public dealii::Function<dim>
    {
        public:
            InitStress(TwoLayerAnalytic &p_tla): 
                dealii::Function<dim>(dim)
            {
                tla = &p_tla;
            } 

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                if (component == 0)
                {
                    return tla->integral_Szx(p(0), p(1));
                } else
                {
                    return tla->integral_Szy(p(0), p(1));
                }
            }
        private:
            TwoLayerAnalytic* tla;
    };

    template <int dim>
    class SimpleInitStress: public dealii::Function<dim>
    {
        public:
            SimpleInitStress(TwoLayerAnalytic &p_tla): 
                dealii::Function<dim>(dim)
            {
                tla = &p_tla;
            } 

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                if (component == 0)
                {
                    return tla->simple_Szx(p(0), p(1));
                } else
                {
                    return tla->simple_Szy(p(0), p(1));
                }
            }
        private:
            TwoLayerAnalytic* tla;
    };

    template <int dim>
    class ExactVelocity: public dealii::Function<dim>
    {
        public:
            ExactVelocity(TwoLayerAnalytic &p_tla): 
                dealii::Function<dim>(1)
            {
                tla = &p_tla;
                t = 0;
            } 

            void set_t(double p_t)
            {
                t = p_t;
            }

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                return tla->integral_velocity(p(0), p(1), t);
            }
        private:
            TwoLayerAnalytic* tla;
            double t;
    };

    template <int dim>
    class SimpleVelocity: public BoundaryCond<dim>
    {
        public:
            SimpleVelocity(TwoLayerAnalytic &p_tla):
                BoundaryCond<dim>()
            {
                tla = &p_tla;
            } 

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                return tla->simple_velocity(p(0), p(1), this->t_);
            }
        private:
            TwoLayerAnalytic* tla;
    };
}
#endif
