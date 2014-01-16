#ifndef __viscosaur_analytic_h
#define __viscosaur_analytic_h
#include <deal.II/base/function.h>
#include <gsl/gsl_integration.h>

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
            boost::array<double, 2> simple_stress(double x, double y);
        
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
    class InitSzx: public dealii::Function<dim>
    {
        public:
            InitSzx(TwoLayerAnalytic &p_tla): 
                dealii::Function<dim>(1)
            {
                tla = &p_tla;
            } 

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                return tla->integral_Szx(p(0), p(1));
            }
        private:
            TwoLayerAnalytic* tla;
    };

    template <int dim>
    class InitSzy: public dealii::Function<dim>
    {
        public:
            InitSzy(TwoLayerAnalytic &p_tla): 
                dealii::Function<dim>(1)
            {
                tla = &p_tla;
            } 

            virtual double value (const dealii::Point<dim>   &p,
                                  const unsigned int  component) const
            {
                return tla->integral_Szy(p(0), p(1));
            }
        private:
            TwoLayerAnalytic* tla;
    };

    template <int dim>
    class Velocity: public dealii::Function<dim>
    {
        public:
            Velocity(TwoLayerAnalytic &p_tla): 
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
}
#endif
