#include <string>
#include <boost/array.hpp>
#include <gsl/gsl_integration.h>

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
            boost::array<double, 2> integral_stress(double x, double y);

        private:
            double fault_slip;
            double fault_depth;
            double shear_modulus;
            double viscosity;
            SlipFnc* slip_fnc;
            int images;

            gsl_integration_workspace* integration;
    };
}
