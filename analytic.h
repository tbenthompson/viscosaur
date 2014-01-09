#include <string>
#include <boost/array.hpp>

class gsl_integration_workspace;
namespace viscosaur
{
    class SlipFnc
    {
        public:
            SlipFnc(double D);
            virtual double call(double z) = 0;
        private:
            double D;
    };

    class ConstantSlipFnc
    {
        public:
            double call(double z);
    };

    class CosSlipFnc
    {
        public:
            double call(double z);
    };

    class TwoLayerAnalytic
    {
        public:
            TwoLayerAnalytic(double fault_slip,
                             double fault_depth,
                             double shear_modulus,
                             double viscosity,
                             SlipFnc* slip_fnc);
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
