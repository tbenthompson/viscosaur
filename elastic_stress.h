#include <boost/array.hpp>
#include <gsl/gsl_integration.h>

namespace viscosaur
{
    class TwoLayerAnalytic
    {
        public:
            TwoLayerAnalytic(double fault_slip,
                             double fault_depth,
                             double shear_modulus,
                             double viscosity,
                             double (*slip_fnc)(double));
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
            double (*slip_fnc)(double);
            int images;

            gsl_integration_workspace* integration;
    };
}
