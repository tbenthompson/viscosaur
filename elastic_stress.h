
#include <boost/array.hpp>

namespace viscosaur
{
    class TwoLayerAnalytic
    {
        public:
            TwoLayerAnalytic(double fault_slip,
                             double fault_depth,
                             double shear_modulus,
                             double viscosity);
            ~TwoLayerAnalytic();

            double velocity(double x, double y, double t);
            boost::array<double, 2> initial_stress(double x, double y);
        
        private:
            double fault_slip;
            double fault_depth;
            double shear_modulus;
            double viscosity;
            int images;
    };
}
