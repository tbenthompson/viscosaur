#define PI 3.14159265358979323846
#include <boost/array.hpp>
#include <math.h>
#include "elastic_stress.h"

namespace viscosaur
{
    TwoLayerAnalytic::TwoLayerAnalytic(double fault_slip,
                             double fault_depth,
                             double shear_modulus,
                             double viscosity)
    {
        this->fault_slip = fault_slip;
        this->fault_depth = fault_depth;
        this->shear_modulus = shear_modulus;
        this->viscosity = viscosity;
        this->images = 20;
    }

    TwoLayerAnalytic::~TwoLayerAnalytic()
    {
    }

    double factorial(int n)
    {
        if(n <= 0)
        {
            return 1; // safeguard 0 and -ve
        }
        double res = n;
        while(--n>1)
        {
            res *= n;
        }
        return res;
    }

    /* Returns the velocity produced by the Nur and Mavko model
     * See the Segall (2010) book for more details. Savage (2000) is
     * another good reference.
     */
    double TwoLayerAnalytic::velocity(double x, double y, double t)
    {
        double v = 0.0;
        double x_scaled = x / this->fault_depth;
        double y_scaled = y / this->fault_depth;
        double t_r = (2 * this->viscosity) / this->shear_modulus;

        double factor;
        double fact;
        double term, term1, term2, term3, term4;
        // Compute the response for each image.
        for (int m = 1; m < this->images; m++)
        {
            fact = factorial(m - 1);
            factor = pow(t / t_r, m - 1) / fact;
            assert(!isnan(factor));
            if (y_scaled > 1) 
            {
                // Deeper than the fault bottom.
                term1 = atan((2 * m + 1 + y_scaled) / x_scaled);
                term2 = -atan((2 * m - 3 + y_scaled) / x_scaled);
                term = (1.0 / (2.0 * PI)) * (term1 + term2);
            }
            else
            {
                // Shallower than the fault bottom
                term1 = atan((2 * m + 1 + y_scaled) / x_scaled);
                term2 = -atan((2 * m - 1 + y_scaled) / x_scaled);
                term3 = atan((2 * m + 1 - y_scaled) / x_scaled);
                term4 = -atan((2 * m - 1 - y_scaled) / x_scaled);
                term = (1.0 / (2.0 * PI)) * (term1 + term2 + term3 + term4);
            }
            v += factor * term;
        }
        v *= exp(-t / t_r) * (1.0 / t_r);
        assert(!isnan(v));
        return v;
    }

    boost::array<double, 2> TwoLayerAnalytic::initial_stress(double x, 
                                                             double y) 
    {
        double factor, main_term, image_term, Szx, Szy;
        factor = (this->fault_slip * this->shear_modulus) / (2 * PI);
        main_term = (y - this->fault_depth) / 
            (pow((y - this->fault_depth), 2) + pow(x, 2));
        image_term = -(y + this->fault_depth) / 
            (pow((y + this->fault_depth), 2) + pow(x, 2));
        Szx = factor * (main_term + image_term);

        main_term = -x / (pow(x, 2) + pow((y - this->fault_depth), 2));
        image_term = x / (pow(x, 2) + pow((y + this->fault_depth), 2));
        Szy = factor * (main_term + image_term);

        //Inner braces for elements of array,
        //Outer braces for struct initialization
        boost::array<double, 2> retval = {{Szx, Szy}};
        return retval;
    }
}
