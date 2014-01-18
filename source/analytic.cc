// #define PI 3.14159265358979323846
#include "analytic.h"
#include <boost/array.hpp>
#include <math.h>

namespace viscosaur
{
    SlipFnc::SlipFnc(double D)
    {
        this->D = D;
    }

    double ConstantSlipFnc::call(double z)
    {
        if (z > this->D)
        {
            return 0.0;
        }
        return 1.0;
    }

    double CosSlipFnc::call(double z)
    {
        if (z > this->D)
        {
            return 0.0;
        }
        return cos(z * dealii::numbers::PI / (2 * this->D));
    }

    TwoLayerAnalytic::TwoLayerAnalytic(double fault_slip,
                             double fault_depth,
                             double shear_modulus,
                             double viscosity,
                             SlipFnc &slip_fnc)
    {
        this->fault_slip = fault_slip;
        this->fault_depth = fault_depth;
        this->shear_modulus = shear_modulus;
        this->viscosity = viscosity;
        this->images = 20;
        this->slip_fnc = &slip_fnc;
        this->integration = gsl_integration_workspace_alloc(1000); 
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
    double TwoLayerAnalytic::simple_velocity(double x, double y, double t)
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
                term = (1.0 / (2.0 * dealii::numbers::PI)) * (term1 + term2);
            }
            else
            {
                // Shallower than the fault bottom
                term1 = atan((2 * m + 1 + y_scaled) / x_scaled);
                term2 = -atan((2 * m - 1 + y_scaled) / x_scaled);
                term3 = atan((2 * m + 1 - y_scaled) / x_scaled);
                term4 = -atan((2 * m - 1 - y_scaled) / x_scaled);
                term = (1.0 / (2.0 * dealii::numbers::PI)) * (term1 + term2 + term3 + term4);
            }
            v += factor * term;
        }
        v *= exp(-t / t_r) * (1.0 / t_r);
        assert(!isnan(v));
        return v;
    }

    double TwoLayerAnalytic::simple_Szx(double x, double y) const
    {
        double factor, main_term, image_term, Szx, Szy;
        factor = (this->fault_slip * this->shear_modulus) / (2 * dealii::numbers::PI);
        main_term = (y - this->fault_depth) / 
            (pow((y - this->fault_depth), 2) + pow(x, 2));
        image_term = -(y + this->fault_depth) / 
            (pow((y + this->fault_depth), 2) + pow(x, 2));
        Szx = factor * (main_term + image_term);
        return Szx;
    }

    double TwoLayerAnalytic::simple_Szy(double x, double y) const
    {
        double factor, main_term, image_term, Szx, Szy;
        factor = (this->fault_slip * this->shear_modulus) / (2 * dealii::numbers::PI);

        main_term = -x / (pow(x, 2) + pow((y - this->fault_depth), 2));
        image_term = x / (pow(x, 2) + pow((y + this->fault_depth), 2));
        Szy = factor * (main_term + image_term);
        return Szy;
    }

    struct AnalyticFncParameters {
        double x;
        double y;
        double D;
        int m;
        SlipFnc* s;
    };

    double term_1_fnc_low(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(z + (2 * p->m) * p->D + p->y,2 ) + pow(p->x, 2));
    }

    double term_2_fnc_low(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(-z + (2 * p->m) * p->D + p->y, 2) + pow(p->x, 2));
    }

    double term_3_fnc_low(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(z + (2 * p->m - 2) * p->D + p->y, 2) + pow(p->x, 2));
    }

    double term_4_fnc_low(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(-z + (2 * p->m - 2) * p->D + p->y, 2) + pow(p->x, 2));
    }

    double term_1_fnc_up(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(z + (2 * p->m) * p->D + p->y, 2) + pow(p->x, 2));
    }

    double term_2_fnc_up(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(-z + (2 * p->m) * p->D + p->y, 2) + pow(p->x, 2));
    }

    double term_3_fnc_up(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(z + (2 * p->m) * p->D - p->y, 2) + pow(p->x, 2));
    }

    double term_4_fnc_up(double z, void * params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * p->x / (pow(-z + (2 * p->m) * p->D - p->y, 2) + pow(p->x, 2));
    }

    /*
     * TODO: Known bug caused by integral_velocity(10000.0, 100000.0, 1.0)
     * Probably an overflow
     */
    double TwoLayerAnalytic::integral_velocity(double x, double y, double t) 
    {
        double v = 0.0;
        double t_r = (2 * viscosity) / shear_modulus;
        double factor, term1, term2, term3, term4;

        gsl_function F;
        AnalyticFncParameters params;
        params.x = x;
        params.y = y;
        params.D = this->fault_depth;
        params.s = slip_fnc;
        F.params = &params;
        double int_result;
        double int_error;

        for (int m = 1; m < images; m++)
        {
            params.m = m;
            factor = pow(t / t_r, m - 1) / factorial(m - 1);
            if (y > this->fault_depth) 
            {
                F.function = &term_1_fnc_low;
                gsl_integration_qags (&F, 0, this->fault_depth,
                                      0, 1e-7, 1000, 
                                      this->integration, 
                                      &int_result, 
                                      &int_error);
                term1 = int_result + 
                    slip_fnc->call(0) * atan(((2 * m) * this->fault_depth + y) / x);

                F.function = &term_2_fnc_low;
                gsl_integration_qags (&F, 0, this->fault_depth,
                                      0, 1e-7, 1000, 
                                      this->integration, 
                                      &int_result, 
                                      &int_error);
                term2 = int_result - 
                    slip_fnc->call(0) * atan(((2 * m) * this->fault_depth + y) / x);

                F.function = &term_3_fnc_low;
                gsl_integration_qags (&F, 0, this->fault_depth,
                                      0, 1e-7, 1000, 
                                      this->integration, 
                                      &int_result, 
                                      &int_error);
                term3 = int_result + 
                    slip_fnc->call(0) * 
                    atan(((2 * m - 2) * this->fault_depth + y) / x);

                F.function = &term_4_fnc_low;
                gsl_integration_qags (&F, 0, this->fault_depth,
                                      0, 1e-7, 1000, 
                                      this->integration, 
                                      &int_result, 
                                      &int_error);
                term4 = int_result - 
                    slip_fnc->call(0) * 
                    atan(((2 * m - 2) * this->fault_depth + y) / x);

                v += factor * (term1 + term2 + term3 + term4);
            }
            else
            {
                F.function = &term_1_fnc_up;
                gsl_integration_qags (&F, 0, this->fault_depth,
                        0, 1e-7, 1000, this->integration, &int_result, &int_error);
                term1 = int_result + slip_fnc->call(0) * atan(((2 * m) * this->fault_depth + y) / x);

                F.function = &term_2_fnc_up;
                gsl_integration_qags (&F, 0, this->fault_depth,
                        0, 1e-7, 1000, this->integration, &int_result, &int_error);
                term2 = int_result - slip_fnc->call(0) * atan(((2 * m) * this->fault_depth + y) / x);

                F.function = &term_3_fnc_up;
                gsl_integration_qags (&F, 0, this->fault_depth,
                        0, 1e-7, 1000, this->integration, &int_result, &int_error);
                term3 = int_result + slip_fnc->call(0) * atan(((2 * m) * this->fault_depth - y) / x);

                F.function = &term_4_fnc_up;
                gsl_integration_qags (&F, 0, this->fault_depth,
                        0, 1e-7, 1000, this->integration, &int_result, &int_error);
                term4 = int_result - slip_fnc->call(0) * atan(((2 * m) * this->fault_depth - y) / x);

                v += factor * (term1 + term2 + term3 + term4);
            }
        }
        v *= (1.0 / (2.0 * dealii::numbers::PI)) * exp(-t / t_r)  / t_r;
        return v;
    }


    double Szx_main_term_fnc(double z, void* params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * (pow(p->y - z, 2) - pow(p->x, 2)) / 
            pow((pow(p->y - z, 2) + pow(p->x, 2)), 2);
    }

    double Szx_image_term_fnc(double z, void* params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * (pow(p->y + z, 2) - pow(p->x, 2)) / 
            pow((pow(p->y + z, 2) + pow(p->x, 2)), 2);
    }

    double Szy_main_term_fnc(double z, void* params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * (-2 * p->x * (p->y - z)) / 
            pow((pow(p->y - z, 2) + pow(p->x, 2)), 2);
    }

    double Szy_image_term_fnc(double z, void* params)
    {
        AnalyticFncParameters* p = static_cast<AnalyticFncParameters*>(params);
        return p->s->call(z) * (-2 * p->x * (p->y + z)) /
            pow((pow(p->y + z, 2) + pow(p->x, 2)), 2);
    }

    double TwoLayerAnalytic::integral_Szx(double x, double y) const
    {
        double factor, main_term, image_term, Szx, Szy;

        gsl_function F;
        AnalyticFncParameters params;
        params.x = x;
        params.y = y;
        params.D = this->fault_depth;
        params.s = slip_fnc;
        F.params = &params;
        double int_error;

        factor = (shear_modulus) / (2 * dealii::numbers::PI);

        F.function = &Szx_main_term_fnc;
        gsl_integration_qags (&F, 0, this->fault_depth,
                0, 1e-7, 1000, this->integration, &main_term, &int_error);
        F.function = &Szx_image_term_fnc;
        gsl_integration_qags (&F, 0, this->fault_depth,
                0, 1e-7, 1000, this->integration, &image_term, &int_error);
        Szx = factor * (main_term + image_term);
        return Szx;
    }

    double TwoLayerAnalytic::integral_Szy(double x, double y) const
    {
        double factor, main_term, image_term, Szx, Szy;

        gsl_function F;
        AnalyticFncParameters params;
        params.x = x;
        params.y = y;
        params.D = this->fault_depth;
        params.s = slip_fnc;
        F.params = &params;
        double int_error;

        factor = (shear_modulus) / (2 * dealii::numbers::PI);

        F.function = &Szy_main_term_fnc;
        gsl_integration_qags (&F, 0, this->fault_depth,
                0, 1e-7, 1000, this->integration, &main_term, &int_error);
        F.function = &Szy_image_term_fnc;
        gsl_integration_qags (&F, 0, this->fault_depth,
                0, 1e-7, 1000, this->integration, &image_term, &int_error);
        Szy = factor * (main_term + image_term);
        return Szy;
    }
}
