#ifndef __viscosaur_nonlin_h
#define __viscosaur_nonlin_h
#include "inv_visc.h"
#include "boost/python/dict.hpp"
#include "boost/python/extract.hpp"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace viscosaur
{
    namespace powerlaw
    {
        namespace bp = boost::python;

        template <int dim>
        class InvViscosityPowerLaw: public InvViscosity<dim>
        {
            public:
                InvViscosityPowerLaw(bp::dict &params)
                {
                    A = bp::extract<double>(params["power_law_A"]);
                    Q = bp::extract<double>(params["power_law_Q"]);
                    n = bp::extract<double>(params["power_law_n"]);
                }

                double get_T(const dealii::Point<dim> &p) const
                {
                    double y = p[1];
                    return 300 + (y / 1000.0) * 20;
                }

                virtual double value(const dealii::Point<dim>  &p,
                                     const dealii::Tensor<1, dim> strs) const
                {
                    double sum;
                    for(int i = 0;i < dim;i++)
                    {
                        sum = strs[i] * strs[i];
                    }
                    double T = get_T(p);
                    double retval = A * pow(sum * 3, (n - 1) / 2.0) * 
                        exp(-Q / (R * T));
                    return retval;
                }

                /* Derivative of the inverse viscosity function with respect to the
                 * stress.
                 */
                virtual double strs_deriv(const dealii::Point<dim>  &p,
                                     const dealii::Tensor<1, dim> strs,
                                     const unsigned int comp) const
                {
                    double sum;
                    for(int i = 0;i < dim;i++)
                    {
                        sum = strs[i] * strs[i];
                    }
                    double T = get_T(p);
                    double retval = A * pow(3, (n - 1) / 2.0) * ((n - 1) / 2.0) * 
                            pow(sum, (n - 3) / 2.0) * (2 * strs[comp]) * 
                            exp(-Q / (R * T));
                    return retval;
                }

                double A;
                double Q;
                double n;
                const static double R = 8.314;
        };
    }
}
#endif
