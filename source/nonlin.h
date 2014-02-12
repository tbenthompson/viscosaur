#ifndef __viscosaur_nonlin_h
#define __viscosaur_nonlin_h
namespace viscosaur
{
    namespace powerlaw
    {
        template <int dim>
        class InvViscosityPowerLaw: public InvViscosity<dim>
        {
            public:
                InvViscosityPowerLaw(bp::dict &params)
                {
                    A = bp::extract<double>(params["power_law_A"]);
                    n = bp::extract<double>(params["power_law_n"]);
                }

                virtual double value(const dealii::Point<dim>  &p,
                                     const dealii::Tensor<1, dim> strs) const
                {
                    double sum;
                    for(int i = 0;i < dim;i++)
                    {
                        sum = strs[i] * strs[i];
                    }
                    double retval = A * pow(sum * 3, (n - 1) / 2.0);
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
                    double retval = A * pow(3, (n - 1) / 2.0) * ((n - 1) / 2.0) * 
                            pow(sum, (n - 3) / 2.0) * (2 * strs[comp]);
                    return retval;
                }

                double A;
                double n;
        };
    }
}
#endif
