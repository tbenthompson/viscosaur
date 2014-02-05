#ifndef __viscosaur_inv_visc_h
#define __viscosaur_inv_visc_h

#include "problem_data.h"
#include <deal.II/base/vectorization.h>
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>

namespace viscosaur
{
    namespace bp = boost::python;

    template <int dim>
    class InvViscosity
    {
        public:
            virtual dealii::VectorizedArray<double>
                value(const dealii::Point<
                        dim, dealii::VectorizedArray<double> > &p,
                      const dealii::Tensor<1, dim, 
                        dealii::VectorizedArray<double> > &strs) const
            {
                dealii::VectorizedArray<double> retval; 
                for(int i = 0; i < p[0].n_array_elements; i++) 
                {
                    dealii::Point<dim, double> newp;
                    dealii::Tensor<1, dim> indv_strs;
                    for(int d = 0; d < dim; d++) 
                    {
                        newp[d] = p[d][i];
                        indv_strs[d] = strs[d][i];
                    }
                    retval.data[i] = value(newp, indv_strs);
                }
                return retval;
            }

            virtual dealii::VectorizedArray<double>
                strs_deriv(const dealii::Point<
                        dim, dealii::VectorizedArray<double> > &p,
                      const dealii::Tensor<1, dim, 
                        dealii::VectorizedArray<double> > &strs) const
            {
                dealii::VectorizedArray<double> retval; 
                for(int i = 0; i < p[0].n_array_elements; i++) 
                {
                    dealii::Point<dim, double> newp;
                    dealii::Tensor<1, dim> indv_strs;
                    for(int d = 0; d < dim; d++) 
                    {
                        newp[d] = p[d][i];
                        indv_strs[d] = strs[d][i];
                    }
                    retval.data[i] = strs_deriv(newp, indv_strs);
                }
                return retval;
            }

            virtual double value(const dealii::Point<dim>  &p,
                                 const dealii::Tensor<1, dim> strs) const = 0;

            /* Derivative of the inverse viscosity function with respect to the
             * stress. Useful for any sort of iterative solver for the ode.
             */
            virtual double strs_deriv(const dealii::Point<dim>  &p,
                                 const dealii::Tensor<1, dim> strs) const = 0;
    };

    template <int dim>
    class InvViscosityTLA: public InvViscosity<dim>
    {
        public:
            InvViscosityTLA(bp::dict &params)
            {
                layer_depth = 
                    bp::extract<double>(params["fault_depth"]);
                inv_viscosity = 1.0 /
                    bp::extract<double>(params["viscosity"]);
            }

            virtual double value(const dealii::Point<dim>  &p,
                                 const dealii::Tensor<1, dim> strs) const
            {
                if (p(1) < layer_depth)
                {
                    return 0;
                }
                return inv_viscosity;
            }

            /* Derivative of the inverse viscosity function with respect to the
             * stress.
             */
            virtual double strs_deriv(const dealii::Point<dim>  &p,
                                 const dealii::Tensor<1, dim> strs) const
            {
                return 0;
            }

            double layer_depth;
            double inv_viscosity;
    };
}
#endif
