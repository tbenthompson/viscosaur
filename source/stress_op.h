#ifndef __viscosaur_stress_op_h
#define __viscosaur_stress_op_h
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace dealii
{
    namespace parallel
    {
        namespace distributed
        {
            template <typename T> class Vector;
        }
    }
}
namespace viscosaur
{
    template <int dim> class ProblemData;
    template <int dim> class InvViscosity;

    template <int dim, int fe_degree>
    class StressOp
    {
        public:
            StressOp(const dealii::MatrixFree<dim,double> &data_in, 
                     const double p_time_step,
                     ProblemData<dim> &p_pd,
                     InvViscosity<dim> &p_inv_visc); 

            /* The main function of the class computes one time step. Call the
             * local_apply function for every cell. Then, uses the inverse mass
             * matrix to compute the solution. Because of the Gauss Lobatto 
             * lagrange interpolation, the mass matrix is diagonal and can
             * be easily inverted.
             */
            void apply(dealii::parallel::distributed::Vector<double> &dst, 
                const dealii::parallel::distributed::Vector<double> &src) const;

        private:
            const dealii::MatrixFree<dim,double> &data;
            dealii::parallel::distributed::Vector<double> inv_mass_matrix;
            ProblemData<dim>* pd;

            double time_step;
            double shear_modulus;
            InvViscosity<dim>* inv_visc;

            /* The partner in crime of the "apply" function above. This computes
             * one time step for one cell. 
             */
            void local_apply(const dealii::MatrixFree<dim,double> &data,
                             dealii::parallel::distributed::Vector<double> &dst,
                             const dealii::parallel::distributed::Vector
                                    <double> &src,
                             const std::pair<unsigned int,
                                             unsigned int> &cell_range) const;
    };
}
#endif
