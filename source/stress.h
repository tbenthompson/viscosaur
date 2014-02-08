#ifndef __viscosaur_stress_h
#define __viscosaur_stress_h

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <boost/shared_ptr.hpp>


namespace viscosaur
{
    template <int dim> class ProblemData; 
    template <int dim> class Solution; 
    template <int dim> class Scheme;
    template <int dim> class OpFactory;

    template <int dim>
    class Stress
    {
        public:
            Stress(ProblemData<dim> &p_pd);
            ~Stress();

            /* Pre-build some of the structures necessary for efficient 
             * updating of the stress.
             */
            void reinit(ProblemData<dim> &p_pd);

            /* Compute and invert the diagonal mass matrix produced by the 
             * GLL quadrature and interpolation.
             */
            void compute_mass_matrix();

            /* The main function of the class computes one time step. Call the
             * local_apply function for every cell. Then, uses the inverse mass
             * matrix to compute the solution. Because of the Gauss Lobatto 
             * lagrange interpolation, the mass matrix is diagonal and can
             * be easily inverted.
             */
            void apply(dealii::parallel::distributed::Vector<double> &dst, 
                const dealii::parallel::distributed::Vector<double> &src,
                Solution<dim> &soln,
                const double time_step);

            /* The partner in crime of the "apply" function above. This computes
             * one time step for one cell. What a messy declaration!
             */
            void local_apply(const dealii::MatrixFree<dim> &data,
                dealii::parallel::distributed::Vector<double> &dst,
                const std::vector<
                    dealii::parallel::distributed::Vector <double> > &src,
                const std::pair<unsigned int, unsigned int> &cell_range);

            void tentative_step(Solution<dim> &soln, Scheme<dim> &scheme,
                    double time_step);
            void correction_step(Solution<dim> &soln, Scheme<dim> &scheme,
                    double time_step);
        private:
            ProblemData<dim>* pd;

            dealii::ConstraintMatrix     constraints;

            dealii::parallel::distributed::Vector<double> inv_mass_matrix;
            
            OpFactory<dim>* op_factory;

            double time_step;
    };
}
#endif
