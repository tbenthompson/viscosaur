#ifndef __viscosaur_matrix_free_calculation_h
#define __viscosaur_matrix_free_calculation_h

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <boost/any.hpp>
namespace viscosaur
{
    //TODO: With some thought, the dependency on problem data could be 
    //refactored out here.
    template <int dim> class ProblemData; 
    template <int dim> class OpFactory;

    template <int dim>
    class MatrixFreeCalculation
    {
        public:
            MatrixFreeCalculation(ProblemData<dim> &p_pd,
                                  dealii::MatrixFree<dim> &p_mf,
                                  dealii::ConstraintMatrix &p_cm,
                                  bool scalar = false);
            ~MatrixFreeCalculation();

            /* Pre-build some of the structures necessary for efficient 
             * updating of the stress.
             */
            virtual void reinit(ProblemData<dim> &p_pd,
                        dealii::MatrixFree<dim> &p_mf,
                        dealii::ConstraintMatrix &p_cm,
                        bool scalar = false);

            /* Compute and invert the diagonal mass matrix produced by the 
             * GLL quadrature and interpolation.
             */
            void compute_mass_matrix(bool scalar);

            /* The main function of the class computes one time step. Call the
             * local_apply function for every cell. Then, uses the inverse mass
             * matrix to compute the solution. Because of the Gauss Lobatto 
             * lagrange interpolation, the mass matrix is diagonal and can
             * be easily inverted.
             */
            void apply(dealii::parallel::distributed::Vector<double> &dst, 
                std::vector<dealii::parallel::distributed::Vector<double>* > 
                    &sources,
                boost::any data);

            void apply(dealii::parallel::distributed::Vector<double> &dst, 
                boost::any data);

            void apply_function(dealii::parallel::distributed::Vector<double> &dst, 
                dealii::Function<dim> &data);

            /* The partner in crime of the "apply" function above. This computes
             * one time step for one cell. What a messy declaration!
             */
            void local_apply(const dealii::MatrixFree<dim> &data,
                dealii::parallel::distributed::Vector<double> &dst,
                const std::vector<
                    dealii::parallel::distributed::Vector<double>* > &src,
                const std::pair<unsigned int, unsigned int> &cell_range);

            ProblemData<dim>* pd;

            dealii::MatrixFree<dim>* mf;

            dealii::ConstraintMatrix* constraints;

            // The reciprocal of the diagonal of the mass matrix.
            dealii::parallel::distributed::Vector<double> inv_mass_matrix;
            
            //Set this pointer to the factory that will create the operators
            //that this class calls to produce its output.
            OpFactory<dim>* op_factory;

            boost::any data;
        protected:
            MatrixFreeCalculation() {};
    };
}

#endif
