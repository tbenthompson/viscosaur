#ifndef __viscosaur_poisson_h
#define __viscosaur_poisson_h

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include "linear_algebra.h"

namespace dealii
{
    //Unfortunately, dealii declares the template defaults in the declaration
    //of these classes, so we lose the capability to use their default spacedim
    //if we forward declare them.
    template <int dim, int spacedim> class DoFHandler;
    template <int dim, int spacedim> class FEValues;
    template <int dim> class Function;
}

namespace viscosaur
{
    /* Forward declare some of the classes needed.
     */
    template <int dim> class ProblemData;
    template <int dim> class Velocity;
    template <int dim> class Solution;


    /* Using higher order polynomials results in a problem with very slow 
     * assembly. I suspect the fe_values.shape_grad call is not caching 
     * the values properly
     */

    /*
     * The Velocity Solver. Most of this code is extracted from tutorial 40
     * on the deal.ii website. Currently located at
     * http://www.dealii.org/8.1.0/doxygen/deal.II/step_40.html 
     *
     * Add some documentation...
     *
     * Note that this entire class is defined in the header. This is required
     * for a templated class. C++11 may have "fixed" this. Check?
     */
    template <int dim>
    class InvViscosity: public dealii::Function<dim>
    {
        public:
            InvViscosity(ProblemData<dim> &p_pd);

            virtual dealii::VectorizedArray<double>
                value(const dealii::Point<
                        dim, dealii::VectorizedArray<double> > &p,
                      const unsigned int component) const
            {
                dealii::VectorizedArray<double> retval; 
                for(int i = 0; i < p[0].n_array_elements; i++) 
                {
                    dealii::Point<dim, double> newp(p[0][i], p[1][i]);
                    retval.data[i] = value(newp, component);
                }
                return retval;
            }

            virtual double value(const dealii::Point<dim>  &p,
                         const unsigned int component) const
            {
                if (p(1) < layer_depth)
                {
                    return 0;
                }
                return inv_viscosity;
            }

            double layer_depth;
            double inv_viscosity;
    };

    template <int dim>
    class Velocity
    {
        public:
            Velocity(Solution<dim> &soln,
                    dealii::Function<dim> &bc,
                    ProblemData<dim> &p_pd);

            void step(Solution<dim> &soln);

            void update_bc(dealii::Function<dim> &bc);
        private:
            void setup_system(dealii::Function<dim> &bc,
                              Solution<dim> &soln);

            /* Build the relevant matrices. */
            void assemble_matrix(Solution<dim> &soln);

            /* Build the rhs */
            void assemble_rhs(Solution<dim> &soln);

            void solve (Solution<dim> &soln);
            
            ProblemData<dim>* pd;
            dealii::Function<dim>* init_cond_Szx;
            dealii::Function<dim>* init_cond_Szy;
            dealii::ConstraintMatrix constraints;
            dealii::ConstraintMatrix hanging_node_constraints;
            LA::MPI::SparseMatrix system_matrix;
            LA::MPI::Vector       locally_relevant_solution;
            LA::MPI::Vector       system_rhs;
    };

}
#endif
