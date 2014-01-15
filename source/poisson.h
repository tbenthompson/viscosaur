#ifndef __viscosaur_poisson_h
#define __viscosaur_poisson_h

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
}

namespace viscosaur
{
    /* Forward declare some of the classes needed.
     */
    template <int dim> class ProblemData;
    template <int dim> class PoissonRHS;


    /* Using higher order polynomials results in a problem with very slow 
     * assembly. I suspect the fe_values.shape_grad call is not caching 
     * the values properly
     */

    /*
     * The Poisson Solver. Most of this code is extracted from tutorial 40
     * on the deal.ii website. Currently located at
     * http://www.dealii.org/8.1.0/doxygen/deal.II/step_40.html 
     *
     * Add some documentation...
     *
     * Note that this entire class is defined in the header. This is required
     * for a templated class. C++11 may have "fixed" this. Check?
     */
    template <int dim>
    class Poisson
    {
        public:
            Poisson(ProblemData<dim> &p_pd);
            ~Poisson ();

            LA::MPI::Vector run (PoissonRHS<dim>* rhs);
            dealii::DoFHandler<dim, dim>* get_dof_handler();

        private:
            void setup_system ();

            /* Assembly functions.
             */
            void fill_cell_matrix(
                    dealii::FullMatrix<double> &cell_matrix,
                    dealii::FEValues<dim, dim> &fe_values,
                    const unsigned int n_q_points,
                    const unsigned int dofs_per_cell);

            void assemble_system (PoissonRHS<dim>* rhs);

            void solve ();

            std::string output_filename(const unsigned int cycle,
                                        const unsigned int subdomain) const;
            void output_results (const unsigned int cycle) const;
            void init_mesh ();
            
            ProblemData<dim>* pd;
            dealii::ConstraintMatrix constraints;
            LA::MPI::SparseMatrix system_matrix;
            LA::MPI::Vector       locally_relevant_solution;
            LA::MPI::Vector       system_rhs;
    };

}
#endif
