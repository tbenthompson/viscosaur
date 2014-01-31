#ifndef __viscosaur_velocity_h
#define __viscosaur_velocity_h

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
    template <int dim> class Scheme;
    template <int dim> class BoundaryCond;

    /*
     * The Velocity Solver. Most of this code is extracted from tutorial 40
     * on the deal.ii website. Currently located at
     * http://www.dealii.org/8.1.0/doxygen/deal.II/step_40.html 
     *
     * Add some documentation...
     */
    template <int dim>
    class Velocity
    {
        //TODO: EVERY call accepts a Scheme. Make this a member pointer.
        public:
            Velocity(Solution<dim> &soln,
                    BoundaryCond<dim> &bc,
                    ProblemData<dim> &p_pd, 
                    Scheme<dim> &sch);

            void step(Solution<dim> &soln, Scheme<dim> &sch);

            void update_bc(BoundaryCond<dim> &bc, Scheme<dim> &sch);
        private:
            void setup_system(BoundaryCond<dim> &bc,
                              Solution<dim> &soln, Scheme<dim> &sch);

            /* Build the relevant matrices. */
            void assemble_matrix(Solution<dim> &soln, Scheme<dim> &sch);

            /* Build the rhs */
            void assemble_rhs(Solution<dim> &soln, Scheme<dim> &sch);

            void solve (Solution<dim> &soln, Scheme<dim> &sch);
            
            ProblemData<dim>* pd;
            dealii::Function<dim>* init_cond_Szx;
            dealii::Function<dim>* init_cond_Szy;
            dealii::ConstraintMatrix constraints;
            LA::MPI::SparseMatrix system_matrix;
            LA::MPI::Vector       locally_relevant_solution;
            LA::MPI::Vector       system_rhs;
    };

}
#endif
