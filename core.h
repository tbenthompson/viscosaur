#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// #define USE_PETSC_LA

namespace LA
{
#ifdef USE_PETSC_LA
    using namespace dealii::LinearAlgebraPETSc;
#else
    using namespace dealii::LinearAlgebraTrilinos;
#endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>

#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>

#include "elastic_stress.h"

namespace viscosaur
{
    using namespace dealii;


    template <int dim>
    class Controller
    {
        public:
            Controller ();
            ~Controller ();

            void run ();

        private:
            void setup_system ();
            void assemble_system ();
            void solve ();
            void refine_grid ();
            void output_results (const unsigned int cycle) const;
            void init_mesh ();

            MPI_Comm              mpi_communicator;
            parallel::distributed::Triangulation<dim>
                                  triangulation;
            DoFHandler<dim>       dof_handler;
            FE_Q<dim>             fe;
            IndexSet              locally_owned_dofs;
            IndexSet              locally_relevant_dofs;
            ConstraintMatrix      constraints;
            LA::MPI::SparseMatrix system_matrix;
            LA::MPI::Vector       locally_relevant_solution;
            LA::MPI::Vector       system_rhs;
            ConditionalOStream    pcout;
            TimerOutput           computing_timer;
    };
}
