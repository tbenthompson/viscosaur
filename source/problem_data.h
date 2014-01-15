#ifndef __viscosaur_problem_data_h
#define __viscosaur_problem_data_h

/* Forward declarations of necessary classes. Maybe move all this out to one 
 * forward declaration file.h
 */ 
namespace boost
{
    namespace python
    {
        class dict;
    }
}
namespace dealii
{
    namespace LinearAlgebraPETSc
    {
        namespace MPI
        {
            class Vector;
        }
    }
    namespace parallel
    {
        namespace distributed
        {
            template <int dim> 
                class Triangulation;
        }
    }
    template <int dim> class FE_Q;
    template <int dim> class QGaussLobatto;
    class ConditionalOStream;
    class TimerOutput;
    class MPI_Comm;
}

namespace viscosaur
{
    template <int dim>
    class ProblemData
    {
        public:
            ProblemData(bp::dict &params);
            void init_mesh();
            void init_dofs();
            void refine_grid(LA::MPI::Vector &local_solution);
            CompressedSimpleSparsityPattern* create_sparsity_pattern(
                    ConstraintMatrix &constraints);
            ConstraintMatrix* create_constraints();

            bp::dict              parameters;
            MPI_Comm              mpi_comm;
            parallel::distributed::Triangulation<dim> triangulation;
            DoFHandler<dim>       dof_handler;
            FE_Q<dim>             fe;
            QGaussLobatto<dim>    quadrature;
            IndexSet              locally_owned_dofs;
            IndexSet              locally_relevant_dofs;
            ConditionalOStream    pcout;
            TimerOutput           computing_timer;
    };
}
#endif
