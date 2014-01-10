
namespace viscosaur
{
    using namespace dealii;
    class Stress
    {
        private:
            parallel::distributed::Triangulation<dim> triangulation;
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
    }
}
