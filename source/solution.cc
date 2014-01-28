#include "solution.h"
#include "problem_data.h"

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/parallel_vector.h>

namespace viscosaur
{
    using namespace dealii;

    template <int dim>
    Solution<dim>::Solution(ProblemData<dim> &p_pd)
    {
        pd = &p_pd;

    }

    template <int dim>
    void
    Solution<dim>::
    apply_init_cond(Function<dim> &init_szx,
                    Function<dim> &init_szy)
    {
        TimerOutput::Scope t(pd->computing_timer, "init_cond");
        VectorTools::interpolate(pd->dof_handler, init_szx, cur_szx);
        VectorTools::interpolate(pd->dof_handler, init_szy, cur_szy);
    }

    template <int dim>
    void
    Solution<dim>::
    start_timestep()
    {
        //Flip the solns to retain the old soln.
        old_szx.swap(cur_szx);
        old_szy.swap(cur_szy);
        old_vel.swap(cur_vel);
        old_vel_for_strs.swap(cur_vel_for_strs);
    }

    template <int dim>
    void Solution<dim>::output(std::string data_dir,
                               std::string filename,
                               Function<dim> &exact) const
    {
        TimerOutput::Scope t(pd->computing_timer, "output");
        parallel::distributed::Vector<double> exact_vel;
        exact_vel.reinit(pd->locally_owned_dofs, pd->locally_relevant_dofs,
                pd->mpi_comm);
        VectorTools::interpolate(pd->dof_handler, exact, exact_vel);
        exact_vel.compress(VectorOperation::add);
        exact_vel.update_ghost_values();

        Vector<double> local_errors(pd->triangulation.n_active_cells());
        VectorTools::integrate_difference(
                pd->dof_handler,
                cur_vel,
                exact,
                local_errors,
                QGauss<dim>(3),
                VectorTools::L2_norm);

        const double total_local_error = local_errors.l2_norm();
        const double total_global_error = std::sqrt (
                dealii::Utilities::MPI::sum (
                    total_local_error * 
                    total_local_error, pd->mpi_comm));
        pd->pcout << "Total exact error: " << total_global_error << std::endl;

        DataOut<dim> data_out;
        data_out.attach_dof_handler(pd->dof_handler);
        data_out.add_data_vector(cur_vel, "vel");
        data_out.add_data_vector(tent_szx, "tentative_szx");
        data_out.add_data_vector(tent_szy, "tentative_szy");
        // data_out.add_data_vector(cur_szx, "szx");
        // data_out.add_data_vector(cur_szy, "szy");
        data_out.add_data_vector(exact_vel, "exact_vel");
        data_out.add_data_vector(local_errors, "error");

        Vector<float> subdomain(pd->triangulation.n_active_cells());
        unsigned int this_subd = pd->triangulation.locally_owned_subdomain();
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = this_subd;
        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::ofstream output((data_dir + "/" + filename + 
                    Utilities::int_to_string(this_subd, 4) + ".vtu").c_str());
        data_out.write_vtu(output);

        if (Utilities::MPI::this_mpi_process(pd->mpi_comm) == 0)
        {
            // Build the list of filenames to store in the master pvtu file
            std::vector<std::string> files;
            for (unsigned int i=0;
                    i<Utilities::MPI::n_mpi_processes(pd->mpi_comm);
                    ++i)
            {
                files.push_back(filename + 
                    Utilities::int_to_string(i, 4) + ".vtu");
            }

            std::ofstream master_output((data_dir + "/" + filename + 
                    Utilities::int_to_string(this_subd, 4) + ".pvtu").c_str());
            // Write the master pvtu record. Load this pvtu file if you want
            // to view all the data at once. Use a tool like visit or paraview.
            data_out.write_pvtu_record(master_output, files);
        }
    }

    template <int dim>
    parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >* 
    Solution<dim>::
    start_refine()
    {
        parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >*
            sol_trans = new parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >(pd->dof_handler);    

        std::vector<const parallel::distributed::Vector<double>* > vecs(3);
        vecs[0] = &cur_vel;
        vecs[1] = &cur_szx;
        vecs[2] = &cur_szy;
        sol_trans->prepare_for_coarsening_and_refinement(vecs);

        return sol_trans;
    }
    

    template <int dim>
    void
    Solution<dim>::
    post_refine(parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >* sol_trans)
    {
        std::vector<parallel::distributed::Vector<double>* > vecs(3);
        vecs[0] = &cur_vel;
        vecs[1] = &cur_szx;
        vecs[2] = &cur_szy;
        sol_trans->interpolate(vecs);
    }

    template class Solution<2>;
    template class Solution<3>;
}
