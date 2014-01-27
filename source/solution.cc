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
    std::string Solution<dim>::output_filename(const unsigned int cycle,
            const unsigned int subdomain) const
    {
        std::string filename = "solution-" +
                Utilities::int_to_string(cycle, 2) +
                "." +
                Utilities::int_to_string(subdomain, 4);
        return filename;
    }


    template <int dim>
    void Solution<dim>::output(const unsigned int cycle,
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

        std::string this_f = output_filename(cycle, this_subd);
        std::ofstream output(("data/" + this_f + ".vtu").c_str());
        data_out.write_vtu(output);

        if (Utilities::MPI::this_mpi_process(pd->mpi_comm) == 0)
        {
            // Build the list of filenames to store in the master pvtu file
            std::vector<std::string> filenames;
            for (unsigned int i=0;
                    i<Utilities::MPI::n_mpi_processes(pd->mpi_comm);
                    ++i)
            {
                std::string f = output_filename(cycle, i);
                filenames.push_back (f + ".vtu");
            }

            std::ofstream master_output(("data/" + this_f + ".pvtu").c_str());
            // Write the master pvtu record. Load this pvtu file if you want
            // to view all the data at once. Use a tool like visit or paraview.
            data_out.write_pvtu_record(master_output, filenames);
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

        // std::vector<const parallel::distributed::Vector<double>* > vecs(3);
        // vecs.push_back(&cur_vel);
        // vecs.push_back(&cur_szx);
        // vecs.push_back(&cur_szy);
        sol_trans->prepare_for_coarsening_and_refinement(cur_vel);

        return sol_trans;
    }
    

    template <int dim>
    void
    Solution<dim>::
    post_refine(parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >* sol_trans)
    {
        // std::vector<parallel::distributed::Vector<double>* > vecs(3);
        // vecs.push_back(&cur_vel);
        // vecs.push_back(&cur_szx);
        // vecs.push_back(&cur_szy);
        // sol_trans->interpolate(vecs);
        parallel::distributed::Vector<double> vec;
        vec.reinit(pd->locally_owned_dofs, pd->mpi_comm);
        sol_trans->interpolate(vec);
        cur_vel = vec;
    }

    template class Solution<2>;
    template class Solution<3>;
}
