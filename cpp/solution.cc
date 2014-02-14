#include "solution.h"
#include "problem_data.h"
#include "op_factory.h"
#include "matrix_free_calculation.h"

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/parallel_vector.h>
#include <boost/python/extract.hpp>


namespace viscosaur
{
    namespace bp = boost::python;
    using namespace dealii;

    template <int dim>
    Solution<dim>::Solution(ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        reinit();
    }


    template<int dim>
    void
    Solution<dim>::
    reinit()
    {
        poisson_soln.reinit(pd->vel_locally_owned_dofs,
            pd->vel_locally_relevant_dofs, pd->mpi_comm);

        pd->strs_matrix_free.initialize_dof_vector(cur_strs);
        pd->strs_matrix_free.initialize_dof_vector(old_strs);
        pd->strs_matrix_free.initialize_dof_vector(old_old_strs);
        pd->strs_matrix_free.initialize_dof_vector(tent_strs);
        pd->vel_matrix_free.initialize_dof_vector(cur_vel);
        pd->vel_matrix_free.initialize_dof_vector(old_vel);
    }

    template <int dim>
    void
    Solution<dim>::
    apply_init_cond(Function<dim> &init_strs,
                    Function<dim> &init_vel)
    {
        TimerOutput::Scope t(pd->computing_timer, "init_cond");

        MatrixFreeCalculation<dim> mfc(*pd, pd->strs_matrix_free, 
                pd->strs_hanging_node_constraints);
        StrsProjectionOpFactory<dim> strs_op_factory;
        mfc.op_factory = &strs_op_factory;
        mfc.apply(cur_strs, &init_strs);

        MatrixFreeCalculation<dim> mfc2(*pd, pd->vel_matrix_free, 
                pd->vel_hanging_node_constraints, true);
        VelProjectionOpFactory<dim> vel_op_factory;
        mfc2.op_factory = &vel_op_factory;
        mfc2.apply(cur_vel, &init_vel);
    }

    template <int dim>
    void
    Solution<dim>::
    init_multistep(Function<dim> &init_strs,
                   Function<dim> &init_vel)
    {
        TimerOutput::Scope t(pd->computing_timer, "init_cond");

        MatrixFreeCalculation<dim> mfc(*pd, pd->strs_matrix_free, 
                pd->strs_hanging_node_constraints);
        StrsProjectionOpFactory<dim> strs_op_factory;
        mfc.op_factory = &strs_op_factory;
        mfc.apply(old_strs, &init_strs);

        MatrixFreeCalculation<dim> mfc2(*pd, pd->vel_matrix_free, 
                pd->vel_hanging_node_constraints, true);
        VelProjectionOpFactory<dim> vel_op_factory;
        mfc2.op_factory = &vel_op_factory;
        mfc2.apply(old_vel, &init_vel);
    }

    template <int dim>
    void
    Solution<dim>::
    start_timestep()
    {
        //Flip the solns to retain the old soln.
        old_old_strs.swap(old_strs);
        old_strs.swap(cur_strs);
        old_vel.swap(cur_vel);
    }

    template <int dim>
    void Solution<dim>::output(std::string data_dir,
                               std::string filename,
                               Function<dim> &exact)
    {
        TimerOutput::Scope t(pd->computing_timer, "output");


        const bool compare_with_exact = bp::extract<bool>
            (pd->parameters["test_output"]);
        parallel::distributed::Vector<double> exact_vel;
        parallel::distributed::Vector<double> error;
        if (compare_with_exact)
        {
            exact_vel.reinit(pd->vel_locally_owned_dofs, 
                             pd->vel_locally_relevant_dofs,
                             pd->mpi_comm);
            VectorTools::interpolate(pd->vel_dof_handler, exact, exact_vel);
            exact_vel.compress(VectorOperation::add);
            exact_vel.update_ghost_values();

            error.reinit(pd->vel_locally_owned_dofs, 
                         pd->vel_locally_relevant_dofs,
                         pd->mpi_comm);
            error = cur_vel;
            error -= exact_vel;

            const double total_local_error = error.l2_norm();
            const double total_global_error = std::sqrt (
                    dealii::Utilities::MPI::sum (
                        total_local_error * 
                        total_local_error, pd->mpi_comm));
            const double total_local_exact_norm = exact_vel.l2_norm();
            const double total_global_exact_norm = std::sqrt (
                    dealii::Utilities::MPI::sum (
                        total_local_exact_norm * 
                        total_local_exact_norm, pd->mpi_comm));
            const double true_error = total_global_error / total_global_exact_norm;
            pd->pcout << "Scaled error: " << true_error << std::endl;
        }

        /* Create our data saving object.
         */
        DataOut<dim> data_out;
        if (compare_with_exact) 
        {
            data_out.add_data_vector(pd->vel_dof_handler, exact_vel, "exact_vel");
            data_out.add_data_vector(pd->vel_dof_handler, error, "error");
        }
        data_out.add_data_vector(pd->vel_dof_handler, cur_vel, "vel");
        data_out.add_data_vector(pd->vel_dof_handler, poisson_soln, "poisson_soln");

        // Output the stresses.
        std::vector<std::string> old_solution_names(2);
        old_solution_names[0] = "old_szx";
        old_solution_names[1] = "old_szy";
        //Current stresses
        data_out.add_data_vector(pd->strs_dof_handler, old_strs, old_solution_names);

        // Output the stresses.
        std::vector<std::string> cur_solution_names(2);
        cur_solution_names[0] = "cur_szx";
        cur_solution_names[1] = "cur_szy";
        //Current stresses
        data_out.add_data_vector(pd->strs_dof_handler, cur_strs, cur_solution_names);
                                 
        std::vector<std::string> tent_solution_names(2);
        tent_solution_names[0] = "tent_szx";
        tent_solution_names[1] = "tent_szy";
        //Tentative stresses
        data_out.add_data_vector(pd->strs_dof_handler, tent_strs, tent_solution_names);


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
    void
    Solution<dim>::
    start_refine()
    {
        TimerOutput::Scope t(pd->computing_timer, "refine");

        parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >*
            vel_sol_trans = new parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >(pd->vel_dof_handler);    

        parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >*
            strs_sol_trans = new parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >(pd->strs_dof_handler);    

        std::vector<const parallel::distributed::Vector<double>* > vel_vecs(1);
        vel_vecs[0] = &cur_vel;

        std::vector<const parallel::distributed::Vector<double>* > strs_vecs(2);
        strs_vecs[0] = &cur_strs;
        strs_vecs[1] = &old_strs;
        vel_sol_trans->prepare_for_coarsening_and_refinement(vel_vecs);
        strs_sol_trans->prepare_for_coarsening_and_refinement(strs_vecs);

        std::vector<parallel::distributed::SolutionTransfer<dim, 
                parallel::distributed::Vector<double> >* > retval(2);
        retval[0] = vel_sol_trans;
        retval[1] = strs_sol_trans;
        sol_trans = retval;
    }
    

    template <int dim>
    void
    Solution<dim>::
    post_refine(Solution<dim> &soln)
    {
        TimerOutput::Scope t(pd->computing_timer, "refine");
        std::vector<parallel::distributed::Vector<double>* > vel_vecs(1);
        vel_vecs[0] = &cur_vel;
        soln.sol_trans[0]->interpolate(vel_vecs);

        std::vector<parallel::distributed::Vector<double>* > strs_vecs(2);
        strs_vecs[0] = &cur_strs;
        strs_vecs[1] = &old_strs;
        soln.sol_trans[1]->interpolate(strs_vecs);
        delete soln.sol_trans[0];
        delete soln.sol_trans[1];
    }

    template class Solution<2>;
    template class Solution<3>;
}
