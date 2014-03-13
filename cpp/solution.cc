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
        TimerOutput::Scope t(pd->computing_timer, "setup_solution");
        pd->mem_matrix_free.initialize_dof_vector(cur_mem);
        pd->mem_matrix_free.initialize_dof_vector(old_mem);
        pd->disp_matrix_free.initialize_dof_vector(cur_disp);
        pd->disp_matrix_free.initialize_dof_vector(old_disp);
        pd->disp_matrix_free.initialize_dof_vector(old_old_disp);
    }

    template <int dim>
    void
    Solution<dim>::
    apply_init_cond(Function<dim> &init_mem,
                    Function<dim> &init_disp)
    {
        TimerOutput::Scope t(pd->computing_timer, "init_cond");

        MatrixFreeCalculation<dim> mfc(*pd, pd->mem_matrix_free, 
                pd->mem_hanging_node_constraints);
        MemProjectionOpFactory<dim> mem_op_factory;
        mfc.op_factory = &mem_op_factory;
        mfc.apply(cur_mem, &init_mem);

        MatrixFreeCalculation<dim> mfc2(*pd, pd->disp_matrix_free, 
                pd->disp_hanging_node_constraints, true);
        DispProjectionOpFactory<dim> disp_op_factory;
        mfc2.op_factory = &disp_op_factory;
        mfc2.apply(cur_disp, &init_disp);
    }

    template <int dim>
    void
    Solution<dim>::
    start_timestep()
    {
        //Put cur_mem in old_mem
        old_mem.swap(cur_mem);
        // Put old_disp in old_old_disp
        old_old_disp.swap(old_disp);
        // Put cur_disp in old_disp
        old_disp.swap(cur_disp);
    }

    template <int dim>
    void Solution<dim>::output(std::string data_dir,
                               std::string filename,
                               Function<dim> &exact)
    {
        TimerOutput::Scope t(pd->computing_timer, "output");


        const bool compare_with_exact = bp::extract<bool>
            (pd->parameters["test_output"]);
        parallel::distributed::Vector<double> exact_disp;
        parallel::distributed::Vector<double> error;
        if (compare_with_exact)
        {
            exact_disp.reinit(pd->disp_locally_owned_dofs, 
                             pd->disp_locally_relevant_dofs,
                             pd->mpi_comm);
            VectorTools::interpolate(pd->disp_dof_handler, exact, exact_disp);
            exact_disp.compress(VectorOperation::add);
            exact_disp.update_ghost_values();

            error.reinit(pd->disp_locally_owned_dofs, 
                         pd->disp_locally_relevant_dofs,
                         pd->mpi_comm);
            error = cur_disp;
            error -= exact_disp;

            const double total_local_error = error.l2_norm();
            const double total_global_error = std::sqrt (
                    dealii::Utilities::MPI::sum (
                        total_local_error * 
                        total_local_error, pd->mpi_comm));
            const double total_local_exact_norm = exact_disp.l2_norm();
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
            data_out.add_data_vector(pd->disp_dof_handler, exact_disp, "exact_disp");
            data_out.add_data_vector(pd->disp_dof_handler, error, "error");
        }
        data_out.add_data_vector(pd->disp_dof_handler, cur_disp, "disp");

        // Output the stresses.
        std::vector<std::string> old_solution_names(2);
        old_solution_names[0] = "old_mem_zx";
        old_solution_names[1] = "old_mem_zy";
        //Current stresses
        data_out.add_data_vector(pd->mem_dof_handler, old_mem, old_solution_names);

        // Output the stresses.
        std::vector<std::string> cur_solution_names(2);
        cur_solution_names[0] = "cur_mem_zx";
        cur_solution_names[1] = "cur_mem_zy";
        //Current stresses
        data_out.add_data_vector(pd->mem_dof_handler, cur_mem, cur_solution_names);
                                 
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
            // And the list for the cross-time visit controller file
            // TODO: Get rid of the static variables!
            static std::vector<std::vector<std::string> > filenames;
            for (unsigned int i=0;
                    i<Utilities::MPI::n_mpi_processes(pd->mpi_comm);
                    ++i)
            {
                std::string name = filename + 
                    Utilities::int_to_string(i, 4) + ".vtu";
                files.push_back(name);
            }
            filenames.push_back(files);

            std::string pvtu_master_filename = filename + 
                    Utilities::int_to_string(this_subd, 4) + ".pvtu";
            std::ofstream master_output((data_dir + "/" + pvtu_master_filename).c_str());
            // Write the master pvtu record. Load this pvtu file if you want
            // to view all the data at once. Use a tool like visit or paraview.
            data_out.write_pvtu_record(master_output, files);

            std::ofstream visit_output((data_dir + "/master.visit").c_str());
            data_out.write_visit_record(visit_output, filenames);
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
            disp_sol_trans = new parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >(pd->disp_dof_handler);    

        parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >*
            mem_sol_trans = new parallel::distributed::SolutionTransfer<dim, 
            parallel::distributed::Vector<double> >(pd->mem_dof_handler);    

        std::vector<const parallel::distributed::Vector<double>* > disp_vecs(1);
        disp_vecs[0] = &cur_disp;

        std::vector<const parallel::distributed::Vector<double>* > mem_vecs(2);
        mem_vecs[0] = &cur_mem;
        mem_vecs[1] = &old_mem;
        disp_sol_trans->prepare_for_coarsening_and_refinement(disp_vecs);
        mem_sol_trans->prepare_for_coarsening_and_refinement(mem_vecs);

        std::vector<parallel::distributed::SolutionTransfer<dim, 
                parallel::distributed::Vector<double> >* > retval(2);
        retval[0] = disp_sol_trans;
        retval[1] = mem_sol_trans;
        sol_trans = retval;
    }
    

    template <int dim>
    void
    Solution<dim>::
    post_refine(Solution<dim> &soln)
    {
        TimerOutput::Scope t(pd->computing_timer, "refine");
        std::vector<parallel::distributed::Vector<double>* > disp_vecs(1);
        disp_vecs[0] = &cur_disp;
        soln.sol_trans[0]->interpolate(disp_vecs);

        std::vector<parallel::distributed::Vector<double>* > mem_vecs(2);
        mem_vecs[0] = &cur_mem;
        mem_vecs[1] = &old_mem;
        soln.sol_trans[1]->interpolate(mem_vecs);
        delete soln.sol_trans[0];
        delete soln.sol_trans[1];
    }

    template class Solution<2>;
    template class Solution<3>;
}
