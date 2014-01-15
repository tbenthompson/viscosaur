#include "rhs.h"
#include "problem_data.h"

#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>

namespace viscosaur
{
    using namespace dealii;

    template <int dim>
    void 
    SinRHS<dim>::
    fill_cell_rhs(Vector<double> &cell_rhs,
                  FEValues<dim> &fe_values,
                  const unsigned int n_q_points,
                  const unsigned int dofs_per_cell)
    {
        for (unsigned int q_point=0; q_point < n_q_points; ++q_point)
        {
            double rhs_value = value(
                    fe_values.quadrature_point(q_point));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                cell_rhs(i) += (rhs_value *
                        fe_values.shape_value(i,q_point) *
                        fe_values.JxW(q_point));
            }
        }
    }

    template <int dim>
    double 
    SinRHS<dim>::
    value(Point<dim> point)
    {
        if (point[1] > 0.5 + 
                0.25 * std::sin(4.0 * numbers::PI * point[0]))
        {
            return 1.0;
        }
        return -1.0;
    }




    template <int dim>
    OneStepRHS<dim>::
    OneStepRHS(Function<dim> &init_cond_Szx,
               Function<dim> &init_cond_Szy,
               ProblemData<dim> &p_pd)
    {
        pd = &p_pd;

        // Vector<double> Szx(dof_handler.n_dofs());
        // VectorTools::interpolate(dof_handler, init_cond_Szx, Szx);

        // Vector<double> Szy(dof_handler.n_dofs());
        // VectorTools::interpolate(dof_handler, init_cond_Szy, Szy);

    }
    //     //Create hanging node constraints for the differentiation matrix so 
    //     //that we don't get unrealistic discontinuities in the rhs. Is this
    //     //actually something that I should be doing?
    //     constraints = *pd->create_constraints();
    //     constraints.close();
    //     CompressedSimpleSparsityPattern* csp = 
    //         pd->create_sparsity_pattern(constraints);
    //     for (int n = 0; n < dim; ++n)
    //     {
    //         diff_matrix[n].reinit(pd->locally_owned_dofs,
    //                               pd->locally_owned_dofs,
    //                               *csp,
    //                               pd->mpi_comm);
    //     }

    //     // DataOut<dim> data_out;
    //     // data_out.attach_dof_handler(dof_handler);
    //     // data_out.add_data_vector(Szx, "Szx"); 
    //     // data_out.add_data_vector(Szy, "Szy"); 
    //     // data_out.build_patches();

    //     // std::ofstream output("abcdef.vtk");
    //     // data_out.write_vtk(output);
    // }

    // template <int dim>
    // void 
    // OneStepRHS<dim>::
    // initialize_diff_operator()
    // {
    //     InitDiffPerTaskData per_task_data (0, pd->fe);
    //     InitDiffScratchData scratch_data (pd->fe, pd->quadrature,
    //                                 update_gradients | update_JxW_values,
    //                                 update_values);
    //     for (unsigned int d=0; d<dim; ++d)
    //     {
    //         diff_matrix[d].reinit(sparsity_pattern);
    //         per_task_data.d = d;
    //         WorkStream::run(dof_handler.begin_active(),
    //                         dof_handler.end(),
    //                         *this,
    //                         &OneStepRHS<dim>::assemble_one_cell_of_diff,
    //                         &OneStepRHS<dim>::copy_diff_local_to_global,
    //                         scratch_data,
    //                         per_task_data);
    //     }
    // }

    // template <int dim>
    // void
    // OneStepRHS<dim>::
    // assemble_one_cell_of_diff(
    //         const typename DoFHandler<dim>::active_cell_iterator &cell,
    //                                InitDiffScratchData &scratch,
    //                                InitDiffPerTaskData &data)
    // {
    //     scratch.fe_values.reinit(std_cxx1x::get<0> (SI.iterators));

    //     std_cxx1x::get<0>(SI.iterators)->get_dof_indices
    //         (data.local_dof_indices);

    //     data.local_grad = 0.;
    //     for (unsigned int q = 0; q < scratch.nqp; ++q)
    //     {
    //         for (unsigned int i = 0; i < data.dpc; ++i)
    //         {
    //             for (unsigned int j = 0; j < data.dpc; ++j)
    //             {
    //                 data.local_grad(i, j) += -scratch.fe_values.JxW(q) *
    //                     scratch.fe_values.shape_grad(i, q)[data.d] *
    //                     scratch.fe_values.shape_value(j, q);
    //             }
    //         }
    //     }
    // }

    // template <int dim>
    // void
    // OneStepRHS<dim>::
    // copy_diff_local_to_global(InitDiffPerTaskData &data)
    // {
    //     for (unsigned int i = 0; i < data.dpc; ++i)
    //     {
    //         for (unsigned int j = 0; j < data.dpc; ++j)
    //         {
    //             diff_matrix[data.d].add(data.local_dof_indices[i],
    //                                     data.local_dof_indices[j],
    //                                     data.local_grad(i, j));
    //         }
    //     }
    // }

    template <int dim>
    void OneStepRHS<dim>::fill_cell_rhs(
             Vector<double> &cell_rhs,
             FEValues<dim> &fe_values,
             const unsigned int n_q_points,
             const unsigned int dofs_per_cell)
    {
        // rhs_tmp = 0.;
        // for (unsigned d = 0; d < dim; ++d)
        // {
        //     diff_matrix[d].Tvmult_add (pres_tmp, u_n[d]);
        // }
    }


    // ESSENTIAL: explicity define the template types we will use.
    // Otherwise, the template definition needs to go in the header file, which
    // is ugly!
    template class PoissonRHS<2>;
    template class PoissonRHS<3>;
    template class SinRHS<2>;
    template class SinRHS<3>;
    template class OneStepRHS<2>;
    template class OneStepRHS<3>;
}
