#ifndef __viscosaur_one_step_rhs_h
#define __viscosaur_one_step_rhs_h
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include "linear_algebra.h"

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

#include "problem_data.h"
namespace viscosaur
{
    using namespace dealii;
    template <int dim>
    class PoissonRHS
    {
        public:
            virtual void fill_cell_rhs(
                     Vector<double> &cell_rhs,
                     FEValues<dim> &fe_values,
                     const unsigned int n_q_points,
                     const unsigned int dofs_per_cell) = 0;
            virtual void start_assembly() = 0;
    };


    template <int dim>
    class SinRHS: public PoissonRHS<dim>
    {
        public:
            virtual void fill_cell_rhs(Vector<double> &cell_rhs,
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
            virtual void start_assembly() {}
        private:
            double value(Point<dim> point)
            {
                if (point[1] > 0.5 + 
                        0.25 * std::sin(4.0 * numbers::PI * point[0]))
                {
                    return 1.0;
                }
                return -1.0;
            };
    };



    template <int dim>
    class OneStepRHS: public PoissonRHS<dim>
    {
        public:
            OneStepRHS(Function<dim> &init_cond_Szx,
                       Function<dim> &init_cond_Szy,
                       ProblemData<dim> &p_pd);
            virtual void fill_cell_rhs(
                     Vector<double> &cell_rhs,
                     FEValues<dim> &fe_values,
                     const unsigned int n_q_points,
                     const unsigned int dofs_per_cell);
            virtual void start_assembly() {}
        private:
            struct InitDiffPerTaskData
            {
                unsigned int              d;
                unsigned int              dpc;
                FullMatrix<double>        local_grad;
                std::vector<types::global_dof_index> local_dof_indices;

                InitDiffPerTaskData(const unsigned int p_d,
                                    const unsigned int p_dpc):
                    d(p_d),
                    dpc(p_dpc),
                    local_grad(p_dpc, p_dpc),
                    local_dof_indices(p_dpc)
                {}
            };

            struct InitDiffScratchData
            {
                unsigned int  nqp;
                FEValues<dim> fe_values;
                InitDiffScratchData(const FE_Q<dim> &fe,
                                   const QGauss<dim> &quad,
                                   const UpdateFlags flags):
                    nqp(quad.size()),
                    fe_values(fe, quad, flags)
                {}
                InitDiffScratchData (const InitDiffScratchData &data):
                    nqp (data.nqp),
                    fe_values(data.fe_values.get_fe(),
                                data.fe_values.get_quadrature(),
                                data.fe_values.get_update_flags())
                {}
            };
            // void initialize_diff_operator();
            // void copy_diff_local_to_global(const InitDiffPerTaskData &data);
            // void assemble_one_cell_of_diff(
            //     const typename DoFHandler<dim>::active_cell_iterator &cell,
            //     InitDiffScratchData &scratch,
            //     InitDiffPerTaskData &data);
            // 
            LA::MPI::SparseMatrix diff_matrix[dim];
            ConstraintMatrix constraints;
            ProblemData<dim>* pd;
    };

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
}
#endif
