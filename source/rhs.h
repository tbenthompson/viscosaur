#ifndef __viscosaur_rhs_h
#define __viscosaur_rhs_h
#include <vector>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include "linear_algebra.h"

namespace dealii
{
    template <typename T> class Vector;
    template <int dim, typename Number> class Point;
    template <int dim> class Function;
    template <int dim, int spacedim> class FE_Q;
    template <int dim> class QGauss;
}
namespace viscosaur
{
    template <int dim> class ProblemData;

    template <int dim>
    class PoissonRHS
    {
        public:
            virtual void fill_cell_rhs(
                     dealii::Vector<double> &cell_rhs,
                     dealii::FEValues<dim, dim> &fe_values,
                     const unsigned int n_q_points,
                     const unsigned int dofs_per_cell) = 0;
            virtual void start_assembly() = 0;
    };

    template <int dim>
    class SinRHS: public PoissonRHS<dim>
    {
        public:
            virtual void fill_cell_rhs(dealii::Vector<double> &cell_rhs,
                                     dealii::FEValues<dim> &fe_values,
                                     const unsigned int n_q_points,
                                     const unsigned int dofs_per_cell);
            virtual void start_assembly() {}
        private:
            double value(dealii::Point<dim, double> point);
    };


    template <int dim>
    class OneStepRHS: public PoissonRHS<dim>
    {
        public:
            OneStepRHS(dealii::Function<dim> &init_cond_Szx,
                       dealii::Function<dim> &init_cond_Szy,
                       ProblemData<dim> &p_pd);
            virtual void fill_cell_rhs(
                     dealii::Vector<double> &cell_rhs,
                     dealii::FEValues<dim> &fe_values,
                     const unsigned int n_q_points,
                     const unsigned int dofs_per_cell);
            virtual void start_assembly() {}
        private:
            struct InitDiffPerTaskData
            {
                unsigned int              d;
                unsigned int              dpc;
                dealii::FullMatrix<double>        local_grad;
                std::vector<dealii::types::global_dof_index> local_dof_indices;

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
                dealii::FEValues<dim> fe_values;
                InitDiffScratchData(const dealii::FE_Q<dim, dim> &fe,
                                   const dealii::QGauss<dim> &quad,
                                   const dealii::UpdateFlags flags):
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
            dealii::ConstraintMatrix constraints;
            ProblemData<dim>* pd;
    };
}
#endif
