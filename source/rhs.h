#ifndef __viscosaur_rhs_h
#define __viscosaur_rhs_h
#include <vector>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>
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
                     const unsigned int dofs_per_cell,
                     std::vector<dealii::types::global_dof_index> indices) = 0;
            virtual void start_assembly() = 0;
    };

    template <int dim>
    class SinRHS: public PoissonRHS<dim>
    {
        public:
            virtual void fill_cell_rhs(dealii::Vector<double> &cell_rhs,       
                                     dealii::FEValues<dim> &fe_values,
                                     const unsigned int n_q_points,
                                     const unsigned int dofs_per_cell,
                     std::vector<dealii::types::global_dof_index> indices);
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
                     const unsigned int dofs_per_cell,
                     std::vector<dealii::types::global_dof_index> indices);
            virtual void start_assembly();
        private:
            class InvViscosity: public dealii::Function<dim>
            {
                public:
                    InvViscosity(ProblemData<dim> &p_pd);

                    virtual double value(const dealii::Point<dim> &p,
                            const unsigned int component) const
                    {
                        if (p(1) < layer_depth)
                        {
                            return 0;
                        }
                        return inv_viscosity;
                    }
                    double layer_depth;
                    double inv_viscosity;
            };
            
            LA::MPI::SparseMatrix diff_matrix[dim];
            LA::MPI::Vector rhs;
            dealii::Function<dim>* init_cond_Szx;
            dealii::Function<dim>* init_cond_Szy;
            dealii::ConstraintMatrix constraints;
            ProblemData<dim>* pd;
    };
}
#endif
