#ifndef __viscosaur_problem_data_h
#define __viscosaur_problem_data_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/matrix_free/matrix_free.h>
#include "linear_algebra.h"

#include <Python.h>
#include <boost/python/dict.hpp>

namespace dealii
{
    class ConstraintMatrix;
    class CompressedSimpleSparsityPattern;
}

namespace viscosaur
{
    template <int dim> class Solution;
    template <int dim> class InvViscosity;

    template <int dim>
    class ProblemData
    {
        public:
            ProblemData(boost::python::dict &params,
                        InvViscosity<dim>* inv_visc);
            ~ProblemData();
            void generate_mesh();
            void load_mesh();
            void init_dofs();
            void start_refine(
                    dealii::parallel::distributed::Vector<double> &soln);
            void execute_refine();
            dealii::CompressedSimpleSparsityPattern* create_sparsity_pattern(
                    dealii::ConstraintMatrix &constraints);
            dealii::ConstraintMatrix* create_constraints();

            boost::python::dict           parameters;
            MPI_Comm                      mpi_comm;
            dealii::parallel::distributed::Triangulation<dim> triangulation;
            dealii::DoFHandler<dim>       dof_handler;
            dealii::FE_Q<dim>             fe;
            dealii::QGaussLobatto<dim>    quadrature;
            dealii::QGaussLobatto<dim-1>  face_quad;
            dealii::QGaussLobatto<1>      one_d_quad;
            dealii::IndexSet              locally_owned_dofs;
            dealii::IndexSet              locally_active_dofs;
            dealii::IndexSet              locally_relevant_dofs;
            dealii::ConditionalOStream    pcout;
            dealii::TimerOutput           computing_timer;
            dealii::ConstraintMatrix      hanging_node_constraints;
            dealii::MatrixFree<dim>       matrix_free;
            InvViscosity<dim>*            inv_visc;
    };
}
#endif
