#ifndef __viscosaur_problem_data_h
#define __viscosaur_problem_data_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
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
    template <int dim>
    class ProblemData
    {
        public:
            ProblemData(boost::python::dict &params);
            ~ProblemData();
            void init_mesh();
            void init_dofs();
            void refine_grid(LA::MPI::Vector &local_solution);
            dealii::CompressedSimpleSparsityPattern* create_sparsity_pattern(
                    dealii::ConstraintMatrix &constraints);
            dealii::ConstraintMatrix* create_constraints();

            boost::python::dict                      parameters;
            MPI_Comm              mpi_comm;
            dealii::parallel::distributed::Triangulation<dim> triangulation;
            dealii::DoFHandler<dim>       dof_handler;
            dealii::FE_Q<dim>             fe;
            dealii::QGaussLobatto<dim>    quadrature;
            dealii::IndexSet              locally_owned_dofs;
            dealii::IndexSet              locally_active_dofs;
            dealii::IndexSet              locally_relevant_dofs;
            dealii::ConditionalOStream    pcout;
            dealii::TimerOutput           computing_timer;
    };
}
#endif
