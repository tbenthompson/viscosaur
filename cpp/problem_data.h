#ifndef __viscosaur_problem_data_h
#define __viscosaur_problem_data_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
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
            ProblemData(boost::python::dict &params);
            ~ProblemData();
            void start_refine(
                    dealii::parallel::distributed::Vector<double> &soln);
            void execute_refine();

            void generate_mesh();
            void initial_refinement();
            void save_mesh(const std::string &filename);
            void load_mesh();
            void init_dofs();

            boost::python::dict           parameters;
            MPI_Comm                      mpi_comm;
            dealii::QGaussLobatto<dim>    quadrature;
            dealii::QGaussLobatto<dim-1>  face_quad;
            dealii::QGaussLobatto<1>      one_d_quad;
            dealii::ConditionalOStream    pcout;
            dealii::TimerOutput           computing_timer;
            dealii::parallel::distributed::Triangulation<dim> triangulation;

            dealii::DoFHandler<dim>       disp_dof_handler;
            dealii::FE_Q<dim> disp_fe;
            dealii::IndexSet              disp_locally_owned_dofs;
            dealii::IndexSet              disp_locally_relevant_dofs;
            dealii::ConstraintMatrix      disp_hanging_node_constraints;
            dealii::MatrixFree<dim>       disp_matrix_free;

            dealii::DoFHandler<dim>       mem_dof_handler;
            dealii::FESystem<dim>         mem_fe;
            dealii::IndexSet              mem_locally_owned_dofs;
            dealii::IndexSet              mem_locally_relevant_dofs;
            dealii::ConstraintMatrix      mem_hanging_node_constraints;
            dealii::MatrixFree<dim>       mem_matrix_free;

    };
}
#endif
