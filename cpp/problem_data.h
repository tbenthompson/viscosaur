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
            ProblemData(boost::python::dict &params,
                        InvViscosity<dim>* inv_visc);
            ~ProblemData();
            void start_refine(
                    dealii::parallel::distributed::Vector<double> &soln);
            void execute_refine();

            dealii::CompressedSimpleSparsityPattern* 
                create_vel_sparsity_pattern(
                    dealii::ConstraintMatrix &constraints);

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

            InvViscosity<dim>*            inv_visc;

            dealii::DoFHandler<dim>       vel_dof_handler;
            dealii::FE_Q<dim>             vel_fe;
            dealii::IndexSet              vel_locally_owned_dofs;
            dealii::IndexSet              vel_locally_relevant_dofs;
            dealii::ConstraintMatrix      vel_hanging_node_constraints;
            dealii::MatrixFree<dim>       vel_matrix_free;

            dealii::DoFHandler<dim>       strs_dof_handler;
            dealii::FESystem<dim>         strs_fe;
            dealii::IndexSet              strs_locally_owned_dofs;
            dealii::IndexSet              strs_locally_relevant_dofs;
            dealii::ConstraintMatrix      strs_hanging_node_constraints;
            dealii::MatrixFree<dim>       strs_matrix_free;

    };
}
#endif
