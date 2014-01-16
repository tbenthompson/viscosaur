#include "rhs.h"
#include "problem_data.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <Python.h>
#include <boost/python/extract.hpp>

namespace viscosaur
{
    namespace bp = boost::python;
    using namespace dealii;

    template <int dim>
    void 
    SinRHS<dim>::
    fill_cell_rhs(Vector<double> &cell_rhs,
                  FEValues<dim> &fe_values,
                  const unsigned int n_q_points,
                  const unsigned int dofs_per_cell,
                     std::vector<types::global_dof_index> indices)
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
    OneStepRHS(Function<dim> &p_init_cond_Szx,
               Function<dim> &p_init_cond_Szy,
               ProblemData<dim> &p_pd)
    {
        pd = &p_pd;
        init_cond_Szx = &p_init_cond_Szx;
        init_cond_Szy = &p_init_cond_Szy;

        // start_assembly();
        // DataOut<dim> data_out;
        // data_out.attach_dof_handler(dof_handler);
        // data_out.add_data_vector(Szx, "Szx"); 
        // data_out.add_data_vector(Szy, "Szy"); 
        // data_out.build_patches();

        // std::ofstream output("abcdef.vtk");
        // data_out.write_vtk(output);
    }

    template <int dim>
    void 
    OneStepRHS<dim>::
    start_assembly()
    {
        TimerOutput::Scope t(pd->computing_timer, "rhs_assembly");
        //Create sparsity pattern and distribute matrices amongst processors.
        ConstraintMatrix temp;
        CompressedSimpleSparsityPattern* csp = 
            pd->create_sparsity_pattern(temp);
        for (int d = 0; d < dim; d++)
        {
            diff_matrix[d].reinit(pd->locally_owned_dofs,
                                  pd->locally_owned_dofs,
                                  *csp,
                                  pd->mpi_comm);
        }

        const unsigned int   dofs_per_cell = pd->fe.dofs_per_cell;
        const unsigned int   n_q_points    = pd->quadrature.size();
        FEValues<dim> fe_values(pd->fe, pd->quadrature,
                                update_values | update_gradients | 
                                update_quadrature_points | update_JxW_values); 
        typename DoFHandler<dim>::active_cell_iterator
            cell = pd->dof_handler.begin_active(),
            endc = pd->dof_handler.end();
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        double value;
        const double shear_modulus = 
            bp::extract<double>(pd->parameters["shear_modulus"]);
        const double time_step = 
            bp::extract<double>(pd->parameters["time_step"]);
        const double factor = 1.0 / (shear_modulus * time_step);
        for (; cell!=endc; ++cell)
        {
            if (!cell->is_locally_owned())
            {
                continue;
            }
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        for (unsigned int d = 0; d < dim; d++)
                        {
                            value = factor * fe_values.JxW(q) *
                                fe_values.shape_grad(i, q)[d] *
                                fe_values.shape_value(j, q);
                            diff_matrix[d].add(local_dof_indices[i],
                                               local_dof_indices[j],
                                               value);
                        }
                    }
                }
            }
        }
        for(int d = 0; d < dim; d++)
        {
            diff_matrix[d].compress(VectorOperation::add);
        }

        LA::MPI::Vector Szx;
        LA::MPI::Vector Szy;
        LA::MPI::Vector inv_visc;
        InvViscosity fnc(*pd);
        {
            TimerOutput::Scope t(pd->computing_timer, "interpolation");

            Szx.reinit(pd->locally_owned_dofs, pd->mpi_comm);
            VectorTools::interpolate(pd->dof_handler, *init_cond_Szx, Szx);
            Szx.compress(VectorOperation::add);

            Szy.reinit(pd->locally_owned_dofs, pd->mpi_comm);
            VectorTools::interpolate(pd->dof_handler, *init_cond_Szy, Szy);
            Szy.compress(VectorOperation::add);

            inv_visc.reinit(pd->locally_owned_dofs, pd->mpi_comm);
            VectorTools::interpolate(pd->dof_handler, fnc, inv_visc);
            inv_visc.compress(VectorOperation::add);
            inv_visc *= -shear_modulus * time_step;
            inv_visc.add(1.0);
        }
        Szx.scale(inv_visc);
        Szy.scale(inv_visc);

        rhs.reinit(pd->locally_owned_dofs, pd->mpi_comm);
        rhs = 0;
        diff_matrix[0].Tvmult_add(rhs, Szx);
        diff_matrix[1].Tvmult_add(rhs, Szy);
    }

    template <int dim>
    void OneStepRHS<dim>::fill_cell_rhs(
             Vector<double> &cell_rhs,
             FEValues<dim> &fe_values,
             const unsigned int n_q_points,
             const unsigned int dofs_per_cell,
             std::vector<types::global_dof_index> indices)
    {
        for(int i = 0; i < indices.size(); i++)
        {
            if (!rhs.in_local_range(indices[i])){
                // cout << "WEIRD" << std::endl;
                continue;
            }
            cell_rhs[i] = rhs[indices[i]];
        }
    }

    template <int dim>
    OneStepRHS<dim>::InvViscosity::InvViscosity(ProblemData<dim> &p_pd){
        layer_depth = 
            bp::extract<double>(p_pd.parameters["fault_depth"]);
        inv_viscosity = 1.0 /
            bp::extract<double>(p_pd.parameters["viscosity"]);
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
