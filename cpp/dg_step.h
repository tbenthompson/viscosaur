#ifndef __viscosaur_dg_step_h
#define __viscosaur_dg_step_h
#include "matrix_free_calculation.h"
#include "op_factory.h"
#include "inv_visc.h"
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/tensor.h>
namespace viscosaur
{
    template <int dim> class ProblemData;
    template <int dim> class Solution;

    template <int dim>
    struct EvalData
    {
        dealii::VectorizedArray<double> mu;
        dealii::VectorizedArray<double> inv_rho;
        double C;
        InvViscosity<dim>* iv;
    };

    template <int dim, int fe_degree>
    class StrsEvalDeriv
    {
        public:
        void hp_local_apply(ProblemData<dim> &pd, 
             dealii::parallel::distributed::Vector<double> &dst,
             const std::vector<
                    dealii::parallel::distributed::Vector<double>* > &src,
             const std::pair<unsigned int, unsigned int> &cell_range,
             boost::any data)
        {
            // Data structures for the cell assembly
            EvalData<dim>* d = boost::any_cast<EvalData<dim>*>(data);
            dealii::FEEvaluationGL<dim, fe_degree> vel_eval(pd.vel_matrix_free);
            dealii::FEEvaluationGL<dim, fe_degree, dim> strs_eval(pd.strs_matrix_free);
            const unsigned int n_q_points = strs_eval.n_q_points;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > 
                old_value;
            dealii::VectorizedArray<double> v_val;
            dealii::VectorizedArray<double> iv;

            // Data structures for the face assembly
            dealii::FEFaceValues<dim> vel_face_values(pd.vel_fe, pd.face_quad,
                                      dealii::update_values);
            dealii::FEFaceValues<dim> vel_face_values_neighbor(pd.vel_fe, 
                                      pd.face_quad, 
                                      dealii::update_values);
            dealii::FEFaceValues<dim> strs_face_values(pd.strs_fe, pd.face_quad,
                                      dealii::update_values | 
                                      dealii::update_normal_vectors | 
                                      dealii::update_JxW_values);
            const dealii::FEValuesExtractors::Vector strs_access(0);
            const unsigned int n_face_q_points = pd.face_quad.size();
            const unsigned int dofs_per_cell = pd.strs_fe.dofs_per_cell;
            double avg_flux, jump_flux;
            dealii::Tensor<1, dim> normal;
            int neighbor_face, component_i;
            std::vector<double> this_vel_face(n_face_q_points);
            std::vector<double> neighbor_vel_face(n_face_q_points);
            std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                vel_eval.reinit(cell);
                strs_eval.reinit(cell);
                vel_eval.read_dof_values(*src[0]);
                strs_eval.read_dof_values(*src[1]);
                vel_eval.evaluate(true, false, false);
                strs_eval.evaluate(true, false, false);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    //Eval deriv here
                    old_value = strs_eval.get_value(q);
                    v_val = vel_eval.get_value(q);
                    iv = d->iv->value(strs_eval.quadrature_point(q),
                                old_value);
                    //strs_eval.submit_value(-d->mu * iv * old_value, q);
                    strs_eval.submit_divergence(-d->mu * v_val, q);
                }
                // strs_eval.integrate(true, true);
                strs_eval.integrate(false, true);
                strs_eval.distribute_local_to_global(dst);
                //Before adding face terms, took 3.77secs

                for (unsigned int v=0; v<pd.strs_matrix_free.n_components_filled(cell); ++v)
                {
                    typename dealii::DoFHandler<dim>::active_cell_iterator cell_iterator =
                        pd.strs_matrix_free.get_cell_iterator(cell, v);
                    typename dealii::DoFHandler<dim>::active_cell_iterator
                        cell_iterator_vel(&pd.triangulation, cell_iterator->level(),
                            cell_iterator->index(), &pd.vel_dof_handler);
                    if (!cell_iterator_vel->is_locally_owned())
                    {
                        continue;
                    }
                    cell_iterator->get_dof_indices(local_dof_indices);

                    // Compute the contribution of the face terms resulting from the
                    // integration by parts of the divergence on the rhs
                    // I will violate the 80 characters rule here. Just too many nested for loops
                    for (unsigned int face = 0; 
                            face < dealii::GeometryInfo<dim>::faces_per_cell; 
                            ++face)
                    {
                        vel_face_values.reinit(cell_iterator_vel, face);
                        strs_face_values.reinit(cell_iterator, face);

                        vel_face_values.get_function_values(*src[0], this_vel_face);
                        bool boundary;
                        if (cell_iterator->face(face)->at_boundary())
                        {
                            boundary = true;
                            for(int nvf = 0; nvf < neighbor_vel_face.size(); nvf++)
                            {
                                neighbor_vel_face[nvf] = 0;
                            }
                        } 
                        else
                        {
                            const typename dealii::DoFHandler<dim>::active_cell_iterator
                                neighbor = cell_iterator_vel->neighbor(face);
                            neighbor_face = cell_iterator_vel->neighbor_of_neighbor(face);
                            vel_face_values_neighbor.reinit(neighbor, neighbor_face);
                            vel_face_values_neighbor.get_function_values(*src[0],
                                            neighbor_vel_face);
                        }
                        for (unsigned int q= 0; q < n_face_q_points; ++q)
                        {
                            normal = strs_face_values.normal_vector(q);
                            avg_flux = 0.5 * d->mu[0] * (this_vel_face[q] + neighbor_vel_face[q]);
                            if (boundary)
                            {
                                avg_flux *= 2; // outflow boundary condition
                                jump_flux = 0.0;
                            }
                            else
                            {
                                jump_flux = 0.5 * d->C * (this_vel_face[q] - neighbor_vel_face[q]);
                            }
                            for (unsigned int i=0; i < dofs_per_cell; ++i)
                            {
                                component_i = pd.strs_fe.system_to_component_index(i).first;
                                dst(local_dof_indices[i]) += 0.5 * (avg_flux + jump_flux) * 
                                    normal[component_i] * 
                                    strs_face_values[strs_access].value(i, q)[component_i] *
                                    strs_face_values.JxW(q);
                            }
                        }
                    }
                }
            }
        }
    };

    template <int dim, int fe_degree>
    class VelEvalDeriv
    {
        public:
        void hp_local_apply(ProblemData<dim> &pd, 
             dealii::parallel::distributed::Vector<double> &dst,
             const std::vector<
                    dealii::parallel::distributed::Vector<double>* > &src,
             const std::pair<unsigned int, unsigned int> &cell_range,
             boost::any data)
        {
            //Data structures for the cell assembly
            EvalData<dim>* d = boost::any_cast<EvalData<dim>*>(data);
            dealii::FEEvaluationGL<dim, fe_degree> vel_eval(pd.vel_matrix_free);
            dealii::FEEvaluationGL<dim, fe_degree, dim> strs_eval(pd.strs_matrix_free);
            const unsigned int n_q_points = vel_eval.n_q_points;
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > 
                strs_val;

            // Data structures for the face assembly
            dealii::FEFaceValues<dim> strs_face_values(pd.strs_fe, pd.face_quad,
                                      dealii::update_values);
            dealii::FEFaceValues<dim> strs_face_values_neighbor(pd.strs_fe, 
                                      pd.face_quad, 
                                      dealii::update_values);
            dealii::FEFaceValues<dim> vel_face_values(pd.vel_fe, pd.face_quad,
                                      dealii::update_values | 
                                      dealii::update_normal_vectors | 
                                      dealii::update_JxW_values);
            const dealii::FEValuesExtractors::Vector strs_access(0);
            const unsigned int n_face_q_points = pd.face_quad.size();
            const unsigned int dofs_per_cell = pd.vel_fe.dofs_per_cell;
            dealii::Tensor<1, dim> avg_flux;
            double jump_flux;
            dealii::Tensor<1, dim> normal;
            int neighbor_face, component_i;
            std::vector<dealii::Tensor<1, dim> > this_strs_face(n_face_q_points);
            std::vector<dealii::Tensor<1, dim> > neighbor_strs_face(n_face_q_points);
            std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                vel_eval.reinit (cell);
                strs_eval.reinit (cell);
                strs_eval.read_dof_values(*src[1]);
                strs_eval.evaluate(true, false, false);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    //Eval deriv here
                    strs_val = strs_eval.get_value(q);
                    vel_eval.submit_gradient(-d->inv_rho * strs_val, q);
                }
                vel_eval.integrate(false, true);
                vel_eval.distribute_local_to_global(dst);

                for (unsigned int v=0; v<pd.vel_matrix_free.n_components_filled(cell); ++v)
                {
                    typename dealii::DoFHandler<dim>::active_cell_iterator cell_iterator_vel = 
                        pd.vel_matrix_free.get_cell_iterator(cell, v);
                    typename dealii::DoFHandler<dim>::active_cell_iterator cell_iterator_strs = 
                        pd.strs_matrix_free.get_cell_iterator(cell, v);
                    if (!cell_iterator_vel->is_locally_owned())
                    {
                        continue;
                    }
                    cell_iterator_vel->get_dof_indices(local_dof_indices);

                    // Compute the contribution of the face terms resulting from the
                    // integration by parts of the divergence on the rhs
                    // I will violate the 80 characters rule here. Just too many nested for loops
                    for (unsigned int face = 0; 
                            face < dealii::GeometryInfo<dim>::faces_per_cell; 
                            ++face)
                    {
                        vel_face_values.reinit(cell_iterator_vel, face);
                        strs_face_values.reinit(cell_iterator_strs, face);

                        strs_face_values[strs_access].get_function_values(*src[1], this_strs_face);
                        bool boundary;
                        if (cell_iterator_vel->face(face)->at_boundary())
                        {
                            boundary = true;
                            for(int nsf = 0; nsf < neighbor_strs_face.size(); nsf++)
                            {
                                neighbor_strs_face[nsf] = 0;
                            }
                        } 
                        else
                        {
                            const typename dealii::DoFHandler<dim>::active_cell_iterator
                                neighbor = cell_iterator_strs->neighbor(face);
                            neighbor_face = cell_iterator_strs->neighbor_of_neighbor(face);
                            strs_face_values_neighbor.reinit(neighbor, neighbor_face);
                            strs_face_values_neighbor[strs_access].get_function_values(*src[1],
                                            neighbor_strs_face);
                        }
                        for (unsigned int q= 0; q < n_face_q_points; ++q)
                        {
                            normal = vel_face_values.normal_vector(q);
                            avg_flux = 0.5 * d->inv_rho[0] * (this_strs_face[q] + neighbor_strs_face[q]);
                            if (boundary)
                            {
                                avg_flux *= 2; // outflow boundary condition
                                jump_flux = 0.0;
                            }
                            else
                            {
                                jump_flux = 0.5 * d->C * normal * (this_strs_face[q] - neighbor_strs_face[q]);
                            }
                            for (unsigned int i=0; i < dofs_per_cell; ++i)
                            {
                                dst(local_dof_indices[i]) += 0.5 * (avg_flux * normal +
                                        jump_flux) *
                                    vel_face_values.shape_value(i, q) *
                                    vel_face_values.JxW(q);
                            }
                        }
                    }
                }
            }
        }
    };

    //This line creates ProjectionOpFactory
    VISCOSAUR_OP_FACTORY(StrsEvalDeriv);
    VISCOSAUR_OP_FACTORY(VelEvalDeriv);

    template <int dim>
    class DGStep
    {
        public:
            DGStep(double dt);
            void step(ProblemData<dim> &pd,
                      Solution<dim> &soln,
                      InvViscosity<dim> &iv,
                      double inv_rho);
            double dt;
    };
}
#endif
