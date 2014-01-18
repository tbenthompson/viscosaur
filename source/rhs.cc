namespace viscosaur
{
    // namespace bp = boost::python;
    // using namespace dealii;

    // template <int dim>
    // void 
    // SinRHS<dim>::
    // fill_cell_rhs(Vector<double> &cell_rhs,
    //               FEValues<dim> &fe_values,
    //               const unsigned int n_q_points,
    //               const unsigned int dofs_per_cell,
    //                  std::vector<types::global_dof_index> indices)
    // {
    //     for (unsigned int q_point=0; q_point < n_q_points; ++q_point)          
    //     {
    //         double rhs_value = value(
    //                 fe_values.quadrature_point(q_point));
    //         for (unsigned int i=0; i<dofs_per_cell; ++i)
    //         {
    //             cell_rhs(i) += (rhs_value *
    //                     fe_values.shape_value(i,q_point) *
    //                     fe_values.JxW(q_point));
    //         }
    //     }
    // }

    // template <int dim>
    // double 
    // SinRHS<dim>::
    // value(Point<dim> point)
    // {
    //     if (point[1] > 0.5 + 
    //             0.25 * std::sin(4.0 * numbers::PI * point[0]))
    //     {
    //         return 1.0;
    //     }
    //     return -1.0;
    // }

}
