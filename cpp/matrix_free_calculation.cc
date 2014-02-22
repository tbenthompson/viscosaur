#include "matrix_free_calculation.h"
#include "op_factory.h"
#include "problem_data.h"

#include <boost/python/extract.hpp>
#include <deal.II/base/function.h>
namespace viscosaur
{
    using namespace dealii;
    namespace bp = boost::python;

    template <int dim>
    MatrixFreeCalculation<dim>::MatrixFreeCalculation(
            ProblemData<dim> &p_pd,
            dealii::MatrixFree<dim> &p_mf,
            dealii::ConstraintMatrix &p_cm,
           bool scalar)
    {
        reinit(p_pd, p_mf, p_cm, scalar);
    }

    template <int dim>
    MatrixFreeCalculation<dim>::~MatrixFreeCalculation()
    {
    }
    
    template <int dim>
    void
    MatrixFreeCalculation<dim>::
    reinit(ProblemData<dim> &p_pd,
           dealii::MatrixFree<dim> &p_mf,
           dealii::ConstraintMatrix &p_cm,
           bool scalar)
    {
        pd = &p_pd;
        mf = &p_mf;
        constraints = &p_cm;
        compute_mass_matrix(scalar);
    }

    template <int dim>
    void
    MatrixFreeCalculation<dim>::
    compute_mass_matrix(bool scalar)
    {
        // Integrate and invert the diagonal mass matrix resulting from the
        // Gauss Lobatto Lagrange elements and quadrature match up.
        mf->initialize_dof_vector(inv_mass_matrix);

        if(scalar)
        {
            this->op_factory = new ScalarMassMatrixOpFactory<dim>();
        }
        else
        {
            this->op_factory = new VectorMassMatrixOpFactory<dim>();
        }
        
        // Use the mass matrix operator 
        std::vector<dealii::parallel::distributed::Vector<double>* >
            empty_source(0);

        mf->cell_loop(&MatrixFreeCalculation<dim>::local_apply, this, inv_mass_matrix,
                empty_source);

        delete this->op_factory;

        inv_mass_matrix.compress(dealii::VectorOperation::add);
        for (unsigned int k=0; k < inv_mass_matrix.local_size(); ++k)
        {
            if (inv_mass_matrix.local_element(k)>1e-15)
            {
                inv_mass_matrix.local_element(k) = 
                    1. / inv_mass_matrix.local_element(k);
            }
            else
            {
                inv_mass_matrix.local_element(k) = 0;
            }
        }
    }

    template <int dim>
    void 
    MatrixFreeCalculation<dim>::
    apply(dealii::parallel::distributed::Vector<double> &dst,
          std::vector<dealii::parallel::distributed::Vector<double>* > &sources,
          boost::any data)
    {
        dst = 0;
        
        this->data = data;

        mf->cell_loop(&MatrixFreeCalculation<dim>::local_apply, this, dst, sources);

        dst.scale(inv_mass_matrix);

        //Apply constraints to set constrained DoFs to their correct value
        constraints->distribute(dst);

        //Spread ghost values across processors
        dst.update_ghost_values();
    }

    template <int dim>
    void 
    MatrixFreeCalculation<dim>::
    apply(dealii::parallel::distributed::Vector<double> &dst,
          boost::any data)
    {
        std::vector<dealii::parallel::distributed::Vector<double>* > sources(0);
        apply(dst, sources, data);
    }

    template <int dim>
    void 
    MatrixFreeCalculation<dim>::
    apply_function(dealii::parallel::distributed::Vector<double> &dst,
                   dealii::Function<dim> &data)
    {
        apply(dst, &data);
    }

    template <int dim> 
    void
    MatrixFreeCalculation<dim>::
    local_apply(const dealii::MatrixFree<dim> &mf_obj,
                dealii::parallel::distributed::Vector<double> &dst,
                const std::vector<
                    dealii::parallel::distributed::Vector<double>* > &src,
                const std::pair<unsigned int, unsigned int> &cell_range)
    {
        // Ask MatrixFree for cell_range for different
        // orders
        const unsigned int max_degree =
            bp::extract<int>(pd->parameters["max_degree"]);
        std::pair<unsigned int, unsigned int> subrange_deg; 
        for(unsigned int deg = 0; deg < max_degree; deg++)
        {
            subrange_deg = mf_obj.create_cell_subrange_hp(cell_range, deg); 
            if (subrange_deg.second > subrange_deg.first) 
                this->op_factory->call(deg, *pd, dst, src, subrange_deg, data);
        }
    }

    template class MatrixFreeCalculation<2>;
    template class MatrixFreeCalculation<3>;
}
