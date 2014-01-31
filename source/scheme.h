#ifndef __viscosaur_scheme_h
#define __viscosaur_scheme_h
namespace dealii
{
    template <int dim> class Function;
    namespace parallel
    {
        namespace distributed
        {
            template <typename T> class Vector;
        }
    }
}
namespace viscosaur
{
    template <int dim, int fe_degree> class StressOp;


    #define FE_DEGREE 2
    template <int dim>
    class Scheme
    {
        public:
            virtual StressOp<dim, FE_DEGREE>* get_tentative_stepper()
            {
                return this->tent_op;
            }
            virtual StressOp<dim, FE_DEGREE>* get_correction_stepper()
            {
                return this->corr_op;
            }
            virtual double poisson_rhs_factor() const = 0;
            virtual void handle_poisson_soln(Solution<dim> &soln,
                dealii::PETScWrappers::MPI::Vector& poisson_soln) const
                = 0;     
            virtual dealii::Function<dim>* handle_bc(dealii::Function<dim> &bc)
                    const = 0;
            StressOp<dim, FE_DEGREE>* tent_op;
            StressOp<dim, FE_DEGREE>* corr_op;
    };

}
#endif
