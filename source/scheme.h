#ifndef __viscosaur_scheme_h
#define __viscosaur_scheme_h
namespace viscosaur
{
    template <int dim, int fe_degree> class StressOp;


    #define FE_DEGREE 2
    template <int dim>
    class Scheme
    {
        public:
            virtual StressOp<dim, FE_DEGREE>* get_tentative_stepper() = 0;
            virtual StressOp<dim, FE_DEGREE>* get_correction_stepper() = 0;
    };
}
#endif
