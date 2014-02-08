#ifndef __viscosaur_op_factory_h
#define __viscosaur_op_factory_h

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/lac/parallel_vector.h>
namespace viscosaur
{
    template <int dim>
    class OpFactory
    {
        public:
        virtual void call(const unsigned int deg,
                ProblemData<dim> &pd,
                dealii::parallel::distributed::Vector<double> &dst,
                const std::vector<
                    dealii::parallel::distributed::Vector <double> >&src,
                const std::pair<unsigned int,unsigned int> &cell_range,
                const double time_step) = 0;
    };
}
    
#define VISCOSAUR_OP_CALL(degree, op_cls) \
    if(deg == degree) \
    {\
        op_cls<dim, degree> op;\
        op.hp_local_apply(pd, dst, src, cell_range, time_step);\
        return;\
    }\

#define VISCOSAUR_OP_FACTORY(op_cls)\
    template <int dim>\
    class op_cls ## Factory: public OpFactory<dim>\
    {\
        public:\
        virtual void call(const unsigned int deg,\
                ProblemData<dim> &pd,\
                dealii::parallel::distributed::Vector<double> &dst,\
                const std::vector<\
                    dealii::parallel::distributed::Vector <double> >&src,\
                const std::pair<unsigned int,unsigned int> &cell_range,\
                const double time_step)\
        {\
            VISCOSAUR_OP_CALL(1, op_cls); \
            VISCOSAUR_OP_CALL(2, op_cls); \
            VISCOSAUR_OP_CALL(3, op_cls); \
            VISCOSAUR_OP_CALL(4, op_cls); \
            VISCOSAUR_OP_CALL(5, op_cls); \
            VISCOSAUR_OP_CALL(6, op_cls); \
            VISCOSAUR_OP_CALL(7, op_cls); \
            VISCOSAUR_OP_CALL(8, op_cls); \
            VISCOSAUR_OP_CALL(9, op_cls); \
            VISCOSAUR_OP_CALL(10, op_cls); \
        }\
    };
#endif
