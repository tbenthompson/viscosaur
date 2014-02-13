#ifndef __viscosaur_op_factory_h
#define __viscosaur_op_factory_h

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
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
                void* data) = 0;
    };
}
    
//NOTE: The use of a void* data pointer allows me to keep this op factory 
//generic
//and reduce boilerplate code. However, the downside is that it is super ugly
//and requires typecasts to retrieve data at the beginning of any actual
//application method.
//BEGIN MACROS
#define VISCOSAUR_OP_CALL(degree, op_cls) \
    if(deg == degree) \
    {\
        op_cls<dim, degree> op;\
        op.hp_local_apply(pd, dst, src, cell_range, data);\
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
                void* data)\
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
//END OF MACROS

namespace viscosaur
{
    template <int dim, int fe_degree>
    class MassMatrixOp
    {
        public:
        void hp_local_apply(ProblemData<dim> &pd, 
             dealii::parallel::distributed::Vector<double> &dst,
             const std::vector<
                    dealii::parallel::distributed::Vector<double> > &src,
             const std::pair<unsigned int, unsigned int> &cell_range,
             void* data)
        {
            dealii::Tensor<1, dim, dealii::VectorizedArray<double> > tensor_one;
            for(int d = 0; d < dim; d++) 
            {
                for(unsigned int array_el = 0; array_el < 
                        tensor_one[0].n_array_elements; array_el++)
                {
                    tensor_one[d][array_el] = 1.0;
                }
            }

            dealii::FEEvaluationGL<dim, fe_degree, dim> 
                fe_eval(pd.strs_matrix_free);

            const unsigned int n_q_points = fe_eval.n_q_points;

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                fe_eval.reinit (cell);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    fe_eval.submit_value(tensor_one, q);
                }
                fe_eval.integrate(true, false);
                fe_eval.distribute_local_to_global(dst);
            }
        }
    };

    //This line creates MassMatrixOpFactory
    VISCOSAUR_OP_FACTORY(MassMatrixOp);
    
    template <int dim, int fe_degree>
    class ProjectionOp
    {
        public:
        void hp_local_apply(ProblemData<dim> &pd, 
             dealii::parallel::distributed::Vector<double> &dst,
             const std::vector<
                    dealii::parallel::distributed::Vector<double> > &src,
             const std::pair<unsigned int, unsigned int> &cell_range,
             void* data)
        {
            dealii::Function<dim>* fnc = (dealii::Function<dim>*)data;

            dealii::FEEvaluationGL<dim, fe_degree, dim> 
                fe_eval(mf);

            const unsigned int n_q_points = fe_eval.n_q_points;

            for (unsigned int cell = cell_range.first; 
                    cell < cell_range.second;
                    ++cell)
            {
                fe_eval.reinit (cell);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    fe_eval.submit_value(fnc(q), q);
                }
                fe_eval.integrate(true, false);
                fe_eval.distribute_local_to_global(dst);
            }
        }
    };

    //This line creates ProjectionOpFactory
    VISCOSAUR_OP_FACTORY(ProjectionOp);
}
#endif
