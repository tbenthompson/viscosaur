#include <Python.h>
#if PY_VERSION_HEX >= 0x03000000
    // The code should never get here, but if it does,
    // we should quit, because the conflicting interpreter and include
    // will cause weird problems.
    #error "Python 3?!" 
#endif
#include <boost/python.hpp>
#include <boost/python/suite/indexing/indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/array.hpp>

#include "tla.h"
#include "inv_visc.h"
#include "problem_data.h"
#include "control.h"
#include "solution.h"
#include "matrix_free_calculation.h"
#include "op_factory.h"
#include "dg_step.h"

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>
namespace vc = viscosaur;



class PythonFunction2D: public dealii::Function<2>
{
    public:
        virtual double value(const dealii::Point<2> &p,
                             const unsigned int component) 
        {
            this->get_value(p[0], p[1], component);
        }

        virtual double value(const dealii::Point<2> &p,
                             unsigned int component) const
        {
            this->get_value(p[0], p[1], component);
        }

        virtual double get_value(const double x, 
                             const double y,
                             const double component) const = 0;
};
class PythonFunction2DWrap: public PythonFunction2D,
                    public boost::python::wrapper<dealii::Function<2> >
{
    virtual double get_value(const double x,
                         const double y,
                         const double component) const
    {
        return this->get_override("get_value")(x, y, component);
    }
};


/* Note: error: ‘dealii::DoFHandler<dim, spacedim>::DoFHandler(const dealii::
 * DoFHandler<dim, spacedim>&) [with int dim = 2; int spacedim = 2]’ is private
 * means that a class' copy constructor is private and we should use
 * boost::noncopyable
 */
BOOST_PYTHON_MODULE(viscosaur)
{
    using namespace boost::python;

    /* Expose some dealii classes.
     */
    class_<dealii::Point<2> >("Point2D", init<double, double>());
    class_<dealii::Function<2>, boost::noncopyable>("Function2D", no_init)
        .def("value", pure_virtual(&dealii::Function<2>::value));
    class_<dealii::ZeroFunction<2>, boost::noncopyable,
        bases<dealii::Function<2> > >("ZeroFunction2D", init<int>())
        .def("value", &dealii::ZeroFunction<2>::value);
    class_<dealii::DoFHandler<2>, boost::noncopyable>("DoFHander2D", no_init);
    void (dealii::MatrixFree<2>::*init_dof_vector)
        (dealii::parallel::distributed::Vector<double>&, unsigned int) const
        = &dealii::MatrixFree<2>::initialize_dof_vector;
    class_<dealii::MatrixFree<2>, boost::noncopyable>("MatrixFree2D", no_init)
        .def("initialize_dof_vector", init_dof_vector);
    class_<dealii::ConstraintMatrix, boost::noncopyable>("ConstraintMatrix",
        no_init);
    class_<dealii::parallel::distributed::Vector<double> >("MPIVector",
        init<>())
        .def(self += dealii::parallel::distributed::Vector<double>());

    class_<PythonFunction2DWrap, boost::noncopyable>("PyFunction2D")
        .def("value", pure_virtual(&PythonFunction2D::get_value));
    

    /* Basic viscosaur functions.
     */
    class_<vc::Vc>("Vc", init<boost::python::list, boost::python::dict>())
        .def("get_rank", &vc::Vc::get_rank);

    /* Solution object
     */
    class_<vc::Solution<2>, boost::noncopyable>("Solution2D", 
        init<vc::ProblemData<2>&>()[with_custodian_and_ward<1,2>()])
        .def("apply_init_cond", &vc::Solution<2>::apply_init_cond)
        .def("reinit", &vc::Solution<2>::reinit)
        .def("output", &vc::Solution<2>::output)
        .def("start_timestep", &vc::Solution<2>::start_timestep)
        .def("start_refine", &vc::Solution<2>::start_refine)
        .def("post_refine", &vc::Solution<2>::post_refine)
        .def_readwrite("cur_disp", &vc::Solution<2>::cur_disp)
        .def_readwrite("cur_mem", &vc::Solution<2>::cur_mem);

    // double (vc::InvViscosity<2>::*f_value)(const dealii::Point<2>&,
    //                                        const double)= &vc::InvViscosity<2>::value;
    class_<vc::InvViscosity<2>, boost::noncopyable>("InvViscosity2D", no_init);
    class_<vc::InvViscosityTLA<2>, bases<vc::InvViscosity<2> > >(
            "InvViscosityTLA2D", init<dict&>())
        .def("value", &vc::InvViscosityTLA<2>::value)
        .def("value_easy", &vc::InvViscosityTLA<2>::value_easy)
        .def("strs_deriv", &vc::InvViscosityTLA<2>::strs_deriv);

    class_<vc::ProblemData<2>, boost::noncopyable>("ProblemData2D",
            init<dict&>()) 
        .def("start_refine", &vc::ProblemData<2>::start_refine)
        .def("execute_refine", &vc::ProblemData<2>::execute_refine)
        .def("save_mesh", &vc::ProblemData<2>::save_mesh)
        .def_readonly("mem_matrix_free",
                &vc::ProblemData<2>::mem_matrix_free)
        .def_readonly("mem_hanging_node_constraints",
                &vc::ProblemData<2>::mem_hanging_node_constraints)
        .def_readonly("disp_matrix_free",
                &vc::ProblemData<2>::disp_matrix_free);

    class_<vc::OpFactory<2>, boost::noncopyable>("OpFactory2D", no_init);
    class_<vc::MemProjectionOpFactory<2>, boost::noncopyable,
           bases<vc::OpFactory<2> > >(
            "MemProjectionOpFactory2D", init<>());
    class_<vc::MatrixFreeCalculation<2>, boost::noncopyable>(
            "MatrixFreeCalculation2D", 
            init<vc::ProblemData<2>&, dealii::MatrixFree<2>&, 
                 dealii::ConstraintMatrix&>())
        .def_readwrite("op_factory", &vc::MatrixFreeCalculation<2>::op_factory)
        .def("apply_function", &vc::MatrixFreeCalculation<2>::apply_function);

    class_<vc::Stepper<2>, boost::noncopyable>
        ("Stepper2D", init<vc::ProblemData<2>&>())
        .def("step", &vc::Stepper<2>::step);

    /* Expose the analytic solution. 
     * The SlipFnc base class is a slightly different boost expose
     * because it is a abstract base class and cannot be directly used.
     */
    class_<vc::TLA::SlipFnc, boost::noncopyable>("SlipFnc", no_init)
        .def("call", pure_virtual(&vc::TLA::SlipFnc::call));
    // Note the "bases<vc::SlipFnc>" to ensure python understand the 
    // inheritance tree.
    class_<vc::TLA::ConstantSlipFnc, bases<vc::TLA::SlipFnc> >
        ("ConstantSlipFnc", init<double>())
        .def("call", &vc::TLA::ConstantSlipFnc::call);
    class_<vc::TLA::CosSlipFnc, bases<vc::TLA::SlipFnc> >
        ("CosSlipFnc", init<double>())
        .def("call", &vc::TLA::CosSlipFnc::call);

    class_<vc::TLA::TwoLayerAnalytic, boost::noncopyable>("TwoLayerAnalytic", 
            init<double, double, double, double,
                 vc::TLA::SlipFnc&>()[with_custodian_and_ward<1,6>()])
        .def("simple_velocity", &vc::TLA::TwoLayerAnalytic::simple_velocity)
        .def("simple_Szx", &vc::TLA::TwoLayerAnalytic::simple_Szx)
        .def("simple_Szy", &vc::TLA::TwoLayerAnalytic::simple_Szy)
        .def("integral_velocity", &vc::TLA::TwoLayerAnalytic::integral_velocity)
        .def("integral_Szx", &vc::TLA::TwoLayerAnalytic::integral_Szx)
        .def("integral_Szy", &vc::TLA::TwoLayerAnalytic::integral_Szy);

    /* Initial conditions functions.
     * Note the three "> > >" -- these must be separated by a space
     */
    class_<vc::TLA::InitStress<2>, bases<dealii::Function<2> > >
        ("InitStress2D", init<vc::TLA::TwoLayerAnalytic&>()
            [with_custodian_and_ward<1,2>()])
        .def("value", &vc::TLA::InitStress<2>::value);
    class_<vc::TLA::SimpleInitStress<2>, bases<dealii::Function<2> > >
        ("SimpleInitStress2D", init<vc::TLA::TwoLayerAnalytic&>()
            [with_custodian_and_ward<1,2>()])
        .def("value", &vc::TLA::SimpleInitStress<2>::value);
    class_<vc::TLA::ExactVelocity<2>, bases<dealii::Function<2> > >
        ("ExactVelocity2D", init<vc::TLA::TwoLayerAnalytic&>()
            [with_custodian_and_ward<1,2>()])
        .def("value", &vc::TLA::ExactVelocity<2>::value)
        .def("set_t", &vc::TLA::ExactVelocity<2>::set_t);

   class_<vc::TLA::SimpleVelocity<2>, bases<dealii::Function<2> > >
        ("SimpleVelocity2D", init<vc::TLA::TwoLayerAnalytic&>()
            [with_custodian_and_ward<1,2>()])
        .def("value", &vc::TLA::SimpleVelocity<2>::value)
        .def("set_t", &vc::TLA::SimpleVelocity<2>::set_t);
}

