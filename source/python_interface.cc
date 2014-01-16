#include <Python.h>
#if PY_VERSION_HEX >= 0x03000000
    // The code should never get here, but if it does,
    // we should quit, because the conflicting interpreter and include
    // will cause weird problems.
    #error "Python 3?!" 
#endif
#include <boost/python.hpp>
#include <boost/array.hpp>

#include "analytic.h"
#include "poisson.h"
#include "problem_data.h"
#include "control.h"
#include "rhs.h"

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
namespace vc = viscosaur;


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
    class_<dealii::Point<3> >("Point3D", init<double, double, double>());
    class_<dealii::Function<2>, boost::noncopyable>("Function2D", no_init)
        .def("value", pure_virtual(&dealii::Function<2>::value));
    class_<dealii::Function<3>, boost::noncopyable>("Function3D", no_init)
        .def("value", pure_virtual(&dealii::Function<3>::value));
    class_<dealii::PETScWrappers::MPI::Vector>("PETScWrappers", no_init);
    class_<dealii::DoFHandler<2>, boost::noncopyable>("DoFHander2D", no_init);

    /* Basic viscosaur functions.
     */
    class_<vc::Vc>("Vc", init<boost::python::list>());

    /* Expose the analytic solution. 
     * The SlipFnc base class is a slightly different boost expose
     * because it is a abstract base class and cannot be directly used.
     */
    class_<vc::SlipFnc, boost::noncopyable>("SlipFnc", no_init)
        .def("call", pure_virtual(&vc::SlipFnc::call));
    // Note the "bases<vc::SlipFnc>" to ensure python understand the 
    // inheritance tree.
    class_<vc::ConstantSlipFnc, bases<vc::SlipFnc> >("ConstantSlipFnc", 
            init<double>())
        .def("call", &vc::ConstantSlipFnc::call);
    class_<vc::CosSlipFnc, bases<vc::SlipFnc> >("CosSlipFnc", init<double>())
        .def("call", &vc::CosSlipFnc::call);

    class_<vc::TwoLayerAnalytic, boost::noncopyable>("TwoLayerAnalytic", 
            init<double, double, double, double,
                 vc::SlipFnc&>())
        .def("simple_velocity", &vc::TwoLayerAnalytic::simple_velocity)
        .def("simple_stress", &vc::TwoLayerAnalytic::simple_stress)
        .def("integral_velocity", &vc::TwoLayerAnalytic::integral_velocity)
        .def("integral_Szx", &vc::TwoLayerAnalytic::integral_Szx)
        .def("integral_Szy", &vc::TwoLayerAnalytic::integral_Szy);

    /* Initial conditions functions.
     * Note the three "> > >" -- these must be separated by a space
     */
    class_<vc::InitSzx<2>, bases<dealii::Function<2> > >
        ("InitSzx2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::InitSzx<2>::value);
    class_<vc::InitSzy<2>, bases<dealii::Function<2> > >
        ("InitSzy2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::InitSzy<2>::value);
    class_<vc::Velocity<2>, bases<dealii::Function<2> > >
        ("Velocity2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::Velocity<2>::value)
        .def("set_t", &vc::Velocity<2>::set_t);

    /* Expose the Poisson solver. I separate the 2D and 3D because exposing
     * the templating to python is difficult.
     * boost::noncopyable is required, because the copy constructor of some
     * of the private members of Poisson are private
     */ 
    class_<vc::PoissonRHS<2>, boost::noncopyable>("PoissonRHS2D", no_init);
        // .def("value", pure_virtual(&vc::PoissonRHS<2>::value));
    class_<vc::PoissonRHS<3>, boost::noncopyable>("PoissonRHS3D", no_init);
        // .def("value", pure_virtual(&vc::PoissonRHS<3>::value));
    class_<vc::SinRHS<2>, bases<vc::PoissonRHS<2> > >("SinRHS2D", init<>());
        // .def("value", &vc::SinRHS<2>::value);

    class_<vc::OneStepRHS<2>, bases<vc::PoissonRHS<2> >, boost::noncopyable>("OneStepRHS2D", 
        init<dealii::Function<2>&, dealii::Function<2>&,
             vc::ProblemData<2>& >());
        // .def("value", &vc::OneStepRHS<2>::value);

    class_<vc::ProblemData<2>, boost::noncopyable>("ProblemData2D",
                                                    init<dict&>());
    class_<vc::ProblemData<3>, boost::noncopyable>("ProblemData3D",
                                                    init<dict&>());
    class_<vc::Poisson<2>, boost::noncopyable>("Poisson2D", 
                                               init<vc::ProblemData<2>&>())
        .def("run", &vc::Poisson<2>::run)
        .def("get_dof_handler", &vc::Poisson<2>::get_dof_handler,
                return_value_policy<reference_existing_object>());
    class_<vc::Poisson<3>, boost::noncopyable>("Poisson3D", 
                                                init<vc::ProblemData<3>&>())
        .def("run", &vc::Poisson<3>::run);
}

