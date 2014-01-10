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
#include "control.h"dealii::Utilities::MPI::MPI_InitFinalize
namespace vc = viscosaur;

BOOST_PYTHON_MODULE(viscosaur)
{
    using namespace boost::python;

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

    class_<vc::TwoLayerAnalytic>("TwoLayerAnalytic", 
            init<double, double, double, double,
                 vc::SlipFnc&>())
        .def("simple_velocity", &vc::TwoLayerAnalytic::simple_velocity)
        .def("simple_stress", &vc::TwoLayerAnalytic::simple_stress)
        .def("integral_velocity", &vc::TwoLayerAnalytic::integral_velocity)
        .def("integral_stress", &vc::TwoLayerAnalytic::integral_stress);

    /* Expose the Poisson solver. I separate the 2D and 3D because exposing
     * the templating to python is difficult.
     * boost::noncopyable is required, because the copy constructor of some
     * of the private members of Poisson are private
     */ 
    class_<vc::Poisson<2>, boost::noncopyable>("Poisson2D", init<>())
        .def("run", &vc::Poisson<2>::run);
    class_<vc::Poisson<3>, boost::noncopyable>("Poisson3D", init<>())
        .def("run", &vc::Poisson<3>::run);
}

// Test with vc.TwoLayerAnalytic(1.0, 10000.0, 3.0e10, 5.0e19, vc.CosSlipFnc(10000)) 
