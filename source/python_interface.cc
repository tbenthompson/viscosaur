
#include <Python.h>
#if PY_VERSION_HEX >= 0x03000000
    // The code should never get here, but if it does,
    // we should quit, because the conflicting interpreter and include
    // will cause weird problems.
    #error "Python 3?!" 
#endif
#include <boost/python.hpp>

#include "analytic.h"
namespace vc = viscosaur;

BOOST_PYTHON_MODULE(viscosaur)
{
    using namespace boost::python;

    class_<vc::SlipFnc, boost::noncopyable>("SlipFnc", no_init)
        .def("call", pure_virtual(&vc::SlipFnc::call));

    class_<vc::ConstantSlipFnc, bases<vc::SlipFnc> >("ConstantSlipFnc", init<double>())
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
}

// Test with vc.TwoLayerAnalytic(1.0, 10000.0, 3.0e10, 5.0e19, vc.CosSlipFnc(10000)) 
