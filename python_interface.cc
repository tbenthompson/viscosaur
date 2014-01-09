
#include <Python.h>
#if PY_VERSION_HEX >= 0x03000000
    // The code should never get here, but if it does,
    // we should quit, because the conflicting interpreter and include
    // will cause weird problems.
    #error "Python 3?!" 
#endif
#include <boost/python.hpp>
#include "elastic_stress.h"
namespace vc = viscosaur;

BOOST_PYTHON_MODULE(viscosaur)
{
    using namespace boost::python;
    // class_<vc::TwoLayerAnalytic>("TwoLayerAnalytic", 
    //         init<double, double, double, double,
    //                  double (*)(double)>())
    //     .def("simple_velocity", &vc::TwoLayerAnalytic::simple_velocity)
    //     .def("simple_stress", &vc::TwoLayerAnalytic::simple_stress)
    //     .def("integral_velocity", &vc::TwoLayerAnalytic::integral_velocity)
    //     .def("integral_stress", &vc::TwoLayerAnalytic::integral_stress);
}
