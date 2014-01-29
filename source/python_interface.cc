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
#include "velocity.h"
#include "problem_data.h"
#include "control.h"
#include "stress.h"
#include "stress_op.h"
#include "solution.h"

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>
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
    class_<dealii::parallel::distributed::SolutionTransfer<2,
            dealii::parallel::distributed::Vector<double> >, boost::noncopyable>
                ("SolutionTransfer2D", no_init);

    /* Basic viscosaur functions.
     */
    class_<vc::Vc>("Vc", init<boost::python::list>())
        .def("get_rank", &vc::Vc::get_rank);

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
        .def("simple_Szx", &vc::TwoLayerAnalytic::simple_Szx)
        .def("simple_Szy", &vc::TwoLayerAnalytic::simple_Szy)
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
    class_<vc::SimpleInitSzx<2>, bases<dealii::Function<2> > >
        ("SimpleInitSzx2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::InitSzx<2>::value);
    class_<vc::SimpleInitSzy<2>, bases<dealii::Function<2> > >
        ("SimpleInitSzy2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::InitSzy<2>::value);
    class_<vc::ExactVelocity<2>, bases<dealii::Function<2> > >
        ("ExactVelocity2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::ExactVelocity<2>::value)
        .def("set_t", &vc::ExactVelocity<2>::set_t);
    class_<vc::SimpleVelocity<2>, bases<dealii::Function<2> > >
        ("SimpleVelocity2D", init<vc::TwoLayerAnalytic&>())
        .def("value", &vc::SimpleVelocity<2>::value)
        .def("set_t", &vc::SimpleVelocity<2>::set_t);
    class_<vc::SimpleDeltaVelocity<2>, bases<dealii::Function<2> > >
        ("SimpleDeltaVelocity2D", init<vc::TwoLayerAnalytic&, double>())
        .def("value", &vc::SimpleDeltaVelocity<2>::value)
        .def("set_t", &vc::SimpleDeltaVelocity<2>::set_t);

    /* Solution object
     */
    class_<vc::Solution<2>, boost::noncopyable>("Solution2D", 
        init<vc::ProblemData<2>&>())
        .def("apply_init_cond", &vc::Solution<2>::apply_init_cond)
        .def("output", &vc::Solution<2>::output)
        .def("start_timestep", &vc::Solution<2>::start_timestep)
        .def("start_refine", &vc::Solution<2>::start_refine,
                return_value_policy<manage_new_object>())
        .def("post_refine", &vc::Solution<2>::post_refine);

    /* Expose the Velocity solver. I separate the 2D and 3D because exposing
     * the templating to python is difficult.
     * boost::noncopyable is required, because the copy constructor of some
     * of the private members of Velocity are private
     */ 

    class_<vc::ProblemData<2>, boost::noncopyable>("ProblemData2D",
                                                    init<dict&>()).
        def("start_refine", &vc::ProblemData<2>::start_refine).
        def("execute_refine", &vc::ProblemData<2>::execute_refine);
    class_<vc::ProblemData<3>, boost::noncopyable>("ProblemData3D",
                                                    init<dict&>());

    class_<vc::Velocity<2>, boost::noncopyable>("Velocity2D", 
        init<vc::Solution<2>&, dealii::Function<2>&,
             vc::ProblemData<2>&>())
        .def("step", &vc::Velocity<2>::step)
        .def("update_bc", &vc::Velocity<2>::update_bc);
    class_<vc::Velocity<3>, boost::noncopyable>("Velocity3D", 
        init<vc::Solution<3>&, dealii::Function<3>&,
             vc::ProblemData<3>&>())
        .def("step", &vc::Velocity<3>::step);

    /* Stress updater.
     */
    class_<vc::Stress<2>, boost::noncopyable>("Stress2D", 
        init<vc::Solution<2>&, vc::ProblemData<2>&>())
        .def("tentative_step", &vc::Stress<2>::tentative_step)
        .def("correction_step", &vc::Stress<2>::correction_step);
}

