#include <string>
#include "control.h"
namespace bp = boost::python;

namespace viscosaur
{
    Vc::Vc(bp::list args)
    {
        int argc = bp::len(args);
        char **argv = new char*[argc];
        for (int i = 0; i < bp::len(args); i++)
        {
            argv[i] = bp::extract<char*>(args[i]);
            // std::cout << argv[i] << std::endl;
        }
        this->mpi = new dealii::Utilities::MPI::MPI_InitFinalize(argc, argv, 1);
        dealii::deallog.depth_console (0);
    }

    Vc::~Vc()
    {
        delete this->mpi;
    }
}
