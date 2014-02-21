#include "control.h"

#include <string>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>

namespace viscosaur
{
    namespace bp = boost::python;
    Vc::Vc(bp::list args, bp::dict params)
    {
        int argc = bp::len(args);
        char **argv = new char*[argc];
        for (int i = 0; i < bp::len(args); i++)
        {
            argv[i] = bp::extract<char*>(args[i]);
            // std::cout << argv[i] << std::endl;
        }
        int num_threads = bp::extract<int>(params["num_threads"]);
        this->mpi = new dealii::Utilities::MPI::MPI_InitFinalize(argc, argv, 1);
        dealii::deallog.depth_console (0);
    }

    int
    Vc::get_rank()
    {
        return dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    }

    Vc::~Vc()
    {
        // delete this->mpi;
    }
}
