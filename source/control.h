#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <boost/python.hpp>

namespace viscosaur
{
    /*
     * Taken from 
     * https://wiki.python.org/moin/boost.python/HowTo#Multithreading_Support_for_my_function 
     * Frees the Python global interpreter lock (GIL) so that multithreading
     * can be performed in C++.
     * Currently unused. Consider removing?
     */
    class ScopedGILRelease
    {
    // C & D -------------------------------------------------------------------------------------------
    public:
        inline ScopedGILRelease()
        {
            m_thread_state = PyEval_SaveThread();
        }

        inline ~ScopedGILRelease()
        {
            PyEval_RestoreThread(m_thread_state);
            m_thread_state = NULL;
        }

    private:
        PyThreadState * m_thread_state;
    };


    /* Entry class for viscosaur. A copy must be maintained at all times
     * Currently only controls MPI, PETSc, dealii init and destruct.
     *
     * We must take the command line arguments from python and convert 
     * them to a form that dealii, MPI, and PETSc prefer.
     */
    class Vc
    {
        public:
            Vc(boost::python::list args);
            ~Vc();

        private:
            dealii::Utilities::MPI::MPI_InitFinalize* mpi;
    };
}
