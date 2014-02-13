#ifndef __viscosaur_linear_algebra_h
#define __viscosaur_linear_algebra_h
// This include just defines which linear algebra package to use.
// It should be included after all the other relevant includes and will fail if
// the proper linear algbra package has not been included.
//
// On my current machine, Trilinos linear algebra seems to be
// about twice as fast as PETSc. This is probably an artifact of some 
// configurations, so flip this flag to try out PETSc (assuming it's
// installed and deal.II is configured to use it). 
// However, the python bindings are not set up for PETSc, so new 
// bindings will need to be made. On the other hand, petsc4py
// might be able to do the job.
// I've now flipped this to using PETSc because I get a weird loss of 
// precision error with the Trilinos AztecOO solvers.
#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
    using namespace dealii::LinearAlgebraPETSc;
#else
    using namespace dealii::LinearAlgebraTrilinos;
#endif
}
#endif
