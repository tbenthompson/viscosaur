#include "poisson.h"

int main(int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace viscosaur;


        std::vector<std::vector<double> > v;
        double x_min = 1.0;
        double x_max = 10000.0;
        double y_min = 0.0;
        double y_max = 20000.0;
        double fault_slip = 1.0;
        double fault_depth = 10000.0;
        double shear_modulus = 30.0e9;
        double viscosity = 5.0e19;
        
        ConstantSlipFnc const_slip(fault_depth);
        TwoLayerAnalytic* tla = new TwoLayerAnalytic(fault_slip,
                fault_depth, shear_modulus, viscosity, const_slip);
        // boost::array<double, 2> fff = abc->simple_stress(1.0, 10000.0);
        // std::cout << def << "   " << abc->simple_velocity(1000.0, 10000.0, 1.0) << std::endl;
        // std::cout << fff[0] << "    " << fff[1] << std::endl;
        // fff = abc->integral_stress(1.0, 10000.0);
        // std::cout << fff[0] << "    " << fff[1] << std::endl;
        boost::array<boost::array<double, 50>, 50> vels;
        for (int i = 0; i < 50; i++) 
        {
            for (int j = 0; j < 50; j++) 
            {
                 vels[i][j] = tla->integral_velocity(10.0 + 5000.0 * i, 0.0 + 5000 * j, 0.0);
                 std::cout << vels[i][j] << std::endl;
            }
        }

        delete tla;
        // return 1;

        // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        // deallog.depth_console (0);

        // {
        //     Poisson<2> laplace_problem_2d;
        //     laplace_problem_2d.run ();
        // }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }

    return 0;
}
