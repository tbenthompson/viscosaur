##
#  CMake script for the step-40 tutorial program:
##

# Set basic CMake variables
PROJECT(viscosaur)
set(CMAKE_MAIN_DIR ".")
set(CMAKE_SOURCE_DIR ${CMAKE_MAIN_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
    "${CMAKE_MAIN_DIR}/cmake/Modules/")
set(CMAKE_BUILD_TYPE Debug)

#
# Are all dependencies fullfilled?
#
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(PythonInterp 2.7 REQUIRED)
FIND_PACKAGE(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)
IF(NOT PYTHONLIBS_FOUND)
    MESSAGE(FATAL_ERROR "\n"
        "Could not find the python libraries. Try setting"
        "PYTHON_LIBRARY OR PYTHON_INCLUDE_DIR"
        )
ENDIF()

MESSAGE(${CMAKE_MODULE_PATH})
FIND_PACKAGE(GSL REQUIRED)
IF(NOT GSL_FOUND)
    MESSAGE(FATAL_ERROR "\n"
        "Could not find the GSL (GNU Scientific Library)."
        )
ENDIF()

FIND_PACKAGE(Boost 1.53 COMPONENTS python REQUIRED)
IF(NOT Boost_FOUND)
    MESSAGE(FATAL_ERROR "\n"
        "Could not find Boost. Boost is required for deal.II, so "
        "it is highly likely that you just need to specify BOOST_ROOT"
        )
ENDIF()

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to 
    cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_PETSC OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "
    Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
        DEAL_II_WITH_PETSC = ON
        DEAL_II_WITH_P4EST = ON
    One or all of these are OFF in your installation but are required for 
    this tutorial step."
    )
ENDIF()
# DEPENDENCY CHECKS DONE

DEAL_II_INITIALIZE_CACHED_VARIABLES()

INCLUDE_DIRECTORIES(${GSL_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_PATH})

FILE(GLOB sources ${CMAKE_SOURCE_DIR}/source/*.cc)
ADD_LIBRARY(viscosaur SHARED ${sources})
DEAL_II_SETUP_TARGET(viscosaur)
TARGET_LINK_LIBRARIES(viscosaur ${GSL_LIBRARIES} 
    ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
SET_TARGET_PROPERTIES(viscosaur PROPERTIES PREFIX "")

