cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)


project(barista NONE)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)

# find ALPS Library
find_package(ALPS REQUIRED PATHS ${ALPS_ROOT_DIR} $ENV{ALPS_HOME} NO_SYSTEM_ENVIRONMENT_PATH)
message(STATUS "Found ALPS: ${ALPS_ROOT_DIR} (revision: ${ALPS_VERSION})")
include(${ALPS_USE_FILE})

# enable C and C++ compilers
enable_language(C CXX Fortran)

# find Eigen3 Library
find_package(Eigen3)

# find Eigen_s
message(STATUS "Display: EIGEN_S_LIB= ${EIGEN_S_LIB}")
message(STATUS "Display: EIGEN_SX_LIB= ${EIGEN_SX_LIB}")
message(STATUS "Display: ALPS_LIB=${ALPS_LIBRARIES}")


# find PETSc/SLEPc
# Normally PETSc is built with MPI, if not, use CC=mpicc, etc, so we need to specify CXX as an option.
set(PETSC_DIR "/home/sakashita/lib/petsc-3.3-p6_installed/")
set(SLEPC_DIR "/home/sakashita/lib/petsc-3.3-p6_installed/")

find_package(PETSc2 COMPONENTS CXX)
message(STATUS "Display_main PETSC_LIBRARIES=${PETSC_LIBRARIES}.")
# Added by Sakashita
set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDES})
message(STATUS "Display_main PETSC_INCLUDE_DIRS=${PETSC_INCLUDE_DIRS}.")
find_package(SLEPc2)
message(STATUS "Display_main CMAKE_REQUIRED_INCLUDES=${CMAKE_REQUIRED_INCLUDES}.")
message(STATUS "Display_main CMAKE_REQUIRED_LIBRARIES=${CMAKE_REQUIRED_LIBRARIES}.")
# trial
#set(PETSC_INCLUDE_DIRS /home/sakashita/development/barista/rokko/script/install_build/include)
#set(SLEPC_INCLUDE_DIRS /home/sakashita/development/barista/rokko/script/install_build/include)
#set(PETSC_LIBRARIES /home/sakashita/development/barista/rokko/script/install_build/lib)
#set(SLEPC_LIBRARIES /home/sakashita/development/barista/rokko/script/install_build/lib) 


# Get Trilinos as one entity
#set(TRILINOS_PREFIX /home/sakashita/lib/trilinos-11.0.3-Source/install_build)

if( DEFINED TRILINOS_PREFIX )

SET(PACKAGE_LIST Anasazi)
#SET(PACKAGE_LIST Anasazi Galeri)  # some Eigenvalue Solver needs Galeri

foreach(package_name ${PACKAGE_LIST})
  INCLUDE(${TRILINOS_PREFIX}/lib/cmake/${package_name}/${package_name}Config.cmake)
  # Echo trilinos build info just for fun
  MESSAGE("\nFound ${package_name}!  Here are the details: ")
  MESSAGE("   ${package_name}_DIR = ${${package_name}_DIR}")
  MESSAGE("   ${package_name}_VERSION = ${${package_name}_VERSION}")
  MESSAGE("   ${package_name}_PACKAGE_LIST = ${${package_name}_PACKAGE_LIST}")
  MESSAGE("   ${package_name}_LIBRARIES = ${${package_name}_LIBRARIES}")
  MESSAGE("   ${package_name}_INCLUDE_DIRS = ${${package_name}_INCLUDE_DIRS}")
  MESSAGE("   ${package_name}_LIBRARY_DIRS = ${${package_name}_LIBRARY_DIRS}")
  MESSAGE("   ${package_name}_TPL_LIST = ${${package_name}_TPL_LIST}")
  MESSAGE("   ${package_name}_TPL_INCLUDE_DIRS = ${${package_name}_TPL_INCLUDE_DIRS}")
  MESSAGE("   ${package_name}_TPL_LIBRARIES = ${${package_name}_TPL_LIBRARIES}")
  MESSAGE("   ${package_name}_TPL_LIBRARY_DIRS = ${${package_name}_TPL_LIBRARY_DIRS}")
  MESSAGE("   ${package_name}_BUILD_SHARED_LIBS = ${${package_name}_BUILD_SHARED_LIBS}")
  MESSAGE("End of ${package_name} details\n")

  # Make sure to use same compilers and flags as ${package_name}
  SET(CMAKE_CXX_COMPILER ${${package_name}_CXX_COMPILER} )
  SET(CMAKE_C_COMPILER ${${package_name}_C_COMPILER} )
  SET(CMAKE_Fortran_COMPILER ${${package_name}_Fortran_COMPILER} )

  SET(CMAKE_CXX_FLAGS  "${${package_name}_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
  SET(CMAKE_C_FLAGS  "${${package_name}_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
  SET(CMAKE_Fortran_FLAGS  "${${package_name}_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")
endforeach(package_name)

foreach(package_name ${PACKAGE_LIST})
  INCLUDE_DIRECTORIES(${${package_name}_INCLUDE_DIRS} ${${package_name}_TPL_INCLUDE_DIRS})
  LINK_DIRECTORIES(${${package_name}_LIBRARY_DIRS} ${${package_name}_TPL_LIBRARY_DIRS})
  SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} ${${package_name}_LIBRARIES} ${${package_name}_TPL_LIBRARIES})
endforeach(package_name)

endif()


# includes
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(BEFORE ${EIGEN3_INCLUDE_DIR})

# directories
add_subdirectory(sample)

# tests
enable_testing()
