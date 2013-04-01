cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)


project(sample_elemental NONE)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)

# enable C and C++ compilers
enable_language(C CXX Fortran)

find_package ( MPI )

SET(CMAKE_CXX_COMPILER "mpicxx" )
SET(CMAKE_C_COMPILER "mpicc" )
SET(CMAKE_Fortran_COMPILER "mpif90" )

# find Eigen3 Library
find_package(Eigen3)

# find Elemental
set(ELEMENTAL_DIR "/home/sakashita/elemental-0.77_installed")
find_package(Elemental)
#message(STATUS  "Display ELEM_INC:  ${ELEM_INC}")
#message(STATUS  "Display ELEM_LIB:  ${ELEM_LIB}")
#message(STATUS  "Display ELEM_LIBS:  ${ELEM_LIBS}")


message(STATUS  "Display ELEMENTAL_INCLUDE_DIRS:  ${ELEMENTAL_INCLUDE_DIRS}")
message(STATUS  "Display ELEMENTAL_LIBRARIES:  ${ELEMENTAL_LIBRARIES}")

message(STATUS "Display_main CMAKE_REQUIRED_INCLUDES=${CMAKE_REQUIRED_INCLUDES}.")
message(STATUS "Display_main CMAKE_REQUIRED_LIBRARIES=${CMAKE_REQUIRED_LIBRARIES}.")

# directories
add_subdirectory(sample_elemental)
