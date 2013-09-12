cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)


project(barista NONE)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)

# include variables for Rokko
include(${ROKKO_DIR}/config/UseRokko.cmake)

# find ALPS Library
find_package(ALPS REQUIRED PATHS ${ALPS_ROOT_DIR} $ENV{ALPS_HOME} NO_SYSTEM_ENVIRONMENT_PATH)
message(STATUS "Found ALPS: ${ALPS_ROOT_DIR} (revision: ${ALPS_VERSION})")
include(${ALPS_USE_FILE})

# enable C and C++ compilers
enable_language(C CXX Fortran)

# find Eigen3 Library
#find_package(Eigen3)

# find Eigen_s
message(STATUS "Display: EIGEN_S_LIB= ${EIGEN_S_LIB}")
message(STATUS "Display: EIGEN_SX_LIB= ${EIGEN_SX_LIB}")
message(STATUS "Display: ALPS_LIB=${ALPS_LIBRARIES}")


message(STATUS "Display_main PETSC_LIBRARIES=${PETSC_LIBRARIES}.")
message(STATUS "Display_main PETSC_INCLUDE_DIRS=${PETSC_INCLUDE_DIRS}.")
message(STATUS "Display_main CMAKE_REQUIRED_INCLUDES=${CMAKE_REQUIRED_INCLUDES}.")
message(STATUS "Display_main CMAKE_REQUIRED_LIBRARIES=${CMAKE_REQUIRED_LIBRARIES}.")

#Elemental
message(STATUS  "Display ELEM_INC:  ${ELEM_INC}")
message(STATUS  "Display ELEM_LIB:  ${ELEM_LIB}")
message(STATUS  "Display ELEM_LIBS:  ${ELEM_LIBS}")

# includes
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${ROKKO_DIR}/include)
list(APPEND ROKKO_INCLUDE_DIR ${BOOST_INCLUDE_DIR})
list(APPEND ROKKO_INCLUDE_DIR ${ELEMENTAL_INCLUDE_DIRS})
list(APPEND ROKKO_INCLUDE_DIR ${PETSC_INCLUDE_DIRS})
list(APPEND ROKKO_INCLUDE_DIR ${SLEPC_INCLUDE_DIR})
list(APPEND ROKKO_INCLUDE_DIR ${Anasazi_INCLUDE_DIRS})
include_directories(${PETSC_INCLUDE_DIRS})
include_directories(${SLEPC_INCLUDE_DIR})

include_directories(${ROKKO_INCLUDE_DIR})
link_directories(${ROKKO_DIR}/lib)
#include_directories(BEFORE ${EIGEN3_INCLUDE_DIR})

# directories
add_subdirectory(sample)
add_subdirectory(sample_elemental)

# tests
enable_testing()
