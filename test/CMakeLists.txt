#
# Rokko: Integrated Interface for libraries of eigenvalue decomposition
#
# Copyright (C) 2012-2013 by Tatsuya Sakashita <t-sakashita@issp.u-tokyo.ac.jp>,
#                            Synge Todo <wistaria@comp-phys.org>,
#               2013-2013    Ryo IGARASHI <rigarash@issp.u-tokyo.ac.jp>
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

# Tests using Boost.test
# Check and build Boost.Test Unit Test Framework
find_package(Boost 1.48.0 COMPONENTS unit_test_framework)
if (Boost_FOUND)
  add_definitions(-DBOOST_TEST_DYN_LINK)
endif (Boost_FOUND)


# multi process tests
set(TESTS hamiltonian_mpi)
foreach(name ${TESTS})
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} rokko ${ELEMENTAL_LIBRARIES} ${SCALAPACK_LIB} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
  add_test(${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/${name})
endforeach(name)
unset(TESTS)

# 2 & 4 processes

# add_rokko_test tests

