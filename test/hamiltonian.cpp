/*****************************************************************************
*
* Rokko: Integrated Interface for libraries of eigenvalue decomposition
*
* Copyright (C) 2012-2013 by Tatsuya Sakashita <t-sakashita@issp.u-tokyo.ac.jp>,
*                            Synge Todo <wistaria@comp-phys.org>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
*****************************************************************************/

#include <iostream>
#include <Eigen/Dense>
#include <barista/hamiltonian.h>

#include <rokko/distributed_matrix.hpp>
#include <rokko/localized_matrix.hpp>
#include <rokko/localized_vector.hpp>

//typedef Eigen::MatrixXd matrix_type;

int main(int argc, char *argv[]) {
  //typedef rokko::matrix_row_major matrix_major;
  typedef rokko::matrix_col_major matrix_major;

  if (argc <= 1) {
    std::cerr << "error: " << argv[0] << " solver_name alps_parameter_file_name" << std::endl;
    exit(34);
  }

  rokko::solver solver(solver_name);
  solver.initialize(argc, argv);

  std::string solver_name(argv[1]);
  std::ifstream  ifs(argv[2]);
  alps::Parameters params(ifs);
  barista::Hamiltonian<> hamiltonian(params);
  int dim = hamiltonian.dimension();
  std::cout << "dim=" << dim << std::endl;

  rokko::distributed_matrix<matrix_major> mat(dim, dim, g, solver);
  hamiltonian.fill(mat);
  mat.print();
  rokko::localized_matrix<matrix_major> lmat1(dim, dim);
  rokko::gather(mat, lmat1, root);

  rokko::localized_matrix<rokko::matrix_col_major> lmat2(dim, dim);
  hamiltonian.fill<double>(lmat2);

  std::cout << "lmat1=" << std::endl << lmat1 << std::endl;
  std::cout << "lmat2=" << std::endl << lmat2 << std::endl;

  return 0;
}
