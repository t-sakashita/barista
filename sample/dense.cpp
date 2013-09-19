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

#include <rokko/serial_solver.hpp>
#include <rokko/localized_matrix.hpp>
#include <rokko/localized_vector.hpp>

#include <rokko/utility/sort_eigenpairs.hpp>

//#include <boost/lexical_cast.hpp>

typedef Eigen::MatrixXd matrix_type;

int main(int argc, char *argv[]) {
  //typedef rokko::matrix_row_major matrix_major;
  typedef rokko::matrix_col_major matrix_major;

  if (argc <= 2) {
    std::cerr << "error: " << argv[0] << " solver_name alps_parameter_file_name" << std::endl;
    exit(34);
  }

  std::string solver_name(argv[1]);
  std::ifstream  ifs(argv[2]);
  alps::Parameters params(ifs);
  barista::Hamiltonian<> hamiltonian(params);
  int dim = hamiltonian.dimension();
  std::cout << "dim=" << dim << std::endl;

  rokko::serial_solver solver(solver_name);
  solver.initialize(argc, argv);

  rokko::localized_matrix<rokko::matrix_col_major> mat(dim, dim);
  hamiltonian.fill(mat);

  std::cout << mat << std::endl;

  rokko::localized_vector w(dim);
  rokko::localized_matrix<rokko::matrix_col_major> eigvec(dim, dim);

  try {
    solver.diagonalize(mat, w, eigvec);
  }
  catch (const char *e) {
    std::cout << "Exception : " << e << std::endl;
    exit(22);
  }

  std::cout << "eigvec:" << eigvec << std::endl;
  
  std::cout.precision(20);
  rokko::localized_vector eigval_sorted(dim);
  rokko::localized_matrix<rokko::matrix_col_major> eigvec_sorted(dim, dim);
  rokko::sort_eigenpairs(w, eigvec, eigval_sorted, eigvec_sorted);
  std::cout << "Computed Eigenvalues= " << eigval_sorted.transpose() << std::endl;
  
  std::cout.precision(3);
  std::cout << "Check the orthogonality of Eigenvectors:" << std::endl
            << eigvec_sorted * eigvec_sorted.transpose() << std::endl;   // Is it equal to indentity matrix?
  //<< eigvecs.transpose() * eigvecs << std::endl;   // Is it equal to indentity matrix?
  
  hamiltonian.fill(mat);
  std::cout << "residual := A x - lambda x = " << std::endl
         << mat * eigvec_sorted.col(1)  -  eigval_sorted(1) * eigvec_sorted.col(1) << std::endl;
  std::cout << "Are all the following values equal to some eigenvalue = " << std::endl
            << (mat * eigvec_sorted.col(0)).array() / eigvec_sorted.col(0).array() << std::endl;
  //cout << "lmat=" << std::endl << lmat << std::endl;

  solver.finalize();
  return 0;
}
