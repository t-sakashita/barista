#include <barista/hamiltonian.h>
#include <iostream>
#include <Eigen/Dense>

typedef Eigen::MatrixXd matrix_type;

int main() {
  alps::Parameters params(std::cin);
  barista::Hamiltonian<> hamiltonian(params);
  matrix_type matrix(hamiltonian.dimension(), hamiltonian.dimension());
  hamiltonian.fill<double>(matrix);
  std::cout << matrix << std::endl;
}
