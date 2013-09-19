#include <barista/hamiltonian.h>
#include <iostream>
#include <Eigen/Dense>

using namespace std;

typedef Eigen::MatrixXd matrix_type;

int main(int argc, char *argv[]) {
  std::ifstream  ifs(argv[1]);
  alps::Parameters params(ifs);     

  barista::Hamiltonian<> hamiltonian(params);
  matrix_type matrix(hamiltonian.dimension(), hamiltonian.dimension());
  hamiltonian.fill(matrix);
  //std::cout << matrix << std::endl;

  Eigen::SelfAdjointEigenSolver<matrix_type>  ES(matrix);
  clock_t start, end;
  start = clock();
  //MPI_Barrier(MPI_COMM_WORLD);
  //start = MPI_Wtime();
  Eigen::VectorXd    ev = ES.eigenvalues();
  matrix_type  U = ES.eigenvectors();
  //MPI_Barrier(MPI_COMM_WORLD);
  //end = MPI_Wtime();
  end = clock();
  double time = (double)(end-start)/CLOCKS_PER_SEC;

  std::cout << "ev=" << ev.transpose() << std::endl;
  std::cout << "U=" << U.transpose() << std::endl;  
  cout << "time=" << time << endl;

}
