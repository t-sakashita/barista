#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}

class Hamiltonian {
public:
  template<typename Scalar,
       int RowsAtCompileTime,
       int ColsAtCompileTime,
       int Options,
       int MaxRowsAtCompileTime,
       int MaxColsAtCompileTime> 
  void fill(Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxRowsAtCompileTime>& m) const { }
};

