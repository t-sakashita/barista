#include <barista/hamiltonian.h>
#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>

#define TIMER
#define DETAIL

// Todo: 処理系ごとに分ける。
#define eigen_init __communication_s_MOD_eigen_init
#define eigen_free __communication_s_MOD_eigen_free
#define CSTAB_get_optdim cstab_get_optdim_  // オブジェクトファイルでは小文字に変換する
#define matrix_set matrix_set_
#define matrix_adjust_s matrix_adjust_s_
#define eigen_s eigen_s_
#define ev_test_2D ev_test_2d_

extern "C" void eigen_init(int&);
extern "C" void eigen_free(int&);
extern "C" void CSTAB_get_optdim(int&, int&, int&, int&, int&);
extern "C" void matrix_set(int&, double*);
extern "C" void matrix_adjust_s(int&, double*, double*, int&);
extern "C" void eigen_s(int&, double*, int&, double*, double*, int&, int&, int&);
extern "C" void ev_test_2D(int&, double*, int&, double*, double*, int&);

/*
extern "C" struct
{
  int   nprocs, myrank;
} usempi_;
*/

extern "C" struct
{
  int   my_col, size_of_col, mpi_comm_col,
    my_row, size_of_row, mpi_comm_row,
    p0_      ,q0_      , n_common,
    diag_0, diag_1;
} cycl2d_;

using namespace std;

typedef Eigen::MatrixXd matrix_type;


#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char *argv[])
{
  int rank, size;
  int para_int;

  MPI_Init(&argc, &argv);/* starts MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);/* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &size);/* get number of processes */
 
  printf( "Hello world from process %d of %d\n", rank, size );

  std::ifstream  ifs(argv[1]);
  alps::Parameters params(ifs);     
  params["L"] = 2;
  if (rank == 0) {
    cout << "L=" << params["L"] << endl;
  }

  barista::Hamiltonian<> hamiltonian(params);
  matrix_type matrix(hamiltonian.dimension(), hamiltonian.dimension());        
  hamiltonian.fill<double>(matrix);                            
  //std::cout << matrix << std::endl;

  int n = hamiltonian.dimension();
  double* a = new double[n*n];
  if (a==NULL) {
    cerr << "error: a" << endl;
    return 1;
  }
  for (int i=0; i<n; ++i) { 
    for (int j=0; j<n; ++j)
      a[i*n+j] = matrix(i,j);
  }


  int m = 32;

  para_int = 2;   eigen_init(para_int);
  
  int NPROW = cycl2d_.size_of_col;
  int NPCOL = cycl2d_.size_of_row;
  int nx = ((n-1)/NPROW+1);

  // main.fでは整数型変数の宣言はなかった。
  int i1 = 6, i2 = 16*4, i3 = 16*4*2, nm;
  CSTAB_get_optdim(nx, i1, i2, i3, nm);
  para_int = 0;   eigen_free(para_int);

  int NB  = 64+32;
  int nmz = ((n-1)/NPROW+1);
  nmz = ((nmz-1)/NB+1)*NB+1;
  int nmw = ((n-1)/NPCOL+1);
  nmw = ((nmw-1)/NB+1)*NB+1;

  if (rank == 0) {
    cout << "nmz=" << nmz << endl;
    cout << "nm=" << nm << endl;
    cout << "nmw=" << nmw << endl;
  }

  int larray = std::max(nmz,nm)*nmw;
  if (rank == 0) {
    cout << "larray=" << larray << endl;
  }

  double* b = new double[larray];
  if (b==NULL) {
    cerr << "error: b" << endl;
    return 1;
  }

  double* z = new double[larray];
  if (z==NULL) {
    cerr << "error: z" << endl;
    return 1;
  }
  double* w = new double[n];
  if (w==NULL) {
    cerr << "error: w" << endl;
    return 1;
  }

  //matrix_set(n, a);
  matrix_adjust_s(n, a, b, nm);

  if (rank == 0) {
    cout << "n=" << n << endl;
    //    for (int i=0; i<n; ++i) {
    //      for (int j=0; j<n; ++j)
    //	std::cout << a[i*n+j] << " ";
    //      cout << endl;
    //}
    cout << "nm=" << nm << endl;
  }

  int zero = 0;
  // 1 -> 0
  eigen_s(n, b, nm, w, z, nm, m, zero);
  if (rank == 0) {
    cout << "finished: eigen_s" << endl;
  }

  cout.precision(20);
  if (rank == 0) {
    cout << "w=" << endl;
    for (int i=0; i<n; ++i) {
      cout << w[i] << " ";
    }
    cout << endl;
  }

  delete [] a;
  delete [] b;
  delete [] z;
  delete [] w;

  MPI_Finalize();
  return 0;
}

