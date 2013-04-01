#include <barista/hamiltonian.h>
#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>

#define TIMER
#define DETAIL

// Todo: 処理系ごとに分ける。
//#define eigen_init __communication_sx_MOD_eigen_init                                                                                                                 
//#define eigen_free __communication_sx_MOD_eigen_free                                                                                                                  
#define eigen_init communication_sx_mp_eigen_init_
#define eigen_free communication_sx_mp_eigen_free_

#define CSTAB_get_optdim cstab_get_optdim_  // オブジェクトファイルでは小文字に変換する
#define matrix_set matrix_set_
#define matrix_adjust_sx matrix_adjust_sx_
#define eigen_sx eigen_sx_
#define ev_test_2D ev_test_2d_

extern "C" void eigen_init(int&);
extern "C" void eigen_free(int&);
extern "C" void CSTAB_get_optdim(int&, int&, int&, int&, int&);
extern "C" void matrix_set(int&, double*);
extern "C" void matrix_adjust_sx(int&, double*, double*, int&);
extern "C" void eigen_sx(int&, double*, int&, double*, double*, double*, double*, int&, int&, int&);
extern "C" void ev_test_2D(int&, double*, int&, double*, double*, int&);

extern "C" struct
{
  int   nprocs, myrank;
} usempi_;

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
  params["L"] = 7;
  cout << "L=" << params["L"] << endl;
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
  int i1 = 6, i2 = 16*2, i3 = 16*4, nm;
  CSTAB_get_optdim(nx, i1, i2, i3, nm);
  para_int = 0;   eigen_free(para_int);

  int NB  = 64+32;
  int nmz = ((n-1)/NPROW+1);
  nmz = ((nmz-1)/NB+1)*NB+1;
  int nmw = ((n-1)/NPCOL+1);
  nmw = ((nmw-1)/NB+1)*NB+1;
  int nme = ((n-1)/2+1)*2;

  int nh = (n-1)/4+1;
  i1 = 4;  //, i2 = 16*2, i3 = 16*4;
  int nnh;
  CSTAB_get_optdim(nh, i1, i2, i3, nnh);
  nnh = 4 * nnh;

  cout << "nmz=" << nmz << endl;
  cout << "nm=" << nm << endl;
  cout << "nmw=" << nmw << endl;
  cout << "nnh=" << nnh << endl;

  int n1x = ((n-1)/usempi_.nprocs+1);
  int larray = max(max( max(nmz,nm) ,nnh)*nmw, n*n1x);

  cout << "larray=" << larray << endl;

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
  double* d = new double[nme];
  if (d==NULL) {
    cerr << "error: d" << endl;
    return 1;
  }
  double* e = new double[2*nme];
  if (e==NULL) {
    cerr << "error: e" << endl;
    return 1;
  }
  double* w = new double[nme];
  if (w==NULL) {
    cerr << "error: w" << endl;
    return 1;
  }

  //matrix_set(n, a);
  matrix_adjust_sx(n, a, b, nm);

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
  eigen_sx(n, b, nm, w, z, d, e, nme, m, zero);

  int* q = new int[n];  
  if (rank == 0) {
    // 固有値を（絶対値のではなく）昇順に並べる
    if (q==NULL) {
      cerr << "error: q" << endl;
      return 1;
    }

    double emax;
    for (int i=0; i<n; ++i) q[i] = i;
    for (int k=0; k<n; ++k) {
      emax = w[q[k]];
      for (int i=k+1; i<n; ++i) {
	if (emax > w[q[i]]) {       // 昇順になっていないとき、交換      
	  emax = w[q[i]];
	  int qq = q[k];
	  q[k] = q[i];
	  q[i] = qq;
	}
      }
    }

    cout.precision(20);
    if (rank == 0) {
      cout << "w=" << endl;
      for (int i=0; i<n; ++i) {
	cout << w[q[i]] << " ";
      }
      cout << endl;
    }
  }

  delete [] a;
  delete [] b;
  delete [] z;
  delete [] w;
  delete [] q;

  MPI_Finalize();
  return 0;
}

