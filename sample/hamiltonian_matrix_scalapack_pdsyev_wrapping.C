#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>

#include "mpi.h"

// Eigen3に関するヘッダファイル
#include <Eigen/Dense>

typedef Eigen::MatrixXd matrix_type;

using namespace std;
using namespace Eigen;

extern "C" {
  /* BLACS */
  void blacs_pinfo_( int& mypnum, int& nprocs); 
  void blacs_get_( const int& context, const int& request, int& value ); 
  void blacs_gridinfo_(const int& ictxt, int& nprow, int& npcol, int& myrow, int& mycol);
  void blacs_gridinit_(const int& ictxt, char* order, int& nprow, int& npcol );
  void blacs_gridexit_(const int& ictxt);
  void blacs_exit_(const int& cont);
  void blacs_barrier_(const int& ictxt, const char* score);
  /* ScaLAPACK */
  void sl_init_(int, int, int);
  void descinit_(int* desc, const int& m, const int& n, const int& mb, const int& nb,
                 const int& irsrc, const int& icsrc, const int& ixtxt, const int& lld, int& info);
  void pdelset_(double* A, const int& ia, const int& ja, const int* descA, const double& alpha);
  void pdelget_(char* scope, char* top, double& alpha, const double* A, const int& ia, const int& ja, const int* descA);
  int numroc_(const int& n, const int& nb, const int& iproc, const int& isrcproc, const int& nprocs);
  void pdsyev_(char* jobz, char* uplo, const int& n,
               double* a, const int& ia, const int& ja, const int* descA,
               double* w, double* z, const int& iz, const int& jz, const int* descZ,
               double* work, const int& lwork, int& info);

  void pdsyevr_(const char* jobz, const char* uplo, const int& n,
                double* A, const int& ia, const int& ja, const int* descA,
                const double& vl, const double& vu, const int& il, const int& iu,
                const int& m, const int& nz, double* w,
                double* Z, const int& iz, const int& jz, const int* descZ,
                double* work, const int& lwork, int* iwork, const int& liwork, int& info);
  void pdsyevd_(const char* jobz, const char* uplo, const int& n,
                double* A, const int& ia, const int& ja, const int* descA,
                const double& vl, const double& vu, const int& il, const int& iu,
                const int& m, const int& nz, double* w,
                double* Z, const int& iz, const int& jz, const int* descZ,
                double* work, const int& lwork, int* iwork, const int& liwork, int& info);

  void pdlaprnt_(const int& m, const int& n, const double* A, const int& ia, const int& ja, const int* descA, const int& irprnt, const int& icprnt, char* cmatnm, const int& nout, double* work);
}

//#define A_global(i,j) A_global[i*M+j]
#define A_global(i,j) A_global[j*dim+i]


void Initialize(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);/* starts MPI */
}

void Finalize()
{
  blacs_gridexit_( 0 );
  blacs_exit_( 1 );
  MPI_Finalize();
  exit(0);
}

class Grid
{
public:
  Grid(MPI_Comm& comm)
  {
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    const int ZERO=0, ONE=1;
    long MINUS_ONE = -1;
    blacs_pinfo_( myrank, nprocs );
    blacs_get_( MINUS_ONE, ZERO, ictxt );
    
    npcol = int(sqrt(nprocs+0.5));
    while (1) {
      if ( npcol == 1 ) break;
      if ( (nprocs % npcol) == 0 ) break;
      npcol = npcol-1;
    }                                                                                                                                                                           
    nprow = nprocs / npcol;
    blacs_gridinit_( ictxt, "Row", nprow, npcol );
    blacs_gridinfo_( ictxt, nprow, npcol, myrow, mycol );
    
    if (myrank == 0) {
      cout << "nprow=" << nprow << "  npcol=" << npcol << endl;
    }
  }

  int myrank, nprocs;
  int myrow, mycol;
  int ictxt, nprow, npcol;
  int info;

};


class Distributed_Matrix
{
public:
  Distributed_Matrix(const int& m_global, const int& n_global, Grid& g) : m_global(m_global), n_global(n_global), g(g)
  {
    // ローカル行列の形状を指定
    mb = m_global / g.nprow;
    if (mb == 0) mb = 1;
    //mb = 10;        
    nb = n_global / g.npcol;
    if (nb == 0) nb = 1;
    //nb = 10;

    const int ZERO=0, ONE=1;
    
    cout << "nb=" << nb << endl;
    m_local = numroc_( m_global, mb, g.myrow, ZERO, g.nprow );
    //if (myrank == 0)
    cout << "mA=" << m_local << "  nprow=" << g.nprow << endl;
    n_local = numroc_( n_global, nb, g.mycol, ZERO, g.npcol );
    //if (myrank == 0)
    cout<< "nA=" << n_local << "  npcol=" << g.npcol << endl;
    int lld = m_local;
    if (lld == 0) lld = 1;
    cout << "lld=" << lld << endl;
    descinit_(desc, m_global, n_global, nb, nb, ZERO, ZERO, g.ictxt, lld, info);
    if (info) {
      cerr << "error " << info << " at descinit function of descA " << "mA=" << m_local << "  nA=" << n_local << "  lld=" << lld << "." << endl;
      exit(1);
    }

    array = new double[m_local * n_local];
    if (array == NULL) {
      cerr << "failed to allocate array." << endl;
      exit(1);
    }
  }

  ~Distributed_Matrix()
  {
   delete[] array;
  }

  int Symmetric_EigenSolver(Distributed_Matrix& A, double* w, Distributed_Matrix& Z);

  int translate_l2g_row(int local_i) const
  {
    return (g.myrow * nb) + local_i + (local_i / nb) * nb * (g.nprow - 1);
  }

  int translate_l2g_col(const int& local_j) const
  {
    return (g.mycol * nb) + local_j + (local_j / nb) * nb * (g.npcol - 1);
  }

  int translate_g2l_row(const int& global_i) const
  {
    const int local_offset_block = global_i / nb;
    return (local_offset_block - g.myrow) / g.nprow * nb + global_i % nb;
  }

  int translate_g2l_col(const int& global_j) const
  {
    const int local_offset_block = global_j / nb;
    return (local_offset_block - g.mycol) / g.npcol * nb + global_j % nb;
  }

  bool is_gindex_myrow(const int& global_i) const
  {
    int local_offset_block = global_i / nb;
    return (local_offset_block % g.nprow) == g.myrow;
  }

  bool is_gindex_mycol(const int& global_j) const
  {
    int local_offset_block = global_j / nb;
    return (local_offset_block % g.npcol) == g.mycol;
  }

  int m_global, n_global;
  int desc[9];
  double* array;
  int mb, nb;
  int m_local, n_local;
private:
  Grid g;
  int info;
};


int Symmetric_EigenSolver(Distributed_Matrix& A, double* w, Distributed_Matrix& Z)
{
  double* work = new double [1];
  long lwork = -1;
  int info = 0;

  const int ZERO=0, ONE=1;
  
  // work配列のサイズの問い合わせ
  pdsyev_( "V",  "U",  A.m_global,  A.array, ONE,  ONE,  A.desc, w, Z.array, ONE, ONE,
 	   Z.desc, work, lwork, info );
  
  lwork = work[0];
  work = new double [lwork];
  if (work == NULL) {
    cerr << "failed to allocate work." << endl;
    return info;
  }
  info = 0;
  
  // 固有値分解
  pdsyev_( "V",  "U",  A.m_global,  A.array,  ONE,  ONE,  A.desc, w, Z.array, ONE, ONE,
	   Z.desc, work, lwork, info );
  
  if (info) {
    cerr << "error at pdsyev function." << endl;
    exit(1);
  }

  return info;
}

void copy2localMatrix( double* A,double* A_global, const int& dim, const int& nprow, const int& npcol, const int& myrow, const int& mycol, const int& nb )
{
  int ZERO = 0;
  int mA = numroc_( dim, nb, myrow, ZERO, nprow );
  int nA = numroc_( dim, nb, mycol, ZERO, npcol );
  /*
  if (myrank == 0)
    cout << "mA=" << mA << "  nprow=" << nprow << endl;
  int nA = numroc_( dim, nb, mycol, ZERO, npcol );
  if (myrank == 0)
    cout<< "nA=" << nA << "  npcol=" << npcol << endl;
  */
  //int lld = mA;
  //if (lld == 0) lld = 1;

  // 各プロセスごとのローカル行列に行列成分を格納
  int global_i, global_j;
  for(int i=0; i<mA; ++i)
    for(int j=0; j<nA; ++j) {
      global_i = (myrow*nb) + i + (i/nb)*nb*(nprow-1);
      global_j = (mycol*nb) + j + (j/nb)*nb*(npcol-1);
      A[j*mA + i] = A_global(global_i, global_j);   // local matrix <- global matrix
    }
}

#include <barista/hamiltonian_scalapack_wrapping.h>

// 固有値ルーチンのテストに使うFrank行列の理論固有値
void exact_eigenvalue_frank(const int ndim, VectorXd& ev1)
{
  double dt = M_PI;   dt /= (2*ndim+1);
  for (int i=0; i<ndim; ++i) ev1[i] = 1 / (2*(1 - cos((2*i+1)*dt)));
}


int main(int argc, char *argv[]) {

   Initialize(argc, argv);
   MPI_Comm comm = (MPI_Comm) MPI_COMM_WORLD;
   Grid g(comm);

   if (g.myrank == 0) {
     cout << "nprow=" << g.nprow << "  npcol=" << g.npcol << endl;
   }

   std::ifstream  ifs(argv[1]);
   alps::Parameters params(ifs);
   barista::Hamiltonian<> hamiltonian(params);
   matrix_type A_global_matrix(hamiltonian.dimension(), hamiltonian.dimension());
   hamiltonian.fill<double>(A_global_matrix);
   int dim = hamiltonian.dimension();

   Distributed_Matrix A(dim, dim, g);
   Distributed_Matrix Z(dim, dim, g);

   hamiltonian.fill<double>(A);

   // デバッグ用グローバル行列
   double* A_global = new double[dim*dim];
   if (A_global == NULL) {
     cerr << "failed to allocate A_global." << endl;
     exit(1);
   }

   // ローカル行列とグローバル行列が対応しているかを確認
   double value;
   if (g.myrank == 0) cout << "A=" << endl;
   for (int i=0; i<dim; ++i) {
     for (int j=0; j<dim; ++j) {
       pdelget_("A", " ", value, A.array, i + 1, j + 1, A.desc);  // Fortranでは添字が1から始まることに注意
       if (g.myrank == 0)  cout << value << "  ";
     }
     if (g.myrank == 0)  cout << endl;
   }

   double *w = new double[dim];
   if (w == NULL) {
     cerr << "failed to allocate w." << endl;
     exit(1);
   }

   Symmetric_EigenSolver(A, w, Z);

   // 固有値の絶対値の降順に固有値(と対応する固有ベクトル)をソート
   // ソート後の固有値の添字をベクトルqに求める．
   double absmax;
   int qq;
   VectorXi q(dim);
   VectorXd eigval_sorted(dim);
   MatrixXd eigvec(dim,dim);
   MatrixXd eigvec_sorted(dim,dim);

   VectorXd th_eigval(dim), eigval(dim);
   exact_eigenvalue_frank(dim, th_eigval); // Frank行列の理論固有値の計算

   // 固有ベクトルの取り出し
   for (int i=0; i<dim; ++i) {
     eigval(i) = w[i];
     for (int j=0; j<dim; ++j) {
       pdelget_("A", " ", value, Z.array, i + 1, j + 1, Z.desc);  // Fortranは添字が1から始まることに注意
       eigvec(i,j) = value;
     }
   }

   // 固有値・固有ベクトルを絶対値の降順にソート
   for (int i=0; i<eigval.size(); ++i) q[i] = i;
   for (int m=0; m<eigval.size(); ++m) {
     absmax = abs(eigval[q[m]]);
     for (int i=m+1; i<eigval.rows(); ++i) {
       if (absmax < abs(eigval[q[i]])) {
	 absmax = eigval[q[i]];
	 qq = q[m];
	 q[m] = q[i];
	 q[i] = qq;
       }
     }
     eigval_sorted(m) = eigval(q[m]);
     eigvec_sorted.col(m) = eigvec.col(q[m]);    
   }

   if (g.myrank==0) {
     cout << "Theoretical eigenvalues= " << th_eigval.transpose() << endl;
     cout << "Computed eigenvalues= " << eigval_sorted.transpose() << endl;
     cout << "Computed ones - theoretical ones= " << (eigval_sorted - th_eigval).transpose() << endl << endl;

     cout << "Eigenvector:" << endl << eigvec_sorted << endl << endl;
     cout << "Check the orthogonality of eigenvectors:" << endl
	  << eigvec_sorted * eigvec_sorted.transpose() << endl;   // Is it equal to indentity matrix?
     cout << "residual := A x - lambda x = " << endl
     	  << A_global_matrix * eigvec_sorted.col(0)  -  eigval_sorted(0) * eigvec_sorted.col(0) << endl;
     cout << "Are all the following values equal to some eigenvalue = " << endl
          << (A_global_matrix * eigvec_sorted.col(0)).array() / eigvec_sorted.col(0).array() << endl;
     cout << "A_global_matrix=" << endl << A_global_matrix << endl;
   }

   delete[] A_global;
   delete[] w;

   Finalize();
}

