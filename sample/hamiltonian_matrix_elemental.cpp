#include "elemental.hpp"

#include <barista/hamiltonian_elemental.h>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace elem;

// Typedef our real type to 'R' for convenience
typedef double R;

typedef Eigen::MatrixXd matrix_type;


int main(int argc, char *argv[])
{
  Initialize( argc, argv );
  
    // Extract our MPI rank
  mpi::Comm comm = mpi::COMM_WORLD;
  const int commRank = mpi::CommRank( comm );
  
  // "/Users/sakashitatatsuya/Downloads/barista_trunk_slepc/sample/hamiltonian_matrix.ip"
  std::ifstream  ifs(argv[1]);
  alps::Parameters params(ifs);
  std::ostream& out = std::cout;

  barista::Hamiltonian<> hamiltonian(params);
  int m,n;
  int N;
  m = n = N = hamiltonian.dimension();
  //std::cout << matrix << std::endl;

  std::ofstream ofs;
  if (commRank==0) {
    ofs.open("elemental_time.txt");
    if (!ofs) {
      Finalize() ;
      return -1;
    }
  }

  // Surround the Elemental calls with try/catch statements in order to 
  // safely handle any exceptions that were thrown during execution.
  try {
    //const int n = Input("--size","matrix size",10);
    const bool print = true;
    //ProcessInput();
    //PrintInputReport();
    
    // Create a 2d process grid from a communicator. In our case, it is
    // MPI_COMM_WORLD. There is another constructor that allows you to 
    // specify the grid dimensions, Grid g( comm, r, c ), which creates an 
    // r x c grid.
    Grid g( comm );
    
    // Create an n x n real distributed matrix.
    // We distribute the matrix using grid 'g'.
    //
    // There are quite a few available constructors, including ones that 
    // allow you to pass in your own local buffer and to specify the 
    // distribution alignments (i.e., which process row and column owns the
    // top-left element)
    DistMatrix<R> H( n, n, g );
    hamiltonian.fill(H);

    if( print )
      H.Print("H");

    // Call the eigensolver. We first create an empty eigenvector 
    // matrix, X, and an eigenvalue column vector, w[VR,* ]
    DistMatrix<R,VR,STAR> w( g );
    DistMatrix<R> X( g );
    // Optional: set blocksizes and algorithmic choices here. See the 
    //           'Tuning' section of the README for details.
    HermitianEig( LOWER, H, w, X ); // only access lower half of H
    
    if( print ) {
      w.Print("Eigenvalues of H");
      X.Print("Eigenvectors of H");
    }

    // Sort the eigensolution,
    SortEig( w, X );
    
    if( print ) {
      w.Print("Sorted eigenvalues of H");
      X.Print("Sorted eigenvectors of H");
    }
  }

  catch( ArgException& e ) {
    // There is nothing to do
  }
  catch( exception& e ) {
    ostringstream os;
    os << "Process " << commRank << " caught exception with message: "
       << e.what() << endl;
    cerr << os.str();
#ifndef RELEASE
    DumpCallStack();
#endif
  }
  
    
  double time;
  //int iter;
  if (commRank == 0) {
    //time = end - start;
    //cout << "time=" << time << endl;
    //ofs << "time=" << time << endl;
    //cout << "iter=" << iter << endl;
    //ofs << "iter=" << iter << endl;
  }

  Finalize();
  return 0;
}


