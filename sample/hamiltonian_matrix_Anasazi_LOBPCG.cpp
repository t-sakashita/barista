#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"
#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziLOBPCGSolMgr.hpp"
#include "AnasaziEpetraAdapter.hpp"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include <barista/hamiltonian_anasazi.h>
#include <iostream>
#include <Eigen/Dense>

typedef Eigen::MatrixXd matrix_type;


int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif
  // My MPI process rank.                                                                                                                                                                  
  const int MyPID = Comm.MyPID();

  // "/Users/sakashitatatsuya/Downloads/barista_trunk_slepc/sample/hamiltonian_matrix.ip"
  std::ifstream  ifs(argv[1]);
  alps::Parameters params(ifs);
  Teuchos::oblackholestream blackHole;
  std::ostream& out = (MyPID == 0) ? std::cout : blackHole;

  barista::Hamiltonian<> hamiltonian(params);
  matrix_type matrix(hamiltonian.dimension(), hamiltonian.dimension());
  hamiltonian.fill<double>(matrix);
  int m,n;
  int N;
  m = n = N = hamiltonian.dimension();
  //std::cout << matrix << std::endl;

  std::ofstream ofs;
  if (MyPID==0) {
    ofs.open("anasazi_time.txt");
    if (!ofs) {
#ifdef HAVE_MPI
      MPI_Finalize() ;
#endif
      return -1;
    }
  }

  //Teuchos::ParameterList GaleriList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  typedef Teuchos::ScalarTraits<double> STS;

  const double one = STS::one();
  const double zero = STS::zero();

  // The problem is defined on a 2D grid, global size is nx * nx.
  //int nx = N; 
  //GaleriList.set("n", nx * nx);
  //GaleriList.set("nx", nx);
  //GaleriList.set("ny", nx);
  //Teuchos::RCP<Epetra_Map> Map = Teuchos::rcp( Galeri::CreateMap("Linear", Comm, GaleriList) );
  //Teuchos::RCP<Epetra_RowMatrix> A = Teuchos::rcp( Galeri::CreateCrsMatrix("Laplace2D", &*Map, GaleriList) );

  // Construct a Map that puts approximately the same number of rows
  // of the matrix A on each processor.
  Epetra_Map RowMap (N, 0, Comm);
  Epetra_Map ColMap (N, 0, Comm);
  // Get update list and number of local equations from newly created Map.
  const int NumMyRowElements = RowMap.NumMyElements ();
  std::vector<int> MyGlobalRowElements (NumMyRowElements);
  RowMap.MyGlobalElements (&MyGlobalRowElements[0]);


  // Create an Epetra_CrsMatrix using the given row map.                                                                                                                                   
  RCP<Epetra_CrsMatrix> A = rcp (new Epetra_CrsMatrix (Copy, RowMap, n));

  // We use info to catch any errors that may have happened during                                                                                                                           // matrix assembly, and report them globally.  We do this so that                                                                                                                          // the MPI processes won't call FillComplete() unless they all                                                                                                                             // successfully filled their parts of the matrix.                                                                                                                                         
  int info = 0;
  try {
    //                                                                                                                                                                                     
    // Compute coefficients for the discrete integral operator.                                                                                                                           
    //                                                                                                                                                                                      
    //std::vector<double> Values (n);
    //std::vector<int> Indices (n);

    //int count;

    hamiltonian.fill<double>(*A, MyGlobalRowElements, NumMyRowElements); //, Values, Indices);

    // Call FillComplete on the matrix.  Since the matrix isn't square,                                                                                                                    
    // we have to give FillComplete the domain and range maps, which in                                                                                                                    
    // this case are the column resp. row maps.                                                                                                                                             
    info = A->FillComplete (ColMap, RowMap);
    /*                                                                                                                                                                                     
    TEST_FOR_EXCEPTION( info != 0, std::runtime_error,                                                                                                                                     
    "FillComplete failed with INFO = " << info << ".");                                                                                                                 
    */
    info = A->OptimizeStorage();
    /*                                                                                                                                                                                     
    TEST_FOR_EXCEPTION( info != 0, std::runtime_error,                                                                                                                                
    "OptimizeStorage failed with INFO = " << info << ".");                                                                                                              
    */
  } catch (std::runtime_error& e) {
    // If multiple MPI processes are reporting errors, sometimes                                                                                                                           
    // forming the error message as a string and then writing it to                                                                                                                        
    // the output stream prevents messages from different processes                                                                                                                        
    // from being interleaved.                                                                                                                                                              
    std::ostringstream os;
    os << "*** Error on MPI process " << MyPID << ": " << e.what();
    cerr << os.str() << endl;
    if (info == 0)
      info = -1; // All procs will share info later on.                                                                                                                                     
  }

  //  Variables used for the Block Davidson Method
  const int    nev         = 5;
  const int    blockSize   = 5;
  const int    numBlocks   = 8;
  const int    maxRestarts = 500;
  const double tol         = 1.0e-8;

  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;

  // Create an Epetra_MultiVector for an initial vector to start the solver.
  // Note:  This needs to have the same number of columns as the blocksize.
  //
  //Teuchos::RCP<Epetra_MultiVector> ivec = Teuchos::rcp( new Epetra_MultiVector(*Map, blockSize) );
  Teuchos::RCP<Epetra_MultiVector> ivec = Teuchos::rcp( new Epetra_MultiVector(ColMap, blockSize) );
  ivec->Random();

  // Create the eigenproblem.
  Teuchos::RCP<Anasazi::BasicEigenproblem<double, MV, OP> > problem =
    Teuchos::rcp( new Anasazi::BasicEigenproblem<double, MV, OP>(A, ivec) );

  // Inform the eigenproblem that the operator A is symmetric
  problem->setHermitian(true);

  // Set the number of eigenvalues requested
  problem->setNEV( nev );

  // Inform the eigenproblem that you are finishing passing it information
  bool boolret = problem->setProblem();
  if (boolret != true) {
    std::cout<<"Anasazi::BasicEigenproblem::setProblem() returned an error." << std::endl;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  // Create parameter list to pass into the solver manager
  Teuchos::ParameterList anasaziPL;
  anasaziPL.set( "Which", "LM" );
  anasaziPL.set( "Block Size", blockSize );
  anasaziPL.set( "Maximum Iterations", 500 );
  anasaziPL.set( "Convergence Tolerance", tol );
  anasaziPL.set( "Verbosity", Anasazi::Errors+Anasazi::Warnings+Anasazi::TimingDetails+Anasazi::FinalSummary );

  // Create the solver manager
  Anasazi::LOBPCGSolMgr<double, MV, OP> anasaziSolver(problem, anasaziPL);

  // Solve the problem
  double start, end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  Anasazi::ReturnType returnCode = anasaziSolver.solve();
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  // Get the eigenvalues and eigenvectors from the eigenproblem
  Anasazi::Eigensolution<double,MV> sol = problem->getSolution();
  std::vector<Anasazi::Value<double> > evals = sol.Evals;
  Teuchos::RCP<MV> evecs = sol.Evecs;

  // Compute residuals.
  std::vector<double> normR(sol.numVecs);
  Teuchos::SerialDenseMatrix<int,double> T(sol.numVecs, sol.numVecs);
  Epetra_MultiVector tempAevec( ColMap, sol.numVecs );
  T.putScalar(0.0); 
  for (int i=0; i<sol.numVecs; i++) {
    T(i,i) = evals[i].realpart;
  }
  A->Apply( *evecs, tempAevec );
  MVT::MvTimesMatAddMv( -1.0, *evecs, T, 1.0, tempAevec );
  MVT::MvNorm( tempAevec, normR );

  if (MyPID == 0) {
  // Print the results
  std::cout<<"Solver manager returned " << (returnCode == Anasazi::Converged ? "converged." : "unconverged.") << std::endl;
  std::cout<<std::endl;
  std::cout<<"------------------------------------------------------"<<std::endl;
  std::cout<<std::setw(16)<<"Eigenvalue"
           <<std::setw(18)<<"Direct Residual"
           <<std::endl;
  std::cout<<"------------------------------------------------------"<<std::endl;
  for (int i=0; i<sol.numVecs; i++) {
    std::cout<<std::setw(16)<<evals[i].realpart
             <<std::setw(18)<<normR[i]/evals[i].realpart
             <<std::endl;
  }
  std::cout<<"------------------------------------------------------"<<std::endl;
  }

  // Print out the map and matrices
  //ColMap.Print (out);

  //A->Print (cout);
  //RowMap.Print (cout);

  double time;
  int iter;
  if (MyPID==0) {
    iter = anasaziSolver.getNumIters();
    Teuchos::Array<Teuchos::RCP<Teuchos::Time> > timer = anasaziSolver.getTimers();
    Teuchos::RCP<Teuchos::Time> _timerSolve = timer[0];
    cout << "timerSolve=" << _timerSolve << endl;
    time = end - start;
    cout << "time=" << time << endl;
    ofs << "time=" << time << endl;
    cout << "iter=" << iter << endl;
    ofs << "iter=" << iter << endl;
  }

#ifdef HAVE_MPI
  MPI_Finalize() ; 
#endif

  return 0;
}


