/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <barista/hamiltonian.h>
#include <iostream>
#include <Eigen/Dense>

using namespace std;

typedef Eigen::MatrixXd matrix_type;

static char help[] = "Compute eigenvalues of spin model by SLEPc.\n\n"
  "This example illustrates EPSSetDeflationSpace(). The example graph corresponds to a "
  "2-D regular mesh. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepceps.h>
#include <petsctime.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc, char *argv[])
{
  EPS            eps;             /* eigenproblem solver context */
  Mat            A;               /* operator matrix */
  Vec            x;
  EPSType  type;
  PetscInt       N,n=10,m,i,j,II,Istart,Iend,nev;
  PetscScalar    w;
  PetscBool      flag;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  std::ifstream  ifs(argv[1]);
  alps::Parameters params(ifs);     
  params["L"] = 3;
  cout << "L=" << params["L"] << endl;
  barista::Hamiltonian<> hamiltonian(params);
  matrix_type matrix(hamiltonian.dimension(), hamiltonian.dimension());        
  hamiltonian.fill<double>(matrix);                            
  //std::cout << matrix << std::endl;
  int dim = hamiltonian.dimension();
  N = hamiltonian.dimension();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  double *Values;
  int *Indices;
  try {
    Values = new double[dim];
    Indices = new int[dim];
    //tmp_vec = new double[dim];
  }
  catch(bad_alloc){ // 例外 bad_alloc をここで受け取る
    cerr << "bad_alloc 例外を受け取りました。" << endl;  
    abort();
  }
  
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (PetscInt i=Istart;i<Iend;++i) {
    PetscInt count = 0;
    for (int j = 0; j < N; ++j) {
      if  (matrix(i,j) != 0) {
	Values[count] = matrix(i,j);
	Indices[count] = j;
	count++;
      }
    }
    ierr = MatSetValues(A, 1, &i, count, Indices, Values, INSERT_VALUES);  //CHKERRQ(ierr);
  }

  /*
  for (II=Istart;II<Iend;++II) { 
    // i = II/n; j = II-i*n;
    for (j=0; j<N; ++j)
      if (matrix(II,j) != 0)
	ierr = MatSetValue(A, II, j, matrix(II,j), INSERT_VALUES);  CHKERRQ(ierr);
  }
  */

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"N=%d\n", N);
  // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a standard eigenvalue problem
  */
  ierr = EPSSetOperators(eps,A,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  
  /*
     Select portion of spectrum
  */
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /*
     Attach deflation space: in this case, the matrix has a constant 
     nullspace, [1 1 ... 1]^T is the eigenvector of the zero eigenvalue
  */
  //ierr = MatGetVecs(A,&x,PETSC_NULL);CHKERRQ(ierr);
  //ierr = VecSet(x,1.0);CHKERRQ(ierr);
  //ierr = EPSSetDeflationSpace(eps,1,&x);CHKERRQ(ierr);
  //ierr = VecDestroy(&x);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscLogDouble v1, v2, elapsed_time;
  ierr = PetscTime(&v1);CHKERRQ(ierr);
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = PetscTime(&v2);CHKERRQ(ierr);
  elapsed_time = v2 - v1;   

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);  CHKERRQ(ierr);

  ierr = EPSPrintSolution(eps,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"elapsed time: =%lf\n", elapsed_time);

  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;

}

