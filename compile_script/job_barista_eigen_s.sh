#!/bin/bash -x
#PJM -L "node=4"
#PJM -L "rscgrp=large"
#PJM --no-stging

#source /work/system/Env_base

export OMP_NUM_THREADS=8
#export PARALLEL=2

cd ${HOME}/project/barista_eigenK/build/sample

mpiexec -n 4 ./hamiltonian_matrix_eigen_s ../../sample/hamiltonian_matrix.ip 

