#!/bin/bash -x
#PJM -L "rscunit=unit1"
#PJM -L "rscgrp=large"
#PJM --rsc-list "node=8"
#PJM --rsc-list "elapse=02:00:00"
#PJM --mpi "proc=8"
#PJM -s

#. /work/system/Env_base

PROF_HOME=/k/home/tcs/opt_fe/opt/FJSVfxlang/1.2.1
export PATH="${PROF_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${PROF_HOME}/lib64:${LD_LIBRARY_PATH}"

export OMP_NUM_THREADS=16
#export PARALLEL=2

cd /k/home/users/zs4/project/barista_anasazi_trunk2/build/sample

fipp -C -Ihwm -d ../../prof mpiexec ./hamiltonian_matrix_AnasaziLOPBCG ../../sample/hamiltonian_matrix.ip


