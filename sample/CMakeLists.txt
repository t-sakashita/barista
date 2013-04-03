set(TESTS hamiltonian_matrix_Eigen) 
#set(TESTS MatrixXd

foreach(name ${TESTS})
  add_executable(${name} ${name}.C)
  target_link_libraries(${name} ${ALPS_LIBRARIES})
  add_alps_test(${name})
endforeach(name)

set(TESTS_SC hamiltonian_matrix_scalapack_pdsyev hamiltonian_matrix_scalapack_pdsyev_globalmat) 

# program using ScaLAPACK
foreach(name ${TESTS_SC})
  add_executable(${name} ${name}.C)
  target_link_libraries(${name} ${ALPS_LIBRARIES})
  target_link_libraries(${name} "-lmkl_scalapack_lp64" "-lmkl_blacs_intelmpi_lp64" "-mkl=parallel")
  add_alps_test(${name})
endforeach(name)

# program using SLEPc
message(STATUS "Display PETSC_INCLUDE_DIRS=${PETSC_INCLUDE_DIR}.")
message(STATUS "Display PETSC_LIBRARIES=${PETSC_LIBRARIES}.")
message(STATUS "Display SLEPC_INCLUDE_DIRS=${SLEPC_INCLUDE_DIR}.")
message(STATUS "Display SLEPC_LIBRARIES=${SLEPC_LIBRARIES}.")

include_directories(${SLEPC_INCLUDE_DIR} ${PETSC_INCLUDE_DIR})

set(TESTS2 hamiltonian_matrix_slepc_globalmat hamiltonian_matrix_slepc_globalmat_values hamiltonian_matrix_slepc)

foreach(name ${TESTS2})
  add_executable(${name} ${name}.C)
  #リンクオプション
  target_link_libraries(${name}  ${ALPS_LIBRARIES} ${SLEPC_LIBRARIES} ${PETSC_LIBRARIES})
  ##コンパイラオプション
  add_alps_test(${name})
endforeach(name)


# program using Anasazi

if( DEFINED TRILINOS_PREFIX )
    #set(TESTS2 hamiltonian_matrix_Anasazi_LOBPCG hamiltonian_matrix_Anasazi_LOPBCG_globalmat)
    set(TESTS2 "")

    foreach(name ${TESTS2})
        add_executable(${name} ${name}.cpp)
        #リンクオプション
	target_link_libraries(${name}  ${ALPS_LIBRARIES} ${ADDITIONAL_LIBRARIES})
        ##LD_FLAG 
        ##コンパイラオプション
        #add_alps_test(${name})
    endforeach(name)
endif()


# program using	eigen_s
if( DEFINED EIGEN_S_LIB AND EIGEN_S_INC )
string(REPLACE " " ";" EIGEN_S_LIB ${EIGEN_S_LIB})
    set(TESTS2 hamiltonian_matrix_eigen_s)
    #set(TESTS2 "")

    foreach(name ${TESTS2})
       include_directories(${name} ${EIGEN_S_INC})
       add_executable(${name} eigen_wrap.f ${name}.C)
       #add_executable(${name} ${name}.C eigen_wrap.f)
#       target_link_libraries(${name}  ${ALPS_LIBRARIES})
#       foreach(libname ${EIGEN_S_LIB})
#          MESSAGE(STATUS "libname=${libname}")
#          target_link_libraries(${name} ${libname})
#       endforeach(libname)
       target_link_libraries(${name}  ${ALPS_LIBRARIES} ${EIGEN_S_LIB})
       ##コンパイラオプション
     #  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${EIGEN_S_COMPILE_FLAGS}")
       add_alps_test(${name})
    endforeach(name)
endif()

if( DEFINED EIGEN_S_LIB AND EIGEN_S_INC )

  set(TESTS  SymmetricEig_iterator) 
  #HermitianEig HermitianEigFromSequential
  include_directories(${ELEMENTAL_INCLUDE_DIRS})

  foreach(name ${TESTS})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ${ELEMENTAL_LIBRARIES})
    #add_alps_test(${name})
  endforeach(name)
endif()


#if( DEFINED TRILINOS_PREFIX )
    set(TESTS2 hamiltonian_matrix_elemental)
    include_directories(${ELEMENTAL_INCLUDE_DIRS})
    foreach(name ${TESTS2})
        add_executable(${name} ${name}.cpp)
        #リンクオプション                                                                                                                                                                   
        target_link_libraries(${name}  ${ALPS_LIBRARIES} ${ELEMENTAL_LIBRARIES})
        ##LD_FLAG                                                                                                                                                                           
        ##コンパイラオプション                                                                                                                                                              
        #add_alps_test(${name})                                                                                                                                                             
    endforeach(name)
#endif()
