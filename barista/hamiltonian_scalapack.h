#ifndef BARISTA_HAMILTONIAN_H
#define BARISTA_HAMILTONIAN_H

#include <alps/parameter.h>
#include <alps/lattice.h>
#include <alps/model.h>
#include <boost/foreach.hpp>

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

namespace barista {

template<typename G = alps::coordinate_graph_type, typename I = short>
class Hamiltonian {

  //Teuchos::ParameterList GaleriList;
  //  using Teuchos::RCP;
  //using Teuchos::rcp;
  //typedef Teuchos::ScalarTraits<double> STS;

private:
  typedef alps::graph_helper<G> lattice_type;
  typedef alps::model_helper<I> model_type;
  typedef typename lattice_type::site_descriptor site_descriptor;
  typedef typename lattice_type::bond_descriptor bond_descriptor;

public:
  Hamiltonian(alps::Parameters const& p) : params_(p), lattice_(params_), model_(lattice_, params_),
    basis_(alps::basis_states_descriptor<I>(model_.basis(), lattice_.graph())) {
  }

  int dimension() const { return basis_.size(); }

  template<typename T, typename Matrix>
  void fill(Matrix& matrix) const {
    typedef T value_type;
    int dim = dimension();
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        matrix(i,j) = 0;
    BOOST_FOREACH(site_descriptor v, lattice_.sites())
      add_to_matrix<value_type>(matrix, v);
    BOOST_FOREACH(bond_descriptor e, lattice_.bonds())
      add_to_matrix<value_type>(matrix, e, lattice_.source(e), lattice_.target(e));
  }

  template<typename T, typename MATRIX>
  void add_to_matrix(MATRIX& matrix, const site_descriptor v) const {
    int t = get(alps::site_type_t(), lattice_.graph(), v);
    int s = get(alps::site_index_t(), lattice_.graph(), v);
    int dim = dimension();
    int ds = basis_.basis().get_site_basis(s).num_states();
    alps::multi_array<T, 2>
      site_matrix(get_matrix(T(), model_.site_term(t), model_.basis().site_basis(t), params_));
    for (int i = 0; i < dim; ++i) {
      int is = basis_[i][s];
      for (int js = 0; js < ds; ++js) {
        typename alps::basis_states<I>::value_type target(basis_[i]);
        target[s] = js;
        int j = basis_.index(target);
        if (j < dim) matrix(i,j) += site_matrix[is][js];
      }
    }
  }

  template<typename T, class MATRIX>
  void add_to_matrix(MATRIX& matrix, bond_descriptor ed, site_descriptor vd0, site_descriptor vd1) const {
    int t = get(alps::bond_type_t(), lattice_.graph(), ed);
    int st0 = get(alps::site_type_t(), lattice_.graph(), vd0);
    int st1 = get(alps::site_type_t(), lattice_.graph(), vd1);
    int s0 = get(alps::site_index_t(), lattice_.graph(), vd0);
    int s1 = get(alps::site_index_t(), lattice_.graph(), vd1);
    int dim = basis_.size();
    int ds0 = basis_.basis().get_site_basis(s0).num_states();
    int ds1 = basis_.basis().get_site_basis(s1).num_states();
    alps::multi_array<T, 4>
      bond_matrix(alps::get_matrix(T(), model_.bond_term(t), model_.basis().site_basis(st0), model_.basis().site_basis(st1), params_));
    for (int i = 0; i < dim; ++i) {
      int is0 = basis_[i][s0];
      int is1 = basis_[i][s1];
      for (int js0 = 0; js0 < ds0; ++js0) {
        for (int js1 = 0; js1 < ds1; ++js1) {
          typename alps::basis_states<I>::value_type target(basis_[i]);
          target[s0] = js0;
          target[s1] = js1;
          int j = basis_.index(target);
          if (j < dim) matrix(i,j) += bond_matrix[is0][is1][js0][js1];
        }
      }
    }
  }

template<typename T>
void fill(Distributed_Matrix& A, const int& nprow, const int& npcol, const int& myrow, const int& mycol, const int& nb) const {

  typedef T value_type;

  int dim = dimension();
  int ZERO = 0;
  int mA = numroc_( dim, nb, myrow, ZERO, nprow );
  int nA = numroc_( dim, nb, mycol, ZERO, npcol );

  int global_i, global_j;
  for(int i=0; i<mA; ++i)
    for(int j=0; j<nA; ++j) {
      A[j*mA + i] = 0;
    }

  BOOST_FOREACH(site_descriptor v, lattice_.sites())
  add_to_matrix<value_type>(A, nprow, npcol, myrow, mycol, nb, v);
  BOOST_FOREACH(bond_descriptor e, lattice_.bonds())
  add_to_matrix<value_type>(A, nprow, npcol, myrow, mycol, nb, e, lattice_.source(e), lattice_.target(e));
}

template<typename T>
void add_to_matrix(Distributed_Matrix& A, const int& nprow, const int& npcol, const int& myrow, const int& mycol, const int& nb, const site_descriptor v) const {
  int ZERO = 0;
  int dim = dimension();
  int mA = numroc_( dim, nb, myrow, ZERO, nprow );
  int nA = numroc_( dim, nb, mycol, ZERO, npcol );

  int t = get(alps::site_type_t(), lattice_.graph(), v);
  int s = get(alps::site_index_t(), lattice_.graph(), v);
  int ds = basis_.basis().get_site_basis(s).num_states();
  alps::multi_array<T, 2>
  site_matrix(get_matrix(T(), model_.site_term(t), model_.basis().site_basis(t), params_));
  for (int local_i = 0; local_i < mA; ++local_i) {
    int i = translate_l2g_row(local_i); //(myrow*nb) + local_i + (local_i/nb)*nb*(nprow-1);
    int is = basis_[i][s];
    for (int js = 0; js < ds; ++js) {
      typename alps::basis_states<I>::value_type target(basis_[i]);
      target[s] = js;
      int j = basis_.index(target);
      int local_offset_block = j / nb;
      if ((local_offset_block % npcol) == mycol) {
	int local_j = translate_l2g_col(j);  //(local_offset_block - mycol) / npcol * nb + j % nb;
	if (local_j >= nA) {
	  std::cerr << "error: local_j=" << local_j << " nA=" << nA << endl;
	  exit(23);
	}
	else
	  A[local_j*mA + local_i] += site_matrix[is][js];    // local matrix <- global matrix
      }
    }
  }
}


template<typename T>
void add_to_matrix(Distributed_Matrix& A, const int& nprow, const int& npcol, const int& myrow, const int& mycol, const int& nb, bond_descriptor ed, site_descriptor vd0, site_descriptor vd1) const {
  int ZERO = 0;
  int dim = dimension();
  int mA = numroc_( dim, nb, myrow, ZERO, nprow );
  int nA = numroc_( dim, nb, mycol, ZERO, npcol );

  int t = get(alps::bond_type_t(), lattice_.graph(), ed);
  int st0 = get(alps::site_type_t(), lattice_.graph(), vd0);
  int st1 = get(alps::site_type_t(), lattice_.graph(), vd1);
  int s0 = get(alps::site_index_t(), lattice_.graph(), vd0);
  int s1 = get(alps::site_index_t(), lattice_.graph(), vd1);
  int ds0 = basis_.basis().get_site_basis(s0).num_states();
  int ds1 = basis_.basis().get_site_basis(s1).num_states();
  alps::multi_array<T, 4>
  bond_matrix(alps::get_matrix(T(), model_.bond_term(t), model_.basis().site_basis(st0), model_.basis().site_basis(st1), params_));
  for (int local_i = 0; local_i < mA; ++local_i) {
    int i = (myrow*nb) + local_i + (local_i/nb)*nb*(nprow-1);
    int is0 = basis_[i][s0];
    int is1 = basis_[i][s1];
    for (int js0 = 0; js0 < ds0; ++js0) {
      for (int js1 = 0; js1 < ds1; ++js1) {
	typename alps::basis_states<I>::value_type target(basis_[i]);
	target[s0] = js0;
	target[s1] = js1;
	int j = basis_.index(target);
	int local_offset_block = j / nb;
	if ((local_offset_block % npcol) == mycol) {
	  int local_j = (local_offset_block - mycol) / npcol * nb + j % nb;
	  if (local_j >= nA) {
	    std::cerr << "error: local_j=" << local_j << " nA=" << nA << endl;
	    exit(23);
	  }
	  else
	    A[local_j*mA + local_i] += bond_matrix[is0][is1][js0][js1];    // local matrix <- global matrix
	}
      }
    }
  }
}


private:
  alps::Parameters params_;
  lattice_type lattice_;
  model_type model_;
  alps::basis_states<I> basis_;
};

} // end namespace barista

#endif // BARISTA_HAMILTONIAN_H
