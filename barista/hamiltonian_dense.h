#ifndef BARISTA_HAMILTONIAN_H
#define BARISTA_HAMILTONIAN_H

#include <alps/parameter.h>
#include <alps/lattice.h>
#include <alps/model.h>
#include <boost/foreach.hpp>

#include <rokko/distributed_matrix.hpp>
#include <iostream>
using namespace std;

namespace barista {

template<typename G = alps::coordinate_graph_type, typename I = short>
class Hamiltonian {

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

  template<typename MATRIX>
  void fill(MATRIX& A) const {  
    int dim = dimension();
    
    BOOST_FOREACH(site_descriptor v, lattice_.sites())
    add_to_matrix(A, v);
    BOOST_FOREACH(bond_descriptor e, lattice_.bonds())
    add_to_matrix(A, e, lattice_.source(e), lattice_.target(e));
  }

  template<typename MATRIX>
  void add_to_matrix(MATRIX& A, const site_descriptor v) const {
    int t = get(alps::site_type_t(), lattice_.graph(), v);
    int s = get(alps::site_index_t(), lattice_.graph(), v);
    int dim = dimension();
    int ds = basis_.basis().get_site_basis(s).num_states();
    alps::multi_array<double, 2>
    site_matrix(get_matrix(double(), model_.site_term(t), model_.basis().site_basis(t), params_));

    // Fill the matrix since we did not pass in a buffer. 
    //
    // We will fill entry (i,j) with the value i+j so that 
    // the global matrix is Hermitian. However, only one triangle of the 
    // matrix actually needs to be filled, the symmetry can be implicit.
    //
    for(int iLocal = 0; iLocal < A.get_m_local(); ++iLocal) {
      int i = A.translate_l2g_row(iLocal);
      int is = basis_[i][s];
      for (int js = 0; js < ds; ++js) {
	typename alps::basis_states<I>::value_type target(basis_[i]);
	target[s] = js;
	int j = basis_.index(target);
	if ((j < dim) && A.is_gindex_mycol(j)) {
	  int jLocal = A.translate_g2l_col(j);
	  A.update_local(iLocal, jLocal, site_matrix[is][js]);
	}
      } // end for js = ...

    } // end for i = ...
  }


  template<typename MATRIX>
  void add_to_matrix(MATRIX& A, bond_descriptor ed, site_descriptor vd0, site_descriptor vd1) const {    
    int t = get(alps::bond_type_t(), lattice_.graph(), ed);
    int st0 = get(alps::site_type_t(), lattice_.graph(), vd0);
    int st1 = get(alps::site_type_t(), lattice_.graph(), vd1);
    int s0 = get(alps::site_index_t(), lattice_.graph(), vd0);
    int s1 = get(alps::site_index_t(), lattice_.graph(), vd1);
    int dim = basis_.size();
    int ds0 = basis_.basis().get_site_basis(s0).num_states();
    int ds1 = basis_.basis().get_site_basis(s1).num_states();
    alps::multi_array<double, 4>
    bond_matrix(alps::get_matrix(double(), model_.bond_term(t), model_.basis().site_basis(st0), model_.basis().site_basis(st1), params_));

    // Fill the matrix since we did not pass in a buffer. 
    //
    // We will fill entry (i,j) with the value i+j so that 
    // the global matrix is Hermitian. However, only one triangle of the 
    // matrix actually needs to be filled, the symmetry can be implicit.
    //
    for(int iLocal = 0; iLocal < A.get_m_local(); ++iLocal) {
      int i = A.translate_l2g_row(iLocal);

      int is0 = basis_[i][s0];
      int is1 = basis_[i][s1];
      for (int js0 = 0; js0 < ds0; ++js0) {
	for (int js1 = 0; js1 < ds1; ++js1) {
	  typename alps::basis_states<I>::value_type target(basis_[i]);
	  target[s0] = js0;
	  target[s1] = js1;
	  int j = basis_.index(target);
	  if ((j < dim) && A.is_gindex_mycol(j)) {
	    int jLocal = A.translate_g2l_col(j);
	    A.update_local(iLocal, jLocal, bond_matrix[is0][is1][js0][js1]);
	  }
	}
      } // end for js0 = ...
    } // end for i = ...
  }


private:
  alps::Parameters params_;
  lattice_type lattice_;
  model_type model_;
  alps::basis_states<I> basis_;
};

} // end namespace barista

#endif // BARISTA_HAMILTONIAN_H
