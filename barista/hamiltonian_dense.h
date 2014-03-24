/*****************************************************************************
 *
 * Barista: Eigen-decomposition library for quantum statistical physics
 *
 * Copyright (C) 2012-2013 by Tatsuya Sakashita <t-sakashita@issp.u-tokyo.ac.jp>,
 *                         by Synge Todo <wistaria@comp-phys.org>,
 *               2014      by Ryo IGARASHI <rigarash@issp.u-tokyo.ac.jp>
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 *****************************************************************************/

#ifndef BARISTA_HAMILTONIAN_DENSE_H
#define BARISTA_HAMILTONIAN_DENSE_H

//#include <alps/parameter.h>
//#include <alps/lattice.h>
//#include <alps/model.h>
//#include <boost/foreach.hpp>

#include <rokko/localized_matrix.hpp>
#include <barista/hamiltonian.h>

#include <iostream>

using namespace std;

namespace barista {


template<>
void Hamiltonian::add_to_matrix(rokko::localized_matrix<rokko::matrix_col_major>& A, const site_descriptor v) const {
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


template<>
void Hamiltonian::add_to_matrix(rokko::distributed_matrix<rokko::matrix_col_major>& A, bond_descriptor ed, site_descriptor vd0, site_descriptor vd1) const {    
    int t = get(alps::bond_type_t(), lattice_.graph(), ed);
    int st0 = get(alps::site_type_t(), lattice_.graph(), vd0);
    int st1 = get(alps::site_type_t(), lattice_.graph(), vd1);
    int s0 = get(alps::site_index_t(), lattice_.graph(), vd0);
    int s1 = get(alps::site_index_t(), lattice_.graph(), vd1);
    int dim = basis_.size();
    int ds0 = basis_.basis().get_site_basis(s0).num_states();
    int ds1 = basis_.basis().get_site_basis(s1).num_states();
    alps::basis_states_descriptor<I> basis(hamiltonian.model_.basis(), hamiltonian.lattice_.graph());
    alps::multi_array<std::pair<double, bool>, 4>
    bond_matrix(alps::get_fermionic_matrix(double(), model_.bond_term(t), model_.basis().site_basis(st0), model_.basis().site_basis(st1), params_));

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
            double val = bond_matrix[is0][is1][js0][js1].first;
            if (bond_matrix[is0][is1][js0][js1].second) {
                // calculate fermionic sign
                bool f = (s1 >= s0);
                int start = std::min(s0, s1);
                int end   = std::max(s0, s1);
                for (int k = start; k < end; ++k) {
                    if (is_fermionic(hamiltonian.model_.site_basis(hamiltonian.lattice_.site_type(k)), basis[k][hamiltonian.basis_[i][k]])) {
                        f = !f;
                    }
                }
                if (f) {
                    val *= -1;
                }
            }
	    int jLocal = A.translate_g2l_col(j);
	    A.update_local(iLocal, jLocal, val);
	  }
	}
      } // end for js0 = ...
    } // end for i = ...
}


} // end namespace barista

#endif // BARISTA_HAMILTONIAN_DENSE_H
