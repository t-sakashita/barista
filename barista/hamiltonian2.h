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

#ifndef BARISTA_HAMILTONIAN_H
#define BARISTA_HAMILTONIAN_H

#include <alps/parameter.h>
#include <alps/lattice.h>
#include <alps/model.h>
#include <boost/foreach.hpp>

#include "Epetra_CrsMatrix.h"
#include "Teuchos_RCP.hpp"

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


template<typename T>
//void fill(Teuchos::RCP<Epetra_CrsMatrix> A, const std::vector<double>& MyGlobalRowElements, const int& NumMyRowElements, std::vector<double>& Values, std::vector<int>& Indices) const {
void fill(Epetra_CrsMatrix& A, const std::vector<int>& MyGlobalRowElements, const int& NumMyRowElements) const {

  typedef T value_type;
  //std::vector<double> Values;
  //std::vector<int> Indices;

  int dim = dimension();

  BOOST_FOREACH(site_descriptor v, lattice_.sites())
  add_to_matrix<value_type>(A, MyGlobalRowElements, NumMyRowElements, v);
  BOOST_FOREACH(bond_descriptor e, lattice_.bonds())
  add_to_matrix<value_type>(A, MyGlobalRowElements, NumMyRowElements, e, lattice_.source(e), lattice_.target(e));
}



template<typename T>
void add_to_matrix(Epetra_CrsMatrix& A, const std::vector<int>& MyGlobalRowElements, const int& NumMyRowElements, const site_descriptor v) const {
  int t = get(alps::site_type_t(), lattice_.graph(), v);
  int s = get(alps::site_index_t(), lattice_.graph(), v);
  int dim = dimension();
  int i;
  std::vector<double> Values(dim);
  std::vector<int> Indices(dim);
  std::vector<double> tmp_vec(dim);
  int count;
  int info;
  int ds = basis_.basis().get_site_basis(s).num_states();
  alps::multi_array<T, 2>
  site_matrix(get_matrix(T(), model_.site_term(t), model_.basis().site_basis(t), params_));
  for (int i_orig = 0; i_orig < NumMyRowElements; ++i_orig) {
    i = MyGlobalRowElements[i_orig];
    for (int j = 0; j < dim; ++j)  tmp_vec[j] = 0;
    int is = basis_[i][s];
    for (int js = 0; js < ds; ++js) {
      typename alps::basis_states<I>::value_type target(basis_[i]);
      target[s] = js;
      int j = basis_.index(target);
      if (j < dim) tmp_vec[j] += site_matrix[is][js];
    } // end for js = ...

    count = 0;
    for (int j = 0; j < dim; ++j) {
      if  (tmp_vec[j] != 0) {
	Values[count] = tmp_vec[j];
	Indices[count] = j;
	count++;
      }
    }
    info = A.InsertGlobalValues (i, count, &Values[0], &Indices[0]);
  } // end for i_orig = ...
}



template<typename T>
void add_to_matrix(Epetra_CrsMatrix& A, const std::vector<int>& MyGlobalRowElements, const int& NumMyRowElements, bond_descriptor ed, site_descriptor vd0, site_descriptor vd1) const {
  int t = get(alps::bond_type_t(), lattice_.graph(), ed);
  int st0 = get(alps::site_type_t(), lattice_.graph(), vd0);
  int st1 = get(alps::site_type_t(), lattice_.graph(), vd1);
  int s0 = get(alps::site_index_t(), lattice_.graph(), vd0);
  int s1 = get(alps::site_index_t(), lattice_.graph(), vd1);
  int dim = basis_.size();
  int i;
  std::vector<double> Values(dim);
  std::vector<int> Indices(dim);
  std::vector<double> tmp_vec(dim);
  int count;
  int info;
  int ds0 = basis_.basis().get_site_basis(s0).num_states();
  int ds1 = basis_.basis().get_site_basis(s1).num_states();
  alps::basis_states_descriptor<I> basis(hamiltonian.model_.basis(), hamiltonian.lattice_.graph());
  alps::multi_array<std::pair<T, bool>, 4>
  bond_matrix(alps::get_fermionic_matrix(T(), model_.bond_term(t), model_.basis().site_basis(st0), model_.basis().site_basis(st1), params_));
  for (int i_orig = 0; i_orig < NumMyRowElements; ++i_orig) {
    i = MyGlobalRowElements[i_orig];
    for (int j = 0; j < dim; ++j)  tmp_vec[j] = 0;
    int is0 = basis_[i][s0];
    int is1 = basis_[i][s1];
    for (int js0 = 0; js0 < ds0; ++js0) {
      for (int js1 = 0; js1 < ds1; ++js1) {
	typename alps::basis_states<I>::value_type target(basis_[i]);
	target[s0] = js0;
	target[s1] = js1;
	int j = basis_.index(target);
	if (j < dim) {
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
            tmp_vec[j] += val;
        }
      }
    } // end for js0 = ...

    count = 0;
    for (int j = 0; j < dim; ++j) {
      if  (tmp_vec[j] != 0) {
        Values[count] = tmp_vec[j];
        Indices[count] = j;
        count++;
      }
    }
    info = A.InsertGlobalValues (i, count, &Values[0], &Indices[0]);
  } // end for i_orig = ...
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
    alps::basis_states_descriptor<I> basis(hamiltonian.model_.basis(), hamiltonian.lattice_.graph());
    alps::multi_array<std::pair<T, bool>, 4>
      bond_matrix(alps::get_fermionic_matrix(T(), model_.bond_term(t), model_.basis().site_basis(st0), model_.basis().site_basis(st1), params_));
    for (int i = 0; i < dim; ++i) {
      int is0 = basis_[i][s0];
      int is1 = basis_[i][s1];
      for (int js0 = 0; js0 < ds0; ++js0) {
        for (int js1 = 0; js1 < ds1; ++js1) {
          typename alps::basis_states<I>::value_type target(basis_[i]);
          target[s0] = js0;
          target[s1] = js1;
          int j = basis_.index(target);
          if (j < dim) {
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
            matrix(i,j) += val;
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
