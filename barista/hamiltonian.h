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

#include <Eigen/Dense>

namespace barista {

template<typename G = alps::coordinate_graph_type, typename I = short>
class Hamiltonian;

template<typename G, typename I>
  void add_to_matrix(const Hamiltonian<G, I>& hamiltonian, Eigen::MatrixXd& matrix, const typename alps::graph_helper<G>::site_descriptor& v) {
  typedef alps::graph_helper<G> lattice_type;
  typedef alps::model_helper<I> model_type;
  typedef typename lattice_type::site_descriptor site_descriptor;
  typedef typename lattice_type::bond_descriptor bond_descriptor;

  int t = get(alps::site_type_t(), hamiltonian.lattice_.graph(), v);
  int s = get(alps::site_index_t(), hamiltonian.lattice_.graph(), v);
  int dim = hamiltonian.dimension();
  int ds = hamiltonian.basis_.basis().get_site_basis(s).num_states();
  alps::multi_array<double, 2>
    site_matrix(get_matrix(double(), hamiltonian.model_.site_term(t), hamiltonian.model_.basis().site_basis(t), hamiltonian.params_));
  for (int i = 0; i < dim; ++i) {
    int is = hamiltonian.basis_[i][s];
    for (int js = 0; js < ds; ++js) {
      typename alps::basis_states<I>::value_type target(hamiltonian.basis_[i]);
      target[s] = js;
      int j = hamiltonian.basis_.index(target);
      if (j < dim) matrix(i,j) += site_matrix[is][js];
    }
  }
}


template<typename G, typename I>
  void add_to_matrix(const Hamiltonian<G, I>& hamiltonian, Eigen::MatrixXd& matrix, typename alps::graph_helper<G>::bond_descriptor ed, typename alps::graph_helper<G>::site_descriptor vd0, typename alps::graph_helper<G>::site_descriptor vd1) {
  int t = get(alps::bond_type_t(), hamiltonian.lattice_.graph(), ed);
  int st0 = get(alps::site_type_t(), hamiltonian.lattice_.graph(), vd0);
  int st1 = get(alps::site_type_t(), hamiltonian.lattice_.graph(), vd1);
  int s0 = get(alps::site_index_t(), hamiltonian.lattice_.graph(), vd0);
  int s1 = get(alps::site_index_t(), hamiltonian.lattice_.graph(), vd1);
  int dim = hamiltonian.basis_.size();
  int ds0 = hamiltonian.basis_.basis().get_site_basis(s0).num_states();
  int ds1 = hamiltonian.basis_.basis().get_site_basis(s1).num_states();
  alps::basis_states_descriptor<I> basis(hamiltonian.model_.basis(), hamiltonian.lattice_.graph());
  alps::multi_array<std::pair<double, bool>, 4>
      bond_matrix(alps::get_fermionic_matrix(double(), hamiltonian.model_.bond_term(t), hamiltonian.model_.site_basis(st0), hamiltonian.model_.site_basis(st1), hamiltonian.params_));
  for (int i = 0; i < dim; ++i) {
    int is0 = hamiltonian.basis_[i][s0];
    int is1 = hamiltonian.basis_[i][s1];
    for (int js0 = 0; js0 < ds0; ++js0) {
      for (int js1 = 0; js1 < ds1; ++js1) {
	typename alps::basis_states<I>::value_type target(hamiltonian.basis_[i]);
	target[s0] = js0;
	target[s1] = js1;
	int j = hamiltonian.basis_.index(target);
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
}



template<typename G, typename I>
void fill_func(const Hamiltonian<G, I>& hamiltonian, Eigen::MatrixXd& matrix) {
  typedef alps::graph_helper<G> lattice_type;
  typedef alps::model_helper<I> model_type;
  typedef typename lattice_type::site_descriptor site_descriptor;
  typedef typename lattice_type::bond_descriptor bond_descriptor;

  int dim = hamiltonian.dimension();
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      matrix(i,j) = 0;
  BOOST_FOREACH(site_descriptor v, hamiltonian.lattice_.sites())
    add_to_matrix(hamiltonian, matrix, v);
  BOOST_FOREACH(bond_descriptor e, hamiltonian.lattice_.bonds())
    add_to_matrix(hamiltonian, matrix, e, hamiltonian.lattice_.source(e), hamiltonian.lattice_.target(e));
}

template<typename G, typename I>
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

  template<typename Matrix>
  void fill(Matrix& matrix) const {
    fill_func(*this, matrix);
  }

    //private:
 public:
  alps::Parameters params_;
  lattice_type lattice_;
  model_type model_;
  alps::basis_states<I> basis_;
};


} // end namespace barista

#endif // BARISTA_HAMILTONIAN_H
