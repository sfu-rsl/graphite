/// @file schur_preconditioner.hpp
#pragma once

#include <graphite/graph.hpp>
#include <graphite/schur.hpp>

namespace graphite {

/// @brief Base class for Schur complement preconditioners
template <typename T, typename S> class SchurPreconditioner {
public:
  virtual ~SchurPreconditioner() = default;

  virtual void update_structure(Graph<T, S> *graph,
                                SchurComplement<T, S> *schur,
                                StreamPool &streams) = 0;

  virtual void update_values(Graph<T, S> *graph, SchurComplement<T, S> *schur,
                             StreamPool &streams) = 0;

  virtual void set_damping_factor(Graph<T, S> *graph,
                                  SchurComplement<T, S> *schur,
                                  T damping_factor, const bool use_identity,
                                  StreamPool &streams) = 0;

  virtual void apply(Graph<T, S> *graph, SchurComplement<T, S> *schur, T *z,
                     const T *r, StreamPool &streams) = 0;
};

} // namespace graphite
