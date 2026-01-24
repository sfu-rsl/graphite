#pragma once
#include <graphite/common.hpp>
#include <graphite/factor.hpp>
#include <graphite/hessian.hpp>
#include <graphite/kernel/vector.hpp>
#include <graphite/preconditioner/preconditioner.hpp>
#include <graphite/stream.hpp>
#include <graphite/utils.hpp>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

namespace graphite {

template <typename T, typename S> class Solver {
public:
  virtual ~Solver() = default;

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  StreamPool &streams) = 0;

  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) = 0;

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams) = 0;

  virtual bool solve(Graph<T, S> *graph, T *delta_x, StreamPool &streams) = 0;
};

} // namespace graphite