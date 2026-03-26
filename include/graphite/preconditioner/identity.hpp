/// @file identity.hpp
#pragma once
#include <graphite/preconditioner/preconditioner.hpp>

namespace graphite {

template <typename T, typename S>
class IdentityPreconditioner : public Preconditioner<T, S> {
private:
  size_t dimension;

public:
  virtual void update_structure(Graph<T, S> *graph, StreamPool &streams) {
    dimension = graph->get_hessian_dimension();
  };

  virtual void update_values(Graph<T, S> *graph, StreamPool &streams){};

  virtual void set_damping_factor(Graph<T, S> *graph, T damping_factor,
                                  const bool use_identity,
                                  StreamPool &streams) {
    (void)graph;
    (void)damping_factor;
    (void)use_identity;
    (void)streams;
  };

  void apply(Graph<T, S> *graph, T *z, const T *r,
             StreamPool &streams) override {
    cudaMemcpy(z, r, dimension * sizeof(T), cudaMemcpyDeviceToDevice);
  }
};

} // namespace graphite