#pragma once
#include <glso/active.hpp>
#include <glso/common.hpp>
#include <glso/differentiation.hpp>
#include <glso/loss.hpp>
#include <glso/op.hpp>
#include <glso/utils.hpp>
#include <glso/vector.hpp>
#include <glso/vertex.hpp>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/universal_vector.h>

namespace glso {

template <typename S> class JacobianStorage {
public:
  std::pair<size_t, size_t> dimensions;

  thrust::device_vector<S> data;
};

template <typename T, typename S> class BaseFactorDescriptor {
public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

  virtual ~BaseFactorDescriptor(){};

  // virtual void error_func(const T** vertices, const T* obs, T* error) = 0;
  virtual bool use_autodiff() = 0;
  virtual void visit_error(GraphVisitor<T, S> &visitor) = 0;
  virtual void visit_error_autodiff(GraphVisitor<T, S> &visitor,
                                    StreamPool &streams) = 0;
  virtual void visit_b(GraphVisitor<T, S> &visitor, T *b,
                       const T *jacobian_scales) = 0;
  virtual void visit_Jv(GraphVisitor<T, S> &visitor, T *out, T *in,
                        const T *jacobian_scales, cudaStream_t *streams,
                        size_t num_streams) = 0;
  virtual void visit_Jtv(GraphVisitor<T, S> &visitor, T *out, T *in,
                         const T *jacobian_scales, cudaStream_t *streams,
                         size_t num_streams) = 0;

  virtual void visit_flag_active_vertices(GraphVisitor<T, S> &visitor,
                                          const uint8_t level) = 0;

  virtual void visit_jacobians(GraphVisitor<T, S> &visitor,
                               StreamPool &streams) = 0;
  virtual void visit_block_diagonal(
      GraphVisitor<T, S> &visitor,
      std::unordered_map<BaseVertexDescriptor<T, S> *,
                         thrust::device_vector<InvP>> &block_diagonals,
      const T *jacobian_scales) = 0;
  virtual void visit_scalar_diagonal(GraphVisitor<T, S> &visitor, T *diagonal,
                                     const T *jacobian_scales) = 0;
  // virtual void apply_op(Op<T>& op) = 0;

  virtual JacobianStorage<S> *get_jacobians() = 0;
  virtual void initialize_jacobian_storage() = 0;
  // virtual size_t get_num_vertices() const = 0;

  // virtual size_t internal_count() const = 0;
  virtual size_t active_count() const = 0;

  virtual size_t get_residual_size() const = 0;
  virtual void scale_jacobians(GraphVisitor<T, S> &visitor,
                               T *jacobian_scales) = 0;

  virtual void initialize_device_ids(const uint8_t optimization_level) = 0;
  virtual void to_device() = 0;

  virtual T chi2(GraphVisitor<T, S> &visitor) = 0;

  virtual void set_jacobian_storage(const bool store) = 0;
  virtual bool store_jacobians() = 0;
  virtual bool supports_dynamic_jacobians() = 0;
};

template <typename T> struct get_vertex_type {
  using type = typename T::VertexType;
};

template <typename T> struct get_vertex_pointer_type {
  using type = typename T::VertexType *;
};

template <typename T> struct get_vertex_pointer_pointer_type {
  using type = typename T::VertexType **;
};

template <typename Tuple, template <typename> class MetaFunc>
struct transform_tuple;

// Partial specialization for std::tuple
template <typename... Ts, template <typename> class MetaFunc>
struct transform_tuple<std::tuple<Ts...>, MetaFunc> {
  using type = std::tuple<typename MetaFunc<Ts>::type...>;
};

template <typename Tuple, template <typename> class MetaFunc>
using transform_tuple_t = typename transform_tuple<Tuple, MetaFunc>::type;

// template <typename T, int E, typename M, typename C, template <typename, int>
// class L, template <typename> class Derived, typename... VDTypes>
template <typename T, typename S, typename FTraits>
class FactorDescriptor : public BaseFactorDescriptor<T, S> {

private:
  std::vector<size_t> global_ids;
  // thrust::host_vector<size_t> host_ids; // local ids
  // thrust::host_vector<T> host_obs;
  std::unordered_map<size_t, size_t> global_to_local_map;
  std::vector<size_t> local_to_global_map;

  HandleManager<size_t> hm;

  bool _store_jacobians;
  size_t _active_count;

public:
  using InvP = std::conditional_t<is_low_precision<S>::value, T, S>;

  // using Traits = FactorTraits<T, Derived>;
  using Traits = class FTraits;

  // static constexpr size_t N = sizeof...(VDTypes);
  static constexpr size_t N =
      std::tuple_size<typename Traits::VertexDescriptors>::value;
  // static constexpr size_t error_dim = E;
  static constexpr size_t error_dim = Traits::dimension;

  std::array<BaseVertexDescriptor<T, S> *, N> vertex_descriptors;
  // using VertexTypesTuple = std::tuple<typename VDTypes::VertexType...>;
  // using VertexPointerTuple = std::tuple<typename VDTypes::VertexType*...>;
  // using VertexPointerPointerTuple = std::tuple<typename
  // VDTypes::VertexType**...>;

  using VertexDescriptorTuple = typename Traits::VertexDescriptors;
  using VertexTypesTuple =
      transform_tuple_t<VertexDescriptorTuple, get_vertex_type>;
  using VertexPointerTuple =
      transform_tuple_t<VertexDescriptorTuple, get_vertex_pointer_type>;
  using VertexPointerPointerTuple =
      transform_tuple_t<VertexDescriptorTuple, get_vertex_pointer_pointer_type>;

  using ObservationType = typename Traits::Observation;
  using ConstraintDataType = typename Traits::Data;
  // using LossType = typename Traits::LossType<error_dim>;
  using LossType = typename Traits::Loss;

  // using ObservationType = M;
  // using ConstraintDataType = C;
  // using LossType = L<T, E>;

  // uninitialized_vector<size_t> device_ids; // local ids
  thrust::host_vector<size_t> host_ids;
  thrust::device_vector<size_t> device_ids;
  uninitialized_vector<ObservationType> device_obs;
  thrust::device_vector<T> residuals;
  uninitialized_vector<S> precision_matrices;
  uninitialized_vector<ConstraintDataType> data;

  uninitialized_vector<T> chi2_vec;
  thrust::device_vector<S> chi2_derivative;
  uninitialized_vector<LossType> loss;

  thrust::host_vector<uint8_t> active;
  thrust::device_vector<uint8_t> device_active;
  thrust::device_vector<size_t> active_indices;

  std::array<JacobianStorage<S>, N> jacobians;
  std::array<S, Traits::dimension * Traits::dimension> default_precision_matrix;

  template <typename... VertexDescPtrs,
            typename = std::enable_if_t<sizeof...(VertexDescPtrs) == N>>
  FactorDescriptor(VertexDescPtrs... vertex_descriptors)
      : _store_jacobians(true), _active_count(0) {
    link_factors({vertex_descriptors...});

    default_precision_matrix = get_default_precision_matrix();
  }

  void visit_error(GraphVisitor<T, S> &visitor) override {
    visitor.template compute_error(this);
  }

  void visit_error_autodiff(GraphVisitor<T, S> &visitor,
                            StreamPool &streams) override {
    visitor.template compute_error_autodiff(this, streams);
  }

  void visit_b(GraphVisitor<T, S> &visitor, T *b,
               const T *jacobian_scales) override {
    visitor.template compute_b(this, b, jacobian_scales);
  }

  void visit_Jv(GraphVisitor<T, S> &visitor, T *out, T *in,
                const T *jacobian_scales, cudaStream_t *streams,
                size_t num_streams) override {
    visitor.template compute_Jv(this, out, in, jacobian_scales, streams,
                                num_streams);
  }

  void visit_Jtv(GraphVisitor<T, S> &visitor, T *out, T *in,
                 const T *jacobian_scales, cudaStream_t *streams,
                 size_t num_streams) override {
    visitor.template compute_Jtv(this, out, in, jacobian_scales, streams,
                                 num_streams);
  }

  void visit_flag_active_vertices(GraphVisitor<T, S> &visitor,
                                  const uint8_t level) override {
    visitor.template visit_flag_active_vertices(this, level);
  }

  void visit_jacobians(GraphVisitor<T, S> &visitor,
                       StreamPool &streams) override {
    if constexpr (std::is_same_v<typename Traits::Differentiation,
                                 DifferentiationMode::Manual>) {
      visitor.template compute_jacobians(this, streams);
    }
  }

  void visit_block_diagonal(
      GraphVisitor<T, S> &visitor,
      std::unordered_map<BaseVertexDescriptor<T, S> *,
                         thrust::device_vector<InvP>> &block_diagonals,
      const T *jacobian_scales) override {

    std::array<InvP *, N> diagonal_blocks;
    for (size_t i = 0; i < N; i++) {
      diagonal_blocks[i] = block_diagonals[vertex_descriptors[i]].data().get();
      // std::cout << "BD size: " <<
      // block_diagonals[vertex_descriptors[i]].size() << std::endl;
    }

    visitor.template compute_block_diagonal(this, diagonal_blocks,
                                            jacobian_scales);
  }

  void visit_scalar_diagonal(GraphVisitor<T, S> &visitor, T *diagonal,
                             const T *jacobian_scales) override {
    visitor.template compute_scalar_diagonal(this, diagonal, jacobian_scales);
  }

  static constexpr size_t get_num_vertices() { return N; }

  JacobianStorage<S> *get_jacobians() override { return jacobians.data(); }

  void reserve(size_t size) {
    global_ids.reserve(N * size);
    host_ids.reserve(N * size);
    device_ids.reserve(N * size);
    device_obs.reserve(size);
    global_to_local_map.reserve(size);
    local_to_global_map.reserve(size);
    precision_matrices.reserve(size * error_dim * error_dim);
    data.reserve(size);
    loss.reserve(size);
    chi2_vec.reserve(size);
    residuals.reserve(size * error_dim);
    active.reserve(size);

    // Prefetch everything
    /*
    int cuda_device = cudaCpuDeviceId;
    constexpr cudaStream_t stream = 0;
    // prefetch_vector_on_device_async(device_ids, cuda_device, stream);
    prefetch_vector_on_device_async(device_obs, cuda_device, stream);
    prefetch_vector_on_device_async(precision_matrices, cuda_device, stream);
    prefetch_vector_on_device_async(data, cuda_device, stream);
    prefetch_vector_on_device_async(loss, cuda_device, stream);
    prefetch_vector_on_device_async(chi2_vec, cuda_device, stream);
    cudaDeviceSynchronize();
    */
  }

  void remove_factor(const size_t id) {
    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return;
    }

    auto local_id = global_to_local_map[id];
    auto last_local_id = internal_count() - 1;

    // copy constraint
    for (size_t i = 0; i < N; i++) {
      global_ids[local_id * N + i] = global_ids[last_local_id * N + i];
    }
    global_ids.resize(global_ids.size() - N);

    // observation
    device_obs[local_id] = device_obs[last_local_id];
    device_obs.pop_back();

    // precision matrix
    constexpr size_t precision_matrix_size = error_dim * error_dim;
    for (size_t i = 0; i < precision_matrix_size; i++) {
      precision_matrices[local_id * precision_matrix_size + i] =
          precision_matrices[last_local_id * precision_matrix_size + i];
    }

    precision_matrices.resize(precision_matrices.size() -
                              precision_matrix_size);

    // Constraint, loss and chi2
    data[local_id] = data[last_local_id];
    data.pop_back();

    loss[local_id] = loss[last_local_id];
    loss.pop_back();

    chi2_vec[local_id] = chi2_vec[last_local_id];
    chi2_vec.pop_back();

    active[local_id] = active[last_local_id];
    active.pop_back();

    // Next, fix ids
    const auto last_global_id = local_to_global_map[last_local_id];
    global_to_local_map[last_global_id] = local_id;
    local_to_global_map[local_id] = last_global_id;

    // Remove unused entries
    global_to_local_map.erase(id);
    local_to_global_map.pop_back();

    hm.release(id);
  }

  size_t add_factor(const std::array<size_t, N> &ids,
                    const ObservationType &obs, const S *precision_matrix,
                    const ConstraintDataType &constraint_data,
                    const LossType &loss_func) {

    const auto id = hm.get();
    const auto local_id = internal_count();

    global_to_local_map.insert({id, local_id});
    local_to_global_map.push_back(id);

    global_ids.insert(global_ids.end(), ids.begin(), ids.end());
    device_obs.push_back(obs);

    constexpr size_t precision_matrix_size = error_dim * error_dim;
    if (precision_matrix) {
      precision_matrices.resize(precision_matrices.size() +
                                precision_matrix_size);
      for (size_t i = 0; i < precision_matrix_size; i++) {
        precision_matrices[local_id * precision_matrix_size + i] =
            precision_matrix[i];
      }
    } else {
      // constexpr auto pmat = get_default_precision_matrix();
      precision_matrices.resize(precision_matrices.size() +
                                precision_matrix_size);

      for (size_t i = 0; i < precision_matrix_size; i++) {
        precision_matrices[local_id * precision_matrix_size + i] =
            default_precision_matrix[i];
      }
    }

    data.push_back(constraint_data);
    loss.push_back(loss_func);
    active.push_back(0);
    return id; // only global within this class (it's just an external id)
  }

  // TODO: Make this private later
  constexpr static std::array<S, error_dim * error_dim>
  get_default_precision_matrix() {
    constexpr size_t E = error_dim;
    return []() constexpr {
      std::array<S, E *E> pmat = {};
      for (size_t i = 0; i < E; i++) {
        pmat[i * E + i] = static_cast<S>(1.0);
      }
      return pmat;
    }
    ();
  }

  void set_active(size_t id, const uint8_t active_value) {
    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return;
    }

    constexpr uint8_t NOT_MSB = 0x7F; // 01111111

    auto local_id = global_to_local_map[id];
    active[local_id] =
        NOT_MSB & active_value; // we reserve the MSB for later use
  }

  void reset_active() {
    thrust::fill(thrust::device, active.begin(), active.end(), 0x1);
  }

  size_t internal_count() const { return global_ids.size() / N; }
  size_t active_count() const override { return _active_count; }

  void initialize_device_ids(const uint8_t optimization_level) override {
    // Map constraint ids to local ids

    // auto start = std::chrono::high_resolution_clock::now();

    // device_ids.resize(global_ids.size());
    host_ids.resize(global_ids.size());
    // prefetch_vector_on_host(device_ids, 0);
    for (size_t i = 0; i < global_ids.size(); i++) {
      host_ids[i] =
          vertex_descriptors[i % N]->get_global_map().at(global_ids[i]);
    }
    device_ids = host_ids;

    device_active = active;
    _active_count = build_active_indices(device_active, active_indices,
                                         internal_count(), optimization_level);

    // std::cout << "Internal count: " << internal_count() << std::endl;
    // std::cout << "Active count: " << _active_count << std::endl;

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Device id building time: " << elapsed.count() << " seconds"
    // << std::endl;
  }

  void to_device() override {

    // device_ids = host_ids;
    // device_obs = host_obs;
    chi2_vec.resize(internal_count());
    chi2_derivative.resize(internal_count());

    // prefetch everything
    int cuda_device = 0;
    constexpr cudaStream_t stream = 0;
    cudaGetDevice(&cuda_device);
    // prefetch_vector_on_device_async(device_ids, cuda_device, stream);
    prefetch_vector_on_device_async(device_obs, cuda_device, stream);
    prefetch_vector_on_device_async(chi2_vec, cuda_device, stream);
    prefetch_vector_on_device_async(data, cuda_device, stream);
    prefetch_vector_on_device_async(loss, cuda_device, stream);
    // std::cout << "Prefetching factor data to device" << std::endl;
    cudaDeviceSynchronize();
    // Resize and reset residuals
    residuals.resize(error_dim * internal_count());
    thrust::fill(residuals.begin(), residuals.end(), 0);
  }

  void link_factors(
      const std::array<BaseVertexDescriptor<T, S> *, N> &vertex_descriptors) {
    this->vertex_descriptors = vertex_descriptors;
  }

  template <std::size_t... I>
  VertexPointerPointerTuple get_vertices_impl(std::index_sequence<I...>) {
    // return std::make_tuple((static_cast<typename std::tuple_element<I,
    // std::tuple<VDTypes...>>::type*>(vertex_descriptors[I])->vertices())...);
    return std::make_tuple(
        (static_cast<typename std::tuple_element<I, VertexDescriptorTuple>::type
                         *>(vertex_descriptors[I])
             ->vertices())...);
  }

  // Return tuple of N vertex pointers from
  // this->vertex_descriptors[i]->vertices()
  VertexPointerPointerTuple get_vertices() {
    return get_vertices_impl(std::make_index_sequence<N>{});
  }

  static constexpr std::array<size_t, N> get_vertex_sizes() {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return std::array<size_t, N>{
          std::tuple_element_t<I, VertexDescriptorTuple>::dim...};
    }
    (std::make_index_sequence<N>{});
  }

  void initialize_jacobian_storage() override {
    for (size_t i = 0; i < N; i++) {
      if (store_jacobians() || !std::is_same_v<typename Traits::Differentiation,
                                               DifferentiationMode::Manual>) {
        jacobians[i].dimensions = {error_dim,
                                   vertex_descriptors[i]->dimension()};
        jacobians[i].data.resize(
            error_dim * vertex_descriptors[i]->dimension() * internal_count());
      }
    }
  }

  virtual size_t get_residual_size() const override {
    return error_dim * internal_count();
  }

  // TODO: Make this consider kernels and active edges
  virtual T chi2(GraphVisitor<T, S> &visitor) override {
    visitor.template compute_chi2(this);
    return thrust::reduce(thrust::device, chi2_vec.begin(), chi2_vec.end(),
                          static_cast<T>(0.0), thrust::plus<T>());
  }

  T chi2(const size_t id) const {
    auto it = global_to_local_map.find(id);
    if (it == global_to_local_map.end()) {
      std::cerr << "Factor with id " << id << " not found." << std::endl;
      return static_cast<T>(0.0);
    }
    return chi2_vec[it->second];
  }

  virtual void scale_jacobians(GraphVisitor<T, S> &visitor,
                               T *jacobian_scales) override {
    visitor.template scale_jacobians(this, jacobian_scales);
  }

  virtual bool use_autodiff() override {
    return use_autodiff_impl<typename Traits::Differentiation>();
  }

  virtual void set_jacobian_storage(const bool store) {
    _store_jacobians = store;
  }

  virtual bool store_jacobians() override { return _store_jacobians; }

  virtual bool supports_dynamic_jacobians() override {
    return std::is_same_v<typename Traits::Differentiation,
                          DifferentiationMode::Manual>;
  }
};

} // namespace glso