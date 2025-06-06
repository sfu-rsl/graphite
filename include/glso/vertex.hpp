#pragma once
#include <glso/common.hpp>
#include <glso/vector.hpp>
#include <glso/visitor.hpp>
#include <type_traits>

namespace glso {

template <typename, typename = void>
struct has_type_alias_State : std::false_type {};

template <typename T>
struct has_type_alias_State<T, std::void_t<typename T::State>>
    : std::true_type {};

template <typename T, typename Fallback, typename = void> struct get_State_or {
  using type = Fallback;
};

// Specialization: use T::State if it exists
template <typename T, typename Fallback>
struct get_State_or<T, Fallback, std::void_t<typename T::State>> {
  using type = typename T::State;
};

// Helper alias
template <typename T, typename Fallback>
using get_State_or_t = typename get_State_or<T, Fallback>::type;

template <typename VertexType, typename State, typename Traits, typename T>
__global__ void backup_state_kernel(VertexType **vertices, State *dst,
                                    const uint32_t *fixed,
                                    const size_t num_vertices) {

  const size_t vertex_id = get_thread_id();

  if (vertex_id >= num_vertices || is_fixed(fixed, vertex_id))
    return;
  if constexpr (has_type_alias_State<Traits>::value) {
    dst[vertex_id] = Traits::get_state(*vertices[vertex_id]);
  } else {
    dst[vertex_id] = *vertices[vertex_id];
  }
}

template <typename VertexType, typename State, typename Traits, typename T>
__global__ void set_state_kernel(VertexType **vertices, const State *src,
                                 const uint32_t *fixed,
                                 const size_t num_vertices) {

  const size_t vertex_id = get_thread_id();

  if (vertex_id >= num_vertices || is_fixed(fixed, vertex_id))
    return;

  if constexpr (has_type_alias_State<Traits>::value) {
    Traits::set_state(*vertices[vertex_id], src[vertex_id]);
  } else {
    *vertices[vertex_id] = src[vertex_id];
  }
}

template <typename T, typename S> class BaseVertexDescriptor {
public:
  using InvP = std::conditional_t<std::is_same<S, ghalf>::value, T, S>;

  virtual ~BaseVertexDescriptor(){};

  // virtual void update(const T* x, const T* delta) = 0;
  virtual void visit_update(GraphVisitor<T, S> &visitor, const T *delta_x,
                            T *jacobian_scales) = 0;
  virtual void visit_augment_block_diagonal(GraphVisitor<T, S> &visitor,
                                            InvP *block_diagonal, T mu) = 0;
  virtual void visit_apply_block_jacobi(GraphVisitor<T, S> &visitor, T *z,
                                        const T *r, InvP *block_diagonal) = 0;
  virtual size_t dimension() const = 0;
  virtual size_t count() const = 0;

  virtual void backup_parameters() = 0;
  virtual void restore_parameters() = 0;
  virtual void to_device() = 0;
  virtual void to_host() = 0;

  virtual const std::unordered_map<size_t, size_t> &get_global_map() const = 0;
  virtual const size_t *get_hessian_ids() const = 0;
  virtual void set_hessian_column(size_t global_id, size_t hessian_column) = 0;
  virtual bool is_fixed(const size_t id) const = 0;
  virtual const uint32_t *get_fixed_mask() const = 0;
};

template <typename T, typename S, typename VTraits>
class VertexDescriptor : public BaseVertexDescriptor<T, S> {
public:
  using InvP = std::conditional_t<std::is_same<S, ghalf>::value, T, S>;

  using Traits = class VTraits;

  using VertexType = typename Traits::Vertex;
  using State = get_State_or_t<Traits, VertexType>;

private:
  thrust::device_vector<VertexType *> x_device;
  thrust::host_vector<VertexType *> x_host;
  thrust::device_vector<State> backup_state;

public:
  // Mappings
  std::unordered_map<size_t, size_t> global_to_local_map;
  std::vector<size_t> local_to_global_map;
  thrust::host_vector<size_t> local_to_hessian_offsets;
  thrust::device_vector<size_t> hessian_ids;
  uninitialized_vector<uint32_t> fixed_mask;

  static constexpr size_t dim = Traits::dimension;

public:
  virtual ~VertexDescriptor(){};

  void visit_update(GraphVisitor<T, S> &visitor, const T *delta_x,
                    T *jacobian_scales) override {
    visitor.template apply_step(this, delta_x, jacobian_scales);
  }

  void visit_augment_block_diagonal(GraphVisitor<T, S> &visitor,
                                    InvP *block_diagonal, T mu) override {
    visitor.template augment_block_diagonal(this, block_diagonal, mu);
  }

  void visit_apply_block_jacobi(GraphVisitor<T, S> &visitor, T *z, const T *r,
                                InvP *block_diagonal) override {
    visitor.template apply_block_jacobi(this, z, r, block_diagonal);
  }

  virtual void to_device() override {
    x_device = x_host;
    hessian_ids = local_to_hessian_offsets;
  }

  virtual void to_host() override { x_host = x_device; }

  VertexType **vertices() { return x_device.data().get(); }

  virtual void backup_parameters() override {
    VertexType **vertices = x_device.data().get();

    const int num_vertices = static_cast<int>(count());
    const int num_threads = num_vertices;
    const int block_size = 256;
    const auto num_blocks = (num_threads + block_size - 1) / block_size;
    backup_state.resize(num_vertices);

    backup_state_kernel<VertexType, State, Traits, T>
        <<<num_blocks, block_size>>>(vertices, backup_state.data().get(),
                                     fixed_mask.data().get(), num_vertices);
  }

  virtual void restore_parameters() override {
    VertexType **vertices = x_device.data().get();

    const int num_vertices = static_cast<int>(count());
    const int num_threads = num_vertices;
    const int block_size = 256;
    const auto num_blocks = (num_threads + block_size - 1) / block_size;
    set_state_kernel<VertexType, State, Traits, T>
        <<<num_blocks, block_size>>>(vertices, backup_state.data().get(),
                                     fixed_mask.data().get(), num_vertices);
  }

  virtual size_t count() const override { return x_host.size(); }

  void reserve(size_t size) {
    x_host.reserve(size);
    global_to_local_map.reserve(size);
    local_to_global_map.reserve(size);
    local_to_hessian_offsets.reserve(size);
    fixed_mask.reserve((size + 31) / 32);
  }

  void remove_vertex(const size_t id) {
    if (count() == 0) {
      return;
    }

    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Vertex with id " << id << " not found." << std::endl;
      return;
    }

    const auto local_id = global_to_local_map[id];
    const auto last_index = x_host.size() - 1;

    // Swap the vertex to be removed with the last vertex
    std::swap(x_host[local_id], x_host[last_index]);
    std::swap(local_to_hessian_offsets[local_id],
              local_to_hessian_offsets[last_index]);

    // Update the global_to_local_map for the swapped vertex
    const auto last_global_id = local_to_global_map[last_index];
    global_to_local_map[last_global_id] = local_id;
    local_to_global_map[local_id] = last_global_id;

    // Remove the last vertex
    x_host.pop_back();
    local_to_hessian_offsets.pop_back();
    global_to_local_map.erase(id);
    local_to_global_map.pop_back();

    // Only need to update the fixed mask for the swapped vertex
    set_fixed(local_id, is_fixed(last_index));

    // Remove unused entry
    if (last_index % 32 == 0) {
      fixed_mask.pop_back();
    }
  }

  void replace_vertex(const size_t id, VertexType *vertex) {
    if (global_to_local_map.find(id) == global_to_local_map.end()) {
      std::cerr << "Vertex with id " << id << " not found." << std::endl;
      return;
    }

    const auto local_id = global_to_local_map[id];
    x_host[local_id] = vertex;
  }

  void add_vertex(const size_t id, VertexType *vertex,
                  const bool fixed = false) {
    // TODO: Find a better way to get the dimension
    // const auto dim = dynamic_cast<Derived<T>*>(this)->dimension();
    // x_host.insert(x_host.end(), value, value+dim);
    x_host.push_back(vertex);
    const auto local_id = x_host.size() - 1;
    global_to_local_map.insert({id, local_id});
    local_to_global_map.push_back(id);
    local_to_hessian_offsets.push_back(0); // Initialize to 0

    // Update fixed mask
    if ((count() + 31) / 32 > fixed_mask.size()) {
      fixed_mask.push_back(static_cast<uint32_t>(fixed));
    } else {
      fixed_mask.back() |= (static_cast<uint32_t>(fixed) << (count() % 32));
    }
  }

  void set_fixed(const size_t id, const bool fixed) {
    const auto local_id = global_to_local_map.at(id);
    if (fixed) {
      fixed_mask[local_id / 32] |= (1 << (local_id % 32));
    } else {
      fixed_mask[local_id / 32] &= ~(1 << (local_id % 32));
    }
  }

  bool is_fixed(const size_t id) const override {
    const auto local_id = global_to_local_map.at(id);
    return (fixed_mask[local_id / 32] & (1 << (local_id % 32))) != 0;
  }

  const uint32_t *get_fixed_mask() const override {
    return fixed_mask.data().get();
  }

  VertexType *get_vertex(const size_t id) {
    // std::array<T, D> vertex_data;
    const auto local_id = global_to_local_map.at(id);
    return x_host[local_id];
    // const T* data_ptr = x_host.data() + local_id * D;
    // std::copy(data_ptr, data_ptr + D, vertex_data.begin());
    // return vertex_data;
  }

  const std::unordered_map<size_t, size_t> &get_global_map() const override {
    return global_to_local_map;
  }

  size_t dimension() const override {
    return dim;
    // return Derived<T>::VertexType::dimension;
  }

  const size_t *get_hessian_ids() const override {
    return hessian_ids.data().get();
  }

  void set_hessian_column(size_t global_id, size_t hessian_column) {
    local_to_hessian_offsets[global_to_local_map.at(global_id)] =
        hessian_column;
  }
};

} // namespace glso