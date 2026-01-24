#pragma once
#include <graphite/ops/common.hpp>
namespace graphite {

namespace ops {

template <typename T, typename S, size_t I, size_t N, typename M, size_t E,
          typename F, typename VT, std::size_t... Is>
__global__ void compute_error_kernel_autodiff(
    const M *obs, T *error,
    const typename F::ConstraintDataType *constraint_data,
    const size_t *active_ids, const size_t *ids, const size_t *hessian_ids,
    const size_t num_threads, VT args, S *jacs, const uint8_t *active_state,
    std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  const auto factor_id = active_ids[idx / vertex_sizes[I]];
  const auto vertex_id = ids[factor_id * N + I];

  // printf("CEAD: Thread %d, Vertex %d, Factor %d\n", idx, vertex_id,
  // factor_id);
  using G = std::conditional_t<is_low_precision<S>::value, T, S>;
  const M *local_obs = obs + factor_id;
  Dual<T, G> local_error[E];
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto v = cuda::std::make_tuple(std::array<Dual<T, G>, vertex_sizes[Is]>{}...);

  auto vargs =
      std::make_tuple((*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  auto copy_vertices = [&v, &vertex_sizes, &vargs](auto &&...ptrs) {
    ((std::tuple_element<Is, typename F::Traits::VertexDescriptors>::type::
          Traits::parameters(*std::get<Is>(vargs),
                             cuda::std::get<Is>(v).data())),
     ...);
  };

  std::apply(copy_vertices, vargs);

  cuda::std::get<I>(v)[idx % vertex_sizes[I]].dual = static_cast<G>(1);

  F::Traits::error(cuda::std::get<Is>(v).data()..., local_obs, local_error,
                   vargs, local_data);

  constexpr auto j_size = vertex_sizes[I] * E;
  // constexpr auto col_offset = I*E;
  const auto col_offset = (idx % vertex_sizes[I]) * E;
  // Store column-major Jacobian blocks.
  // Write one scalar column (length E) of the Jacobian matrix.
  // TODO: make sure this only writes to each location once
  // The Jacobian is stored as E x vertex_size in col major

  // Only run once per factor - this check won't work for multiple kernel
  // launches
  // TODO: make sure this only writes to each location once for the error
  if (idx % vertex_sizes[I] == 0) {
#pragma unroll
    for (size_t i = 0; i < E; ++i) {
      error[factor_id * E + i] = local_error[i].real;
    }
  }

  // This should write one Jacobian column per dimension per vertex for each
  // factor We only need a Jacobian if the vertex is not fixed
  if (!is_vertex_active(active_state, vertex_id)) {
    return;
  }

  if constexpr (std::is_same<S, __half>::value) {
// Need to clamp range
#pragma unroll
    for (size_t i = 0; i < E; ++i) {
      jacs[j_size * factor_id + col_offset + i] =
          static_cast<S>(std::clamp(local_error[i].dual, -65504.0f, 65504.0f));
    }
  } else {
#pragma unroll
    for (size_t i = 0; i < E; ++i) {
      jacs[j_size * factor_id + col_offset + i] = local_error[i].dual;
    }
  }
}

template <typename T, typename S, typename F, typename VT, std::size_t... Is>
void launch_kernel_autodiff(
    F *f, std::array<const size_t *, F::get_num_vertices()> &hessian_ids,
    VT &verts, std::array<S *, F::get_num_vertices()> &jacs,
    const size_t num_factors, StreamPool &streams, std::index_sequence<Is...>) {
  (([&] {
     constexpr auto num_vertices = F::get_num_vertices();
     const auto num_threads = num_factors * F::get_vertex_sizes()[Is];
     // std::cout << "Launching autodiff kernel" << std::endl;
     // std::cout << "Num threads: " << num_threads << std::endl;
     size_t threads_per_block = 256;
     size_t num_blocks =
         (num_threads + threads_per_block - 1) / threads_per_block;

     // std::cout << "Checking obs ptr: " << f->device_obs.data().get() <<
     // std::endl; std::cout << "Checking residual ptr: " <<
     // f->residuals.data().get() << std::endl; std::cout << "Checking ids
     // ptr: " << f->device_ids.data().get() << std::endl;

     compute_error_kernel_autodiff<T, S, Is, num_vertices,
                                   typename F::ObservationType, F::error_dim, F,
                                   typename F::VertexPointerPointerTuple>
         <<<num_blocks, threads_per_block, 0, streams.select(Is)>>>(
             f->device_obs.data().get(), f->residuals.data().get(),
             f->data.data().get(), f->active_indices.data().get(),
             f->device_ids.data().get(), hessian_ids[Is], num_threads, verts,
             jacs[Is], f->vertex_descriptors[Is]->get_active_state(),
             std::make_index_sequence<num_vertices>{});
   }()),
   ...);
}

template <typename T, typename S, typename F>
void compute_error_autodiff(F *f, StreamPool &streams) {
  // Assume autodiff

  // Then for each vertex, we need to compute the error
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // At this point all necessary data should be on the GPU
  // std::array<T*, num_vertices> verts;
  auto verts = f->get_vertices();
  std::array<S *, num_vertices> jacs;
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    // verts[i] = f->vertex_descriptors[i]->vertices();
    jacs[i] = f->jacobians[i].data.data().get();
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();

    // Important: Must clear Jacobian storage
    thrust::fill(thrust::cuda::par_nosync.on(streams.select(i)),
                 f->jacobians[i].data.begin(), f->jacobians[i].data.end(),
                 static_cast<S>(0));
  }

  const auto num_factors = f->active_count();

  if constexpr (!is_analytical<F>()) {
    launch_kernel_autodiff(f, hessian_ids, verts, jacs, num_factors, streams,
                           std::make_index_sequence<num_vertices>{});
  }
  streams.sync_n(num_vertices);
}

// TODO: Make this more efficient and see if code can be shared with the
// autodiff kernel
template <typename T, size_t N, typename M, size_t E, typename F, typename VT,
          std::size_t... Is>
__global__ void
compute_error_kernel(const M *obs, T *error,
                     const typename F::ConstraintDataType *constraint_data,
                     const size_t *active_ids, const size_t *ids,
                     const size_t num_threads, VT args,
                     std::index_sequence<Is...>) {
  const size_t idx = get_thread_id();

  if (idx >= num_threads) {
    return;
  }

  constexpr auto vertex_sizes = F::get_vertex_sizes();
  const auto factor_id = active_ids[idx];

  const M *local_obs = obs + factor_id;
  T *local_error = error + factor_id * E;
  const typename F::ConstraintDataType *local_data =
      constraint_data + factor_id;

  auto v = cuda::std::make_tuple(std::array<T, vertex_sizes[Is]>{}...);

  auto vargs =
      std::make_tuple((*(std::get<Is>(args) + ids[factor_id * N + Is]))...);

  auto copy_vertices = [&v, &vertex_sizes, &vargs](auto &&...ptrs) {
    ((std::tuple_element<Is, typename F::Traits::VertexDescriptors>::type::
          Traits::parameters(*std::get<Is>(vargs),
                             cuda::std::get<Is>(v).data())),
     ...);
  };

  std::apply(copy_vertices, vargs);

  F::Traits::error(cuda::std::get<Is>(v).data()..., local_obs, local_error,
                   vargs, local_data);

#pragma unroll
  for (size_t i = 0; i < E; ++i) {
    error[factor_id * E + i] = local_error[i];
  }
}

template <typename T, typename F> void compute_error(F *f) {
  // Then for each vertex, we need to compute the error
  constexpr auto num_vertices = F::get_num_vertices();
  constexpr auto vertex_sizes = F::get_vertex_sizes();

  // At this point all necessary data should be on the GPU
  auto verts = f->get_vertices();
  std::array<const size_t *, num_vertices> hessian_ids;
  for (int i = 0; i < num_vertices; i++) {
    hessian_ids[i] = f->vertex_descriptors[i]->get_hessian_ids();
  }

  constexpr auto error_dim = F::error_dim;
  const auto num_factors = f->active_count();

  const auto num_threads = num_factors;
  size_t threads_per_block = 256;
  size_t num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  compute_error_kernel<T, num_vertices, typename F::ObservationType,
                       F::error_dim, F, typename F::VertexPointerPointerTuple>
      <<<num_blocks, threads_per_block>>>(
          f->device_obs.data().get(), f->residuals.data().get(),
          f->data.data().get(), f->active_indices.data().get(),
          f->device_ids.data().get(), num_threads, verts,
          std::make_index_sequence<num_vertices>{});

  cudaStreamSynchronize(0);
}

} // namespace ops
} // namespace graphite