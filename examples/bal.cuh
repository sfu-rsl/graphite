/// @file bal.cuh
#pragma once
#include "reprojection_error.cuh"
#include <Eigen/Core>
#include <graphite/common.hpp>
#include <graphite/factor.hpp>
#include <graphite/vertex.hpp>

namespace graphite {

template <typename T> using Point = Eigen::Matrix<T, 3, 1>;

template <typename T> using Camera = Eigen::Matrix<T, 9, 1>;

template <typename T> struct PointTraits {
  static constexpr size_t dimension = 3;
  using Vertex = Point<T>;

  template <typename P>
  d_fn static void parameters(const Vertex &vertex, P *parameters) {
    Eigen::Map<Eigen::Matrix<P, dimension, 1>> params_map(parameters);
    params_map = vertex.template cast<P>();
  }

  d_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }
};

template <typename T> struct CameraTraits {
  static constexpr size_t dimension = 9;
  using State = Camera<T>; // State can be optionally defined
  using Vertex = Camera<T>;

  template <typename P>
  d_fn static void parameters(const Vertex &vertex, P *parameters) {
    Eigen::Map<Eigen::Matrix<P, dimension, 1>> params_map(parameters);
    params_map = vertex.template cast<P>();
  }

  d_fn static void update(Vertex &vertex, const T *delta) {
    Eigen::Map<const Eigen::Matrix<T, dimension, 1>> d(delta);
    vertex += d;
  }

  // Defining the state requires custom setters and getters
  d_fn static State get_state(const Vertex &vertex) { return vertex; }

  d_fn static void set_state(Vertex &vertex, const State &state) {
    vertex = state;
  }
};

template <typename T, typename S>
using PointDescriptor = VertexDescriptor<T, S, PointTraits<T>>;

template <typename T, typename S>
using CameraDescriptor = VertexDescriptor<T, S, CameraTraits<T>>;

template <typename T, typename S> struct ReprojectionErrorTraits {
  static constexpr size_t dimension = 2;
  using VertexDescriptors =
      std::tuple<CameraDescriptor<T, S>, PointDescriptor<T, S>>;
  using Observation = Eigen::Matrix<T, dimension, 1>;
  using Data = Empty;
  using Loss = DefaultLoss<T, dimension>;
  // using Differentiation = DifferentiationMode::Auto;
  using Differentiation = DifferentiationMode::Manual;

  // You can pass in vertex references (class references), parameter blocks
  // (pointer to 1D parameters), or both. The framework will automatically call
  // your function with the correct arguments.
  template <typename D>
  d_fn static void error(const D *camera, const D *point,
                         const Observation &obs, D *error) {
    bal_reprojection_error_simple<D, Observation, T>(camera, point, &obs,
                                                     error);
  }

  template <typename D, size_t I>
  d_fn static void jacobian(const Camera<T> &camera, const Point<T> &point,
                            const Observation &obs, D *jacobian) {
    bal_jacobian_simple<T, D, I>(camera.data(), point.data(), &obs, jacobian);
  }
};

template <typename T, typename S>
using ReprojectionError = FactorDescriptor<T, S, ReprojectionErrorTraits<T, S>>;

} // namespace graphite