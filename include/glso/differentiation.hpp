#pragma once
#include <type_traits>
namespace glso {

struct DifferentiationMode {
  struct Auto {};
  struct Manual {};
};

template <typename DiffMode> constexpr bool use_autodiff_impl() {
  return false;
}

template <typename F> constexpr bool is_analytical() {
  return std::is_same_v<typename F::Traits::Differentiation, DifferentiationMode::Manual>;
}

template <> constexpr bool use_autodiff_impl<DifferentiationMode::Auto>() {
  return true;
}

}