#pragma once

namespace graphite {

template <typename T> class Op {

public:
  virtual ~Op(){};

  // template <typename F>
  // void apply_to(F* factor) {
  //     std::cout << "Op applied!" << std::endl;
  // }
};

template <typename T, typename Derived> class OpImpl : Op<T> {
public:
  template <typename F> void apply_to(F *factor) {
    std::cout << "OpImpl applied!" << std::endl;
  }
};

} // namespace graphite