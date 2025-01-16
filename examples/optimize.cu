#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <glso/core.hpp>

namespace glso {
template <typename T>
class TestFactor : public AutoDiffFactorDescriptor<T, TestFactor> {
};
} // namespace glso

int main(void) {

    using namespace glso;

    // Create graph
    Graph graph;


    // Create vertices

    // Create edges
    constexpr size_t iterations = 10;
    Optimizer opt;
    std::cout << "Optimizing!" << std::endl;

    opt.optimize(&graph, iterations);
    std::cout << "Done optimizing!" << std::endl;

    return 0;
}
