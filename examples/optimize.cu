#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <glso/core.hpp>

namespace glso {
// template <typename T>
// class TestFactor : public AutoDiffFactorDescriptor<T, TestFactor> {
// };

template<typename T>
class Radius: public VertexDescriptor<T, Radius> {
    public:

    static void update(T* x, const T* delta) {
        x[0] += delta[0]; 
    }

    size_t dimension() const override {
        return 1;
    }
};

template <typename T>
class CircleFactor : public AutoDiffFactorDescriptor<T, CircleFactor> {
public:
    static void error_func(const T** vertices, const T* obs, T* error) {
        T r = vertices[0][0];
        T x = obs[0];
        T y = obs[1];
        error[0] = x*x + y*y - r*r;
    }
};
} // namespace glso

int main(void) {

    using namespace glso;

    // Create graph
    Graph<double> graph;


    // Create vertices
    Radius<double>* radius = new Radius<double>();
    double r = 5.0;
    const auto vertex_id = radius->add_vertex(&r);
    graph.add_vertex_descriptor(radius);

    // Create edges
    CircleFactor<double> * f = new CircleFactor<double>();
    graph.add_factor_descriptor(f);
    // Optimize
    constexpr size_t iterations = 10;
    Optimizer opt;
    std::cout << "Optimizing!" << std::endl;

    opt.optimize(&graph, iterations);
    std::cout << "Done optimizing!" << std::endl;

    return 0;
}
