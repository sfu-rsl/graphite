#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <array>
#include <glso/core.hpp>
#include <random>

namespace glso {

template<typename T>
class Point: public VertexDescriptor<T, 2, Point> {
    public:

    __device__ static void update(T* x, const T* delta) {
        x[0] += delta[0]; 
        x[1] += delta[1];
    }

};

template <typename T>
class CircleFactor : public AutoDiffFactorDescriptor<T, 1, 1, CircleFactor, Point<T>> {
public:

    template <typename D>
    __device__ static void error(const D* point, const D* obs, D* error) {
        D x = point[0];
        D y = point[1];
        D r = obs[0];

        error[0] = x*x + y*y - r*r;
    }
};
} // namespace glso

int main(void) {

    using namespace glso;

    initialize_cuda();

    // Create graph
    Graph<double> graph;


    // Create vertices
    Point<double>* point_desc = new Point<double>();
    graph.add_vertex_descriptor(point_desc);

    const size_t num_vertices = 5;

    for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        double point[2] = {4.1, 3.8};
        point_desc->add_vertex(vertex_id, point);
    }

    // Create edges
    auto factor_desc = graph.add_factor_descriptor<CircleFactor<double>>(point_desc);

    for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        const double radius = 4.0;
        factor_desc->add_factor({vertex_id}, {radius}, nullptr);
    }

    // Optimize
    constexpr size_t iterations = 5;
    Optimizer opt;
    std::cout << "Graph built with " << num_vertices << " vertices and " << factor_desc->count() << " factors." << std::endl;
    std::cout << "Optimizing!" << std::endl;

    opt.optimize(&graph, iterations);
    std::cout << "Done optimizing!" << std::endl;


    return 0;
}
