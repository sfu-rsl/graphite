#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <array>
#include <glso/core.hpp>
#include <random>

namespace glso {

template<typename T>
class Point {
    public:
        T x;
        T y;

        Point() : x(0), y(0) {}

        Point(T x, T y) : x(x), y(y) {}

        __host__ __device__ std::array<T, 2> parameters() const {
            return std::array<T, 2>{x, y};
        }

};

// template<typename T>
// class Point: public BaseVertex<T, 2>{
//     private:
//     T x;
//     T y;

//     public:
 
//     Point() : x(0), y(0) {}

//     Point(T x, T y): x(x), y(y) {}

//     __host__ __device__ virtual void update(const T* delta) override {
//         x += delta[0]; 
//         y += delta[1];
//     }

//     __host__ __device__ virtual std::array<T, 2> params() const override {
//         // out[0] = x;
//         // out[1] = y;
//         return std::array<T, 2>{x, y};
//     }

//     __host__ __device__ virtual void set_params(const T* params) override {
//         x = params[0];
//         y = params[1];
//     }

// };

// template<typename T>
// class Point: public VertexDescriptor<T, 2, Point> {
//     public:

//     __device__ static void update(T* x, const T* delta) {
//         x[0] += delta[0]; 
//         x[1] += delta[1];
//     }

// };

template<typename T>
class PointSet: public VertexDescriptor<T, 2, Point<T>, Point<T>, PointSet> {
    public:

    // using VertexType = Point<T>;

    __host__ __device__ static void update(Point<T> & vertex, const T* delta) {
        vertex.x += delta[0];
        vertex.y += delta[1];
    }

    // __host__ __device__ static std::array<T, 2> parameters(Point<T>& vertex) {
    //     return std::array<T, 2>{vertex.x, vertex.y};
    // }

    __host__ __device__ static Point<T> get_state(const Point<T>& vertex) {
        return vertex;
    }

    __host__ __device__ static void set_state(Point<T>& vertex, const Point<T>& state) {
        vertex = state;
    }

};

template <typename T>
class CircleFactor : public AutoDiffFactorDescriptor<T, 1, 1, CircleFactor, PointSet<T>> {
public:

    template <typename D>
    __device__ static void error(const D* point, const D* obs, D* error, const std::tuple<Point<T>*> & vertices) {
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
    PointSet<double>* points = new PointSet<double>();
    graph.add_vertex_descriptor(points);

    const size_t num_vertices = 5;
    double center[2] = {0.0, 0.0};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 2 * M_PI);

    const double radius = 4.0;
    const double sigma = 0.3;

    std::normal_distribution<double> n1(0.0, sigma);
    std::normal_distribution<double> n2(0.0, sigma);

    for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        double angle = dist(gen);
        double point[2] = {center[0] + radius * cos(angle), center[1] + radius * sin(angle)};
        

        point[0] += n1(gen);
        point[1] += n2(gen);

        std::cout << "Adding point " << vertex_id << "=(" << point[0] << ", " << point[1] << ") with radius=" << sqrt(point[0]*point[0] + point[1]*point[1]) << std::endl;
        points->add_vertex(vertex_id, Point(point[0], point[1]));
    }

    // Create edges
    auto factor_desc = graph.add_factor_descriptor<CircleFactor<double>>(points);

    for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        factor_desc->add_factor({vertex_id}, {radius}, nullptr);
    }

    // Optimize
    constexpr size_t iterations = 100;
    Optimizer opt;
    std::cout << "Graph built with " << num_vertices << " vertices and " << factor_desc->count() << " factors." << std::endl;
    std::cout << "Optimizing!" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    opt.optimize(&graph, iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Optimization took " << elapsed.count() << " seconds." << std::endl;

    // Read back optimized values
    for (size_t vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        const auto point = points->get_vertex(vertex_id);
        const auto [x, y] = point;
        
        std::cout << "Optimized point " << vertex_id << "=(" << x << ", " << y << ") with radius=" << sqrt(x*x + y*y) << std::endl;
    }


    return 0;
}
