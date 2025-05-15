#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <array>
#include <glso/core.hpp>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>


namespace glso {

template<typename T>
class Point {
    public:
        T x;
        T y;
        T z;

        Point() : x(0), y(0), z(0) {}

        Point(T x, T y, T z) : x(x), y(y), z(z) {}


        using State = Point<T>;
        constexpr static size_t dimension = 3;


        SOLVER_FUNC std::array<T, dimension> parameters() const {
            return {x, y, z};
        }
    
        SOLVER_FUNC void update(const T* delta) {
            x += delta[0];
            y += delta[1];
            z += delta[2];
        }
    
        SOLVER_FUNC State get_state() const {
            return *this;
        }
    
        SOLVER_FUNC void set_state(const State& state) {
            x = state.x;
            y = state.y;
            z = state.z;
        }


};

template<typename T>
class Camera {
    public:
        std::array<T, 9> params;

        Camera() : params{} {}
        Camera(const std::array<T, 9> & cam) : params(cam) {}

        // Camera(T x, T y) : x(x), y(y) {}


        using State = Camera<T>;
        constexpr static size_t dimension = 9;


        SOLVER_FUNC std::array<T, dimension> parameters() const {
            return params;
        }
    
        SOLVER_FUNC void update(const T* delta) {
            Eigen::Map<Eigen::Matrix<T, 9, 1>> p(params.data());
            Eigen::Map<const Eigen::Matrix<T, 9, 1>> d(delta);
            p += d;
        }
    
        SOLVER_FUNC State get_state() const {
            return *this;
        }
    
        SOLVER_FUNC void set_state(const State& state) {
            params = state.params;
        }

};

template<typename T>
class PointDescriptor: public VertexDescriptor<T, Point<T>, PointDescriptor> {};

template<typename T>
class CameraDescriptor: public VertexDescriptor<T, Camera<T>, CameraDescriptor> {};


using Observation2D = Eigen::Vector2d;
using ConstraintData = unsigned char;
template <typename T>
class ReprojectionError : public AutoDiffFactorDescriptor<T, 2, Observation2D, ConstraintData, DefaultLoss, ReprojectionError, CameraDescriptor<T>, PointDescriptor<T>> {
public:

    template <typename D, typename M>
    __device__ static void error(
        const D* camera,
        const D* point, 
        const M* obs, 
        D* error, 
        const std::tuple<Camera<T>*, Point<T>*> & vertices, 
        const ConstraintData* data) {

            // Compute the reprojection error according to the camera model
            
            /*
                https://grail.cs.washington.edu/projects/bal/
                Camera Model
                We use a pinhole camera model; the parameters we estimate for each camera area rotation R, 
                a translation t, a focal length f and two radial distortion parameters k1 and k2. 
                The formula for projecting a 3D point X into a camera R,t,f,k1,k2 is:

                P  =  R * X + t       (conversion from world to camera coordinates)
                p  = -P / P.z         (perspective division)
                p' =  f * r(p) * p    (conversion to pixel coordinates)

                where P.z is the third (z) coordinate of P. 
                In the last equation, r(p) is a function that computes a scaling factor to undo the radial distortion:

                r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.

                This gives a projection in pixels, where the origin of the image is the center of the image, 
                the positive x-axis points right, and the positive y-axis points up (in addition, in the camera coordinate system, 
                the positive z-axis points backwards, so the camera is looking down the negative z-axis, as in OpenGL). 
                        
            */

            Eigen::Map<const Eigen::Matrix<D, 3, 1>> X(point);
            Eigen::Map<const Eigen::Matrix<D, 9, 1>> cam(camera);
            Eigen::Map<const Eigen::Matrix<T, 2, 1>> observation(obs->data());
            Eigen::Matrix<D, 3, 1> P;

            // Rodrigues formula for rotation vector (angle-axis) with Taylor expansion for small theta
            // Adapted from g2o bal example
            Eigen::Matrix<D, 3, 1> rvec = cam.template head<3>();
            D theta = rvec.norm();

            if (theta > D(0)) {
                Eigen::Matrix<D, 3, 1> axis = rvec / theta;
                D cth = cos(theta);
                D sth = sin(theta);

                Eigen::Matrix<D, 3, 1> axx = axis.cross(X);
                D adx = axis.dot(X);

                P = X*cth + axx * sth + axis * adx * (D(1) - cth);

               
            } else {
                auto rxx = rvec.cross(X);
                P = X + rxx + D(0.5) * rvec.cross(rxx);
            }

            // Translate
            Eigen::Matrix<D, 3, 1> t = cam.template segment<3>(3);
            P += t;

            // Perspective division
            Eigen::Matrix<D, 2, 1> p = -P.template head<2>() / P(2);

            // Radial distortion
            D f = cam(6);
            D k1 = cam(7);
            D k2 = cam(8);

            D r2 = p.squaredNorm();
            D radial_distortion = D(1.0) + k1 * r2 + k2 * r2 * r2;

            // Project to pixel coordinates and compute reprojection error
            Eigen::Matrix<D, 2, 1> reprojection_error = f * radial_distortion * p - observation.template cast<D>();

            // Assign the error
            error[0] = reprojection_error(0);
            error[1] = reprojection_error(1);

            
        }
};
} // namespace glso

int main(void) {

    using namespace glso;

    // std::string file_path = "../data/bal/problem-16-22106-pre.txt";
    // std::string file_path = "../data/bal/problem-21-11315-pre.txt";
        std::string file_path = "../data/bal/problem-257-65132-pre.txt";


    initialize_cuda();

    // Create graph
    Graph<double> graph;

    size_t num_points = 0;
    size_t num_cameras = 0;
    size_t num_observations = 0;

    // Open the file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return -1;
    }

    // Read the number of cameras, points, and observations
    file >> num_cameras >> num_points >> num_observations;
    std::cout << "Number of cameras: " << num_cameras << std::endl;
    std::cout << "Number of points: " << num_points << std::endl;
    std::cout << "Number of observations: " << num_observations << std::endl;

    thrust::universal_vector<Point<double>> points(num_points);
    thrust::universal_vector<Camera<double>> cameras(num_cameras);

    // Create vertices
    auto point_desc = new PointDescriptor<double>();
    point_desc->reserve(num_points);
    graph.add_vertex_descriptor(point_desc);

    auto camera_desc = new CameraDescriptor<double>();
    camera_desc->reserve(num_cameras);
    graph.add_vertex_descriptor(camera_desc);

    

    // Create edges
    auto r_desc = graph.add_factor_descriptor<ReprojectionError<double>>(camera_desc, point_desc);
    r_desc->reserve(num_observations);

    

    const auto loss = DefaultLoss<double, 2>();
    Eigen::Matrix2d precision_matrix = Eigen::Matrix2d::Identity();


    auto start = std::chrono::steady_clock::now();

    // Read observations and create constraints
    for (size_t i = 0; i < num_observations; ++i) {
        size_t camera_idx, point_idx;
        double x, y;

        // Read observation data
        file >> camera_idx >> point_idx >> x >> y;

        // Store the observation
        const Observation2D obs(x, y);

        // Add constraint to the graph
        r_desc->add_factor(
            {camera_idx, point_idx},
            obs,
            precision_matrix.data(),
            0,
            loss
        );
    }
    std::cout << "Adding constraints took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " seconds." << std::endl;

    start = std::chrono::steady_clock::now();
    // Create all camera vertices
    for (size_t i = 0; i < num_cameras; ++i) {
        std::array<double, 9> camera_params;
        for (size_t j = 0; j < 9; ++j) {
            file >> camera_params[j];
        }
        cameras[i] = Camera<double>(camera_params);
        camera_desc->add_vertex(i, &cameras[i]);
    }

    std::cout << "Adding cameras took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " seconds." << std::endl;

    start = std::chrono::steady_clock::now();
    // Create all point vertices
    for (size_t i = 0; i < num_points; ++i) {
        double point_params[3];
        for (size_t j = 0; j < 3; ++j) {
            file >> point_params[j];
        }
        points[i] = Point<double>(point_params[0], point_params[1], point_params[2]);
        point_desc->add_vertex(i, &points[i]);
    }
    std::cout << "Adding points took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " seconds." << std::endl;
    file.close();
    


    // Optimize
    constexpr size_t iterations = 50;
    Optimizer opt;
    std::cout << "Graph built with " << num_cameras << " cameras, " << num_points << " points, and " << r_desc->count() << " observations." << std::endl;
    std::cout << "Optimizing!" << std::endl;

    start = std::chrono::steady_clock::now();
    opt.optimize(&graph, iterations, 1e4);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Optimization took " << elapsed.count() << " seconds." << std::endl;
    std::cout << "MSE: " << graph.chi2() / num_observations << std::endl;

    return 0;
}
