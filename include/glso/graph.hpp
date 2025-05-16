#pragma once
#include <glso/visitor.hpp>
#include <glso/factor.hpp>
#include <glso/vertex.hpp>
#include <glso/solver.hpp>
#include <limits>

namespace glso {

    class BlockCoordinates {
        public:
        size_t row;
        size_t col;
    };

    template<typename T>
    class HessianBlocks {
        public:
    
        std::pair<size_t, size_t> dimensions;
        size_t num_blocks;
        thrust::device_vector<T> data;
        thrust::device_vector<BlockCoordinates> block_coordinates;


        void resize(size_t num_blocks, size_t rows, size_t cols) {
            dimensions = {rows, cols};
            this->num_blocks = num_blocks;
            data.resize(rows*cols*num_blocks);
            block_coordinates.resize(num_blocks);
        }
    
    };

template<typename T=double>
class Graph {

    private:

    GraphVisitor<T> visitor;

    std::vector<BaseVertexDescriptor<T>*> vertex_descriptors;
    std::vector<BaseFactorDescriptor<T>*> factor_descriptors;

    // std::unordered_map<std::pair<size_t, size_t>, std::shared_ptr<JacobianStorage<T>>> jacobians;
    // std::unordered_map<size_t, std::pair<size_t, size_t>> hessian_to_local_map;
    // std::unordered_map<size_t, size_t> hessian_offset;
    // Solver buffers
    thrust::device_vector<T> delta_x;
    thrust::device_vector<T> b;
    thrust::device_vector<T> jacobian_scales;

    // thrust::device_vector<T> error;

    T damping_factor;

    // Make this modular later
    PCGSolver<T> solver;

    public:

    Graph(): visitor(delta_x, b) {
        damping_factor = 1e-2;
    }

    thrust::device_vector<T>& get_delta_x() {
        return delta_x;
    }
    thrust::device_vector<T>& get_b() {
        return b;
    }

    void add_vertex_descriptor(BaseVertexDescriptor<T>* descriptor) {
        vertex_descriptors.push_back(descriptor);
    }

    template<typename F, typename ... Ts>
    F* add_factor_descriptor(Ts ... vertices) {

        // Link factor to vertices
        auto factor = new F();
        factor->link_factors({vertices...});

        factor_descriptors.push_back(factor);

        return factor;

    }

    void set_damping_factor(T factor) {
        damping_factor = factor;
    }

    T get_damping_factor() {
        return damping_factor;
    }

    bool initialize_optimization() {

        // For each vertex descriptor, take global to local id mapping and transform it into a Hessian 
        // column to local id mapping.
        
        std::vector<std::pair<size_t, std::pair<size_t, size_t>>> global_to_local_combined;

        for (size_t i = 0; i < vertex_descriptors.size(); ++i) {
            const auto& map = vertex_descriptors[i]->get_global_map();
            for (const auto& entry : map) {
            global_to_local_combined.push_back({entry.first, {i, entry.second}});
            }
        }
        
        // Sort the combined list by global ID
        std::sort(global_to_local_combined.begin(), global_to_local_combined.end(), 
            [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Assign Hessian columns to local indices
        // hessian_to_local_map.clear();
        // hessian_offset.clear();
        size_t hessian_column = 0;
        size_t block_column = 0;
        // size_t offset = 0;
        for (const auto& entry : global_to_local_combined) {
            // hessian_to_local_map.insert({hessian_column, entry.second});
            // hessian_offset.insert({hessian_column, offset});
            if (!vertex_descriptors[entry.second.first]->is_fixed(entry.first)) {
                vertex_descriptors[entry.second.first]->set_hessian_column(entry.first, hessian_column);
                hessian_column += vertex_descriptors[entry.second.first]->dimension();
                block_column++;
            }
        }
        // // Assign global offset into error vector for each factor
        // size_t error_offset = 0;
        // for (auto & desc: factor_descriptors) {
        //     error_offset += desc->set_error_offset(offset);
        // }

        // Copy vertex values to device
        for (auto & desc: vertex_descriptors) {
            desc->to_device();
        }

        // Copy factors to device
        for (auto & desc: factor_descriptors) {
            desc->to_device();
        }

        // Initialize Jacobian storage
        for (auto & f: factor_descriptors) {
            f->initialize_jacobian_storage();
        }

        return true;
    }

    bool build_structure() {
        // Allocate storage for solver vectors
        size_t size_x = 0;
        for (const auto & desc: vertex_descriptors) {
            size_x += desc->dimension()*desc->count(); 
        }

        // size_t size_error = 0;
        // for (const auto & desc: factor_descriptors) {
        //     size_error += desc->error_dimension()(0)*desc->count();
        // }

        delta_x.resize(size_x);
        b.resize(size_x);
        jacobian_scales.resize(size_x);
        // error.resize(size_error);

        // // Build Hessian structure
        // std::unordered_map<std::pair<size_t, size_t>, size_t> block_counts;
        // for (auto & desc: factor_descriptors) {
        //     // const auto num_factors = desc->count();
        //     desc->count_blocks(block_counts);
        // }
        

        return true;
    }

    void compute_error() {
        for (auto & factor: factor_descriptors) {
            factor->visit_error(visitor); // TODO: Make non-autodiff version
        }
        cudaDeviceSynchronize();
    }

    T chi2() {
        T chi2 = 0;
        for (auto & factor: factor_descriptors) {
            chi2 += factor->chi2(visitor);
        }
        return chi2;
    }

    void linearize() {
        // clear delta_x and b
        thrust::fill(delta_x.begin(), delta_x.end(), 0);
        thrust::fill(b.begin(), b.end(), 0);
        thrust::fill(jacobian_scales.begin(), jacobian_scales.end(), 0);


        for (auto & factor: factor_descriptors) {
            // compute error
            if (factor->use_autodiff()) {
                factor->visit_error_autodiff(visitor);
            }
            else {
                factor->visit_error(visitor);
                // manually compute Jacobians
                throw std::runtime_error("Manual Jacobian computation not implemented");
            }
        }

        cudaDeviceSynchronize();

        // Compute chi2
        chi2();

        // Compute Jacobian scale
        constexpr bool scale_jacobians = true;
        if (scale_jacobians) {
            for (auto & factor: factor_descriptors) {
                factor->visit_scalar_diagonal(visitor, jacobian_scales.data().get());
            }
        }
        else {
            thrust::fill(jacobian_scales.begin(), jacobian_scales.end(), 1.0);
        }

        cudaDeviceSynchronize();
        thrust::transform(
            jacobian_scales.begin(), jacobian_scales.end(), jacobian_scales.begin(),
            [] __device__ (T value) {
                // return 1.0 / (1.0 + sqrt(value));
                return 1.0 / (std::numeric_limits<T>::epsilon() + sqrt(value));
            }
        );

        // Scale Jacobians
        for (auto & factor: factor_descriptors) {
            factor->scale_jacobians(visitor, jacobian_scales.data().get());
        }
        cudaDeviceSynchronize();


        // Calculate b=J^T * r
        for (auto & fd: factor_descriptors) {
            fd->visit_b(visitor);
        }

        cudaDeviceSynchronize();

    }

    bool compute_step() {
        // Solve for delta_x
        thrust::fill(delta_x.begin(), delta_x.end(), 0);

        // Print delta_x
        // std::cout << "Delta x before solve: " << std::endl;
        // for (size_t i = 0; i < delta_x.size(); i++) {
        //     std::cout << delta_x[i] << " ";
        // }
        // std::cout << std::endl;

        // Print b
        // std::cout << "b before solve: " << std::endl;
        // for (size_t i = 0; i < b.size(); i++) {
        //     std::cout << b[i] << " ";
        // }
        // std::cout << std::endl;
        
        // auto start = std::chrono::steady_clock::now();

        solver.solve(
            visitor, 
            vertex_descriptors, 
            factor_descriptors,
            b.data().get(),
            delta_x.data().get(), 
            delta_x.size(), damping_factor, 50, 1e-6);

        // auto end = std::chrono::steady_clock::now();
        // std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Solver execution time: " << elapsed.count() << " seconds" << std::endl;

        // Print delta_x after solve
        // std::cout << "Delta x after solve: " << std::endl;
        // for (size_t i = 0; i < delta_x.size(); i++) {
        //     std::cout << delta_x[i] << " ";
        // }
        // std::cout << std::endl;

        // // If everything was okay, scale the step
        // thrust::transform(
        //     delta_x.begin(), delta_x.end(), jacobian_scales.begin(), delta_x.begin(),
        //     [] __device__ (T dx, T scale) {
        //         return dx * scale;
        //     }
        // );

        return true;
    }

    void apply_step() {
        for (auto & desc: vertex_descriptors) {
            desc->visit_update(visitor, jacobian_scales.data().get());
        }
        cudaDeviceSynchronize();
    }

    void backup_parameters() {

        for (const auto & desc: vertex_descriptors) {
            desc->backup_parameters();
        }

        cudaDeviceSynchronize();
    }

    void revert_parameters() {

        for (auto & desc: vertex_descriptors) {
            desc->restore_parameters();
        }

        cudaDeviceSynchronize();

    }

    void to_host() {
        for (auto & desc: vertex_descriptors) {
            desc->to_host();
        }
    }

};

}