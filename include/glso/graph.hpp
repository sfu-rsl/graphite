#pragma once
#include <glso/visitor.hpp>
#include <glso/factor.hpp>
#include <glso/vertex.hpp>

namespace glso {

template<typename T=double>
class Graph {

    private:

    GraphVisitor<T> visitor;

    std::vector<BaseVertexDescriptor<T>*> vertex_descriptors;
    std::vector<BaseFactorDescriptor<T>*> factor_descriptors;

    // std::unordered_map<std::pair<size_t, size_t>, std::shared_ptr<JacobianStorage<T>>> jacobians;
    std::unordered_map<size_t, std::pair<size_t, size_t>> hessian_to_local_map;
    std::unordered_map<size_t, size_t> hessian_offset;
    // Solver buffers
    thrust::device_vector<T> delta_x;
    thrust::device_vector<T> b;
    thrust::device_vector<T> x_backup;

    public:

    void add_vertex_descriptor(BaseVertexDescriptor<T>* descriptor) {
        vertex_descriptors.push_back(descriptor);
    }

    template<typename F, typename ... Ts>
    F* add_factor_descriptor(Ts ... vertices) {

        // Link factor to vertices
        auto factor = new F();
        factor->link_factors({vertices...});

        factor_descriptors.push_back(factor);

        // Create Jacobian storage for this factor
        // const auto & info_list = descriptor->get_jacobian_info();

        // for (const auto & info: info_list) {
        //     jacobians.insert({info.dim, std::make_shared<JacobianStorage<T>>(info.nnz())});
        // }

        return factor;

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
        hessian_to_local_map.clear();
        hessian_offset.clear();
        size_t hessian_column = 0;
        size_t offset = 0;
        for (const auto& entry : global_to_local_combined) {
            hessian_to_local_map.insert({hessian_column, entry.second});
            hessian_offset.insert({hessian_column, offset});
            offset += vertex_descriptors[entry.second.first]->dimension();
            vertex_descriptors[entry.second.first]->set_hessian_column(entry.first, hessian_column);
            hessian_column++;
        }

        // Transform global vertex ids into local ids for factors
        // for (const auto & [global_id, second]: global_to_local_combined) {
        //     const auto & [vd_idx, local_id] = second;
        
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

        delta_x.resize(size_x);
        b.resize(size_x);
        x_backup.resize(size_x);

        return true;
    }

    void linearize() {
        // clear delta_x and b
        thrust::fill(delta_x.begin(), delta_x.end(), 0);
        thrust::fill(b.begin(), b.end(), 0);


        for (auto & factor: factor_descriptors) {
            // compute error
            factor->visit_error(visitor);
            if (!factor->use_autodiff()) {
                // manually compute Jacobians
            }
        }



        // TODO: Calculate b=J^T * r
    }

    bool compute_step() {

        return true;
    }

    void apply_step() {
        for (auto & desc: vertex_descriptors) {
            desc->visit_update(visitor);
        }
    }

    void backup_parameters() {
        size_t offset = 0;
        for (const auto & desc: vertex_descriptors) {
            const size_t param_size = desc->dimension()*desc->count();
            const T* x = desc->x();
            thrust::copy(x, x+param_size, x_backup.begin()+offset);
            offset += param_size;
        }

    }

    void revert_parameters() {
        size_t offset = 0;
        for (auto & desc: vertex_descriptors) {
            const size_t param_size = desc->dimension()*desc->count();
            T* x = desc->x();
            thrust::copy(x_backup.begin()+offset, x_backup.begin(), x);
            offset += param_size;
        }

    }

};

}