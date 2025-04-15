#pragma once
#include <glso/common.hpp>
#include <glso/vertex.hpp>
#include <thrust/inner_product.h>

namespace glso {
template<typename T>
class JacobianStorage {
public:

std::pair<size_t, size_t> dimensions;

thrust::device_vector<T> data;


};

template <typename T>
class BaseFactorDescriptor {
public:
    virtual ~BaseFactorDescriptor() {};

    // virtual void error_func(const T** vertices, const T* obs, T* error) = 0;
    virtual bool use_autodiff() = 0;
    virtual void visit_error(GraphVisitor<T>& visitor) = 0;
    virtual void visit_b(GraphVisitor<T>& visitor) = 0;
    virtual void visit_Jv(GraphVisitor<T>& visitor, T* out, T* in) = 0;
    virtual void visit_Jtv(GraphVisitor<T>& visitor, T* out, T* in) = 0;
 
    virtual JacobianStorage<T>* get_jacobians() = 0;
    virtual void initialize_jacobian_storage() = 0;
    // virtual size_t get_num_vertices() const = 0;

    virtual size_t count() const = 0;
    virtual size_t get_residual_size() const = 0;

    virtual void to_device() = 0;

    virtual size_t set_error_offset(size_t offset) = 0;

    virtual T chi2() = 0;

};

template <typename T, int E, int M, template <typename> class Derived, typename... VertexTypes>
class FactorDescriptor : public BaseFactorDescriptor<T> {

private:
    std::vector<size_t> global_ids;
    thrust::host_vector<size_t> host_ids; // local ids
    thrust::host_vector<T> host_obs;
    thrust::host_vector<T> host_hessian_ids;
    thrust::host_vector<T> host_error_offsets; // we may not need this

public:

    static constexpr size_t N = sizeof...(VertexTypes);
    static constexpr size_t observation_dim = M;
    static constexpr size_t error_dim = E;

    std::array<BaseVertexDescriptor<T>*, N> vertex_descriptors;
    using VertexTypesTuple = std::tuple<VertexTypes...>;

    thrust::device_vector<size_t> device_ids;
    thrust::device_vector<T> device_obs;
    thrust::device_vector<T> residuals;
    thrust::device_vector<size_t> device_hessian_ids;
    thrust::device_vector<size_t> device_error_offsets; // we may not need this

    void visit_error(GraphVisitor<T>& visitor) override {
        visitor.template compute_error<Derived<T>, VertexTypes...>(dynamic_cast<Derived<T>*>(this));
    }

    void visit_b(GraphVisitor<T>& visitor) override {
        visitor.template compute_b<Derived<T>, VertexTypes...>(dynamic_cast<Derived<T>*>(this));
    }

    void visit_Jv(GraphVisitor<T>& visitor, T* out, T* in) override {
        visitor.template compute_Jv<Derived<T>, VertexTypes...>(dynamic_cast<Derived<T>*>(this), out, in);
    }

    void visit_Jtv(GraphVisitor<T>& visitor, T* out, T* in) override {
        visitor.template compute_Jtv<Derived<T>, VertexTypes...>(dynamic_cast<Derived<T>*>(this), out, in);
    }

    // std::vector<JacobianStorage<T>> jacobians;
    std::array<JacobianStorage<T>, N> jacobians;
    
    static constexpr size_t get_num_vertices() {
        return N;
    }

    // std::vector<JacobianStorage<T>> &  get_jacobians() override {
    //     return jacobians;
    // }

    JacobianStorage<T>* get_jacobians() override {
        return jacobians.data();
    }

    void add_factor(const std::array<size_t, N>& ids, const std::array<T, M>& obs, const T* precision_matrix) {
        
        global_ids.insert(global_ids.end(), ids.begin(), ids.end());
        host_obs.insert(host_obs.end(), obs.begin(), obs.end());

    }

    size_t count() const override {
        return device_ids.size()/N;
    }

    void to_device() override {

        // Map constraint ids to local ids
        for (size_t i = 0; i < global_ids.size(); i++) {
            host_ids.push_back(vertex_descriptors[i%N]->get_global_map().at(global_ids[i]));
        }

        device_ids = host_ids;
        device_obs = host_obs;
        device_error_offsets = host_error_offsets;

        // Resize and reset residuals
        residuals.resize(error_dim*count());
        thrust::fill(residuals.begin(), residuals.end(), 0);
    }

    void link_factors(const std::array<BaseVertexDescriptor<T>*, N>& vertex_descriptors) {
        this->vertex_descriptors = vertex_descriptors;
    }

    static constexpr std::array<size_t, N> get_vertex_sizes() {
        return {VertexTypes::dim...};
    }

    void initialize_jacobian_storage() override {
        for (size_t i = 0; i < N; i++) {
            jacobians[i].dimensions = {error_dim, vertex_descriptors[i]->dimension()};
            jacobians[i].data.resize(error_dim*vertex_descriptors[i]->dimension()*count());
        }
    }

    size_t set_error_offset(size_t offset) override {
        // Just assume all edges are active for now
        host_error_offsets.clear();
        host_error_offsets.resize(count());


        for (size_t i = 0; i < count(); i++) {
            host_error_offsets[i] = offset;
            offset += error_dim;
        }

        return offset;
    }

    virtual size_t get_residual_size() const override {
        return error_dim*count();
    }

    // TODO: Make this consider kernels and active edges
    virtual T chi2() override {
        T chi2 = thrust::inner_product(residuals.begin(), residuals.end(), residuals.begin(), 0.0);
        return chi2;
    }

};

// Templated derived class for AutoDiffFactorDescriptor using CRTP
// N is the number of vertices involved in the constraint
// M is the dimension of each observation
template <typename T, int E, int M, template <typename> class Derived, typename... VertexTypes>
class AutoDiffFactorDescriptor : public FactorDescriptor<T, E, M, Derived, VertexTypes...> {
public:
    virtual bool use_autodiff() override {
        return true;
    }
};
}