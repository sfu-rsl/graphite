#pragma once
#include <glso/common.hpp>
#include <glso/vertex.hpp>
#include <thrust/inner_product.h>
#include <thrust/universal_vector.h>
#include <glso/utils.hpp>
#include <glso/loss.hpp>

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
    virtual void visit_error_autodiff(GraphVisitor<T>& visitor) = 0;
    virtual void visit_b(GraphVisitor<T>& visitor) = 0;
    virtual void visit_Jv(GraphVisitor<T>& visitor, T* out, T* in) = 0;
    virtual void visit_Jtv(GraphVisitor<T>& visitor, T* out, T* in) = 0;
 
    virtual JacobianStorage<T>* get_jacobians() = 0;
    virtual void initialize_jacobian_storage() = 0;
    // virtual size_t get_num_vertices() const = 0;

    virtual size_t count() const = 0;
    virtual size_t get_residual_size() const = 0;

    virtual void to_device() = 0;

    virtual T chi2(GraphVisitor<T>& visitor) = 0;

};

template <typename T, int E, typename M, typename C, template <typename, int> class L, template <typename> class Derived, typename... VDTypes>
class FactorDescriptor : public BaseFactorDescriptor<T> {

private:
    std::vector<size_t> global_ids;
    // thrust::host_vector<size_t> host_ids; // local ids
    // thrust::host_vector<T> host_obs;

public:

    static constexpr size_t N = sizeof...(VDTypes);
    static constexpr size_t error_dim = E;

    std::array<BaseVertexDescriptor<T>*, N> vertex_descriptors;
    using VertexTypesTuple = std::tuple<typename VDTypes::VertexType...>;
    using VertexPointerTuple = std::tuple<typename VDTypes::VertexType*...>;
    using ObservationType = M;
    using ConstraintDataType = C;
    using LossType = L<T, E>;

    thrust::universal_vector<size_t> device_ids;
    thrust::universal_vector<M> device_obs;
    thrust::device_vector<T> residuals;
    thrust::universal_vector<T> precision_matrices;
    thrust::universal_vector<C> data; 

    thrust::universal_vector<T> chi2_vec;
    thrust::universal_vector<LossType> loss;

    void visit_error(GraphVisitor<T>& visitor) override {
        visitor.template compute_error<Derived<T>, VDTypes...>(dynamic_cast<Derived<T>*>(this));
    }

    void visit_error_autodiff(GraphVisitor<T>& visitor) override {
        visitor.template compute_error_autodiff<Derived<T>, VDTypes...>(dynamic_cast<Derived<T>*>(this));
    }

    void visit_b(GraphVisitor<T>& visitor) override {
        visitor.template compute_b<Derived<T>, VDTypes...>(dynamic_cast<Derived<T>*>(this));
    }

    void visit_Jv(GraphVisitor<T>& visitor, T* out, T* in) override {
        visitor.template compute_Jv<Derived<T>, VDTypes...>(dynamic_cast<Derived<T>*>(this), out, in);
    }

    void visit_Jtv(GraphVisitor<T>& visitor, T* out, T* in) override {
        visitor.template compute_Jtv<Derived<T>, VDTypes...>(dynamic_cast<Derived<T>*>(this), out, in);
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

    void add_factor(const std::array<size_t, N>& ids, const M& obs, const T* precision_matrix, const C& constraint_data, const LossType& loss_func) {
        
        global_ids.insert(global_ids.end(), ids.begin(), ids.end());
        device_obs.push_back(obs);

        constexpr size_t precision_matrix_size = error_dim*error_dim;
        if (precision_matrix) {
            precision_matrices.insert(precision_matrices.end(), precision_matrix, precision_matrix + precision_matrix_size);
        }
        else {
            constexpr auto pmat = get_default_precision_matrix();
            precision_matrices.insert(precision_matrices.end(), pmat.data(), pmat.data() + precision_matrix_size);

        }

        data.push_back(constraint_data);
        loss.push_back(loss_func);
    }

    // TODO: Make this private later
    constexpr static std::array<T, E*E> get_default_precision_matrix() {
        return []() constexpr {
            std::array<T, E*E> pmat = {};
            for (size_t i = 0; i < E; i++) {
            pmat[i*E + i] = 1.0;
            }
            return pmat;
        }();
    }

    size_t count() const override {
        return device_ids.size()/N;
    }

    void to_device() override {

        // Map constraint ids to local ids
        
        // auto start = std::chrono::high_resolution_clock::now();

        device_ids.resize(global_ids.size());
        prefetch_vector_on_host(device_ids, 0);
        for (size_t i = 0; i < global_ids.size(); i++) {
            device_ids[i] = vertex_descriptors[i%N]->get_global_map().at(global_ids[i]);
        }

        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Device id building time: " << elapsed.count() << " seconds" << std::endl;

        // device_ids = host_ids;
        // device_obs = host_obs;
        chi2_vec.resize(count());

        // prefetch everything
        int cuda_device = 0;
        constexpr cudaStream_t stream = 0;
        cudaGetDevice(&cuda_device);
        prefetch_vector_on_device_async(device_ids, cuda_device, stream);
        prefetch_vector_on_device_async(device_obs, cuda_device, stream);
        prefetch_vector_on_device_async(chi2_vec, cuda_device, stream);
        prefetch_vector_on_device_async(data, cuda_device, stream);
        prefetch_vector_on_device_async(loss, cuda_device, stream);
        // std::cout << "Prefetching factor data to device" << std::endl;
        cudaDeviceSynchronize();
        // Resize and reset residuals
        // std::cout << "Resizing residuals to: " << error_dim*count() << std::endl;
        residuals.resize(error_dim*count());
        // std::cout << "Filling residuals with zeros" << std::endl;
        thrust::fill(residuals.begin(), residuals.end(), 0);
        // std::cout << "Resizing residuals to: " << error_dim*count() << std::endl;
    }

    void link_factors(const std::array<BaseVertexDescriptor<T>*, N>& vertex_descriptors) {
        this->vertex_descriptors = vertex_descriptors;
    }

    template <std::size_t... I>
    VertexPointerTuple get_vertices_impl(std::index_sequence<I...>) {
        return std::make_tuple((static_cast<typename std::tuple_element<I, std::tuple<VDTypes...>>::type*>(vertex_descriptors[I])->vertices())...);
    }
    
    // Return tuple of N vertex pointers from this->vertex_descriptors[i]->vertices()
    VertexPointerTuple get_vertices() {
        return get_vertices_impl(std::make_index_sequence<N>{});
    }

    static constexpr std::array<size_t, N> get_vertex_sizes() {
        return {VDTypes::dim...};
    }

    void initialize_jacobian_storage() override {
        for (size_t i = 0; i < N; i++) {
            jacobians[i].dimensions = {error_dim, vertex_descriptors[i]->dimension()};
            jacobians[i].data.resize(error_dim*vertex_descriptors[i]->dimension()*count());
        }
    }

    virtual size_t get_residual_size() const override {
        return error_dim*count();
    }

    // TODO: Make this consider kernels and active edges
    virtual T chi2(GraphVisitor<T>& visitor) override {
        // T chi2 = thrust::inner_product(residuals.begin(), residuals.end(), residuals.begin(), 0.0);
        // return chi2;
        visitor.template compute_chi2<Derived<T>>(dynamic_cast<Derived<T>*>(this));
        return thrust::reduce(chi2_vec.begin(), chi2_vec.end(), 0.0, thrust::plus<T>());
    }

};

// Templated derived class for AutoDiffFactorDescriptor using CRTP
// N is the number of vertices involved in the constraint
// M is the dimension of each observation
template <typename T, int E, typename M, typename C, template <typename, int> class L, template <typename> class Derived, typename... VDTypes>
class AutoDiffFactorDescriptor : public FactorDescriptor<T, E, M, C, L, Derived, VDTypes...> {
public:
    virtual bool use_autodiff() override {
        return true;
    }
};
}