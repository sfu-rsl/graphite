#pragma once
#include <graphite/common.hpp>
#include <graphite/stream.hpp>
#include <utility>
#include <unordered_map>
#include <thrust/universal_vector.h>


namespace graphite {
    class BlockCoordinates {
        public:
        size_t row;
        size_t col;
    };

    using BlockDimension = BlockCoordinates;
}

namespace std {
    template <>
    struct hash<graphite::BlockCoordinates> {
        size_t operator()(const graphite::BlockCoordinates& bc) const {
            // Combine row and col into a single hash value
            // Requires C++ 20
            return std::hash<std::pair<size_t, size_t>>{}(
                std::make_pair(bc.row, bc.col));
        }
    };
}

namespace graphite {

    template <typename T> class HessianBlocks {
        public:
        std::pair<size_t, size_t> dimensions;
        size_t num_blocks;
        thrust::device_vector<T> data;
        thrust::device_vector<BlockCoordinates> block_coordinates;

        void resize(size_t num_blocks, size_t rows, size_t cols) {
            dimensions = {rows, cols};
            this->num_blocks = num_blocks;
            data.resize(rows * cols * num_blocks);
            block_coordinates.resize(num_blocks);
        }

        void fill(const T& value) {
            thrust::fill(thrust::device, data.begin(), data.end(), value);
        }

        void zero() {
            thrust::fill(thrust::device, data.begin(), data.end(), static_cast<T>(0));
        }
        
    };

    class MulItem {
        public:
        size_t factor_id; // used to look up each Jacobian per vertex descriptor
        size_t output_id; // used to find the location of the output Hessian block
    };

    // Create one list per combination of vertex descriptors per factor
    class MulList {
        private:
        thrust::unified_vector<MulItem> items;
        public:
        void add_item(const MulItem& item) {
            items.push_back(item);
        }

        void clear() {
            items.clear();
        }

        size_t size() const {
            return items.size();
        }
    };

    template <typename T, typename S>
    class Hessian {
        private:

        // Returns coordinates of upper triangular filled-in Hessian blocks
        std::vector<BlockCoordinates> get_block_coordinates(Graph<T, S>* graph) {
           // For a constraint to contribute to the Hessian,
           // the constraint must be active, and both variables must be active (non-fixed),
           // and the block must reside in the upper triangular part of the Hessian (i.e., row index <= column index)
           thrust::device_vector<BlockCoordinates> block_coords;
           auto & f_desc = graph->get_factor_descriptors();
           for (auto & f: f_desc) {
                const size_t num_vertices = f->get_num_vertices();
                const auto & active_factors = f->active_indices;
                const auto device_ids = f->device_ids.data().get();
                for (size_t i = 0; i < num_vertices; i++) {
                    const auto vi_active = f->vertex_descriptors[i]->active_state.data().get();
                    const vi_block_ids = f->vertex_descriptors[i]->block_ids.data().get();
                    for (size_t j = i; j < num_vertices; j++) {
                        const auto vj_active = f->vertex_descriptors[j]->active_state.data.get();
                        const vj_block_ids = f->vertex_descriptors[j]->block_ids.data().get();
                        // Iterate over active factors and generate block coordinates
                        thrust::transform_if(
                            thrust::device,
                            active_factors.begin(),
                            active_factors.end(),
                            block_coords.begin(),
                            [=] __device__ (size_t factor_idx) {
                                const size_t vi_id = device_ids[factor_idx * num_vertices + i];
                                const size_t vj_id = device_ids[factor_idx * num_vertices + j];

                                const auto block_i = vi_block_ids[vi_id];
                                const auto block_j = vj_block_ids[vj_id];
                                return BlockCoordinates{block_i, block_j};
                            },
                            [] __device__ (const size_t & factor_idx) {
                                const auto vi_id = device_ids[factor_idx * num_vertices + i];
                                const auto vj_id = device_ids[factor_idx * num_vertices + j];
                                return (is_vertex_active(vi_active, vi_id) && is_vertex_active(vj_active, vj_id));
                            }
                        );
                    }
                }
            }

            thrust::sort(thrust::device, block_coords.begin(), block_coords.end(),
                [] __device__ (const BlockCoordinates & a, const BlockCoordinates & b) {
                    // sort 2D coordinate
                    if (a.row == b.row) {
                        return a.col < b.col;
                    }
                    return a.row < b.row;
                }
            );

            // Remove duplicates
            auto end_it = thrust::unique(thrust::device, block_coords.begin(), block_coords.end(),
                [] __device__ (const BlockCoordinates & a, const BlockCoordinates & b) {
                    return (a.row == b.row) && (a.col == b.col);
                }
            );

            // Resize the vector to remove duplicates
            block_coords.resize(end_it - block_coords.begin());


            // Copy to host and convert to std::vector
            std::vector<BlockCoordinates> host_block_coords(block_coords.size());
            thrust::copy(thrust::device, block_coords.begin(), block_coords.end(), host_block_coords.begin());
            return host_block_coords;
        }

        void compute_hessian_blocks(Graph<T, S>* graph) {
            thrust::fill(thrust::device, d_hessian.begin(), d_hessian.end(), static_cast<S>(0.0));

        }

        void build_indices(Graph<T, S>* graph) {

        }


        // Data
        std::unordered_map<BlockCoordinates, size_t> block_indices;
        thrust::device_vector<S> d_hessian;


        public:

        Hessian() = default;


        void build(Graph<T, S>* graph, T damping_factor, streams &streams) {
            // Implementation for building the Hessian matrix
            
            // Assume we don't have a GPU hash map.
            // First we need to count how many Hessian blocks we have
            // We'll create a set of Hessian block coordinates by iterating over descriptors
            // Ignore blocks not in the upper triangular part
            const auto block_coords = get_block_coordinates(graph);

            // Then we need to allocate memory for each block
            // We can iterate the set and figure out the total memory,
            // then allocate a big chunk and assign pointers accordingly
            // TODO: Maybe we can use an GPU exclusive scan instead?
            size_t num_values = 0;
            for (const auto & coord : block_coords) {
                block_indices[coord] = num_values;
                num_values += graph->get_variable_dimension(coord.row) * graph->get_variable_dimension(coord.col);
            }
            d_hessian.resize(num_values);

            // Then for each GPU block, need to calculate the Hessian block values
            // for each descriptor combination in a constraint, we basically need a factor ID, each jacobian pointer for each vertex descriptor,
            // the precision matrix data pointer, and the output location (idx or pointer)
            compute_hessian_blocks(graph);

            // We need to end up with a block CSC-style representation
            // where we can iterate down the blocks in each block columnn
            // and retrieve the data pointer for each block for the purpose of
            // constructing a scalar CSC-style representation.
            build_indices(graph);

            // Old - can't use without GPU hash map
            // // 1. First, need to count how many Hessian blocks we have
            // // Only need to count the upper triangular part
            // // Note: constraint must be active and both variables must be
            // // active (non-fixed) for the Hessian block to be counted
            

            // // Allocate memory for counting
            // hessian_counts.resize(hessian_count_map.size());
            // thrust::fill(thrust::device, hessian_counts.begin(), hessian_counts.end(), 0);

            // // Now actually count the blocks


            // // 2. Allocate space for Hessian blocks based on count

            // // Create GPU hash table per Hessian block size
            // // where the key is the Hessian block index (i,j)
            // // and the value is the pointer to the Hessian block


            // // 3. Compute the Hessian blocks:
            // // For each constraint
            // // For each variable in a constraint
            // // For the other variables including itself
            // // Compute J_i^T * Sigma^{-1} * J_j (where j can equal i)



        }

        // Need functions for applying damping factor and recalculating Hessian with same structure


    };

}