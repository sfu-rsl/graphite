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
        thrust::unified_vector<size_t> hessian_counts;
        // First value is the number of blocks (upper bound guess), second is the pointer to the count value
        std::unordered_map<BlockDimension, std::pair<size_t, size_t*>> hessian_count_map;
        
        public:

        Hessian() = default;


        void build(Graph<T, S>* graph, T damping_factor, streams &streams) {
            // Implementation for building the Hessian matrix
            
            // Assume we don't have a GPU hash map.
            // First we need to count how many Hessian blocks we have
            // We'll create a set of Hessian block coordinates by iterating over descriptors
            // Ignore blocks not in the upper triangular part

            // Then we need to allocate memory for each block
            // We can iterate the set and figure out the total memory,
            // then allocate a big chunk and assign pointers accordingly

            // Then for each GPU block, need to calculate the Hessian block values
            // for each descriptor combination in a constraint, we basically need a factor ID, each jacobian pointer for each vertex descriptor,
            // the precision matrix data pointer, and the output location (idx or pointer)

            // We need to end up with a block CSC-style representation
            // where we can iterate down the blocks in each block columnn
            // and retrieve the data pointer for each block for the purpose of
            // constructing a scalar CSC-style representation.



            // Old - can't use without GPU hash map
            // // 1. First, need to count how many Hessian blocks we have
            // // Only need to count the upper triangular part
            // // Note: constraint must be active and both variables must be
            // // active (non-fixed) for the Hessian block to be counted
            // hessian_counts.clear();
            // hessian_count_map.clear();
            // // size_t total_count = 0;
            // for (auto & f: f->get_factor_descriptors()) {
            //     for (auto & vi: f->vertex_descriptors) {
            //         const size_t dim_i = vi->dimension();
            //         for (auto & vj: f->vertex_descriptors) {
            //             const size_t dim_j = vj->dimension();
            //             const auto active_count = f->active_count();
            //             // total_count += active_count;
            //             hessian_count_map[{dim_i, dim_j}].first += active_count;
            //         }
            //     }
            // }
            

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


    };

}