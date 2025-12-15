#pragma once
#include <graphite/common.hpp>
#include <graphite/stream.hpp>
#include <utility>
#include <unordered_map>
#include <thrust/universal_vector.h>
#include <graphite/utils.hpp>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <graphite/block.hpp>




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

    template <typename T, typename S>
    class Hessian {
        public:

        // Returns coordinates of upper triangular filled-in Hessian blocks
        std::vector<BlockCoordinates> get_block_coordinates(Graph<T, S>* graph) {
           // For a constraint to contribute to the Hessian,
           // the constraint must be active, and both variables must be active (non-fixed),
           // and the block must reside in the upper triangular part of the Hessian (i.e., row index <= column index)
           thrust::device_vector<BlockCoordinates> block_coords;
           auto & f_desc = graph->get_factor_descriptors();
           for (auto & f: f_desc) {
                f->get_hessian_block_coordinates(block_coords);
           }

            // std::cout << "Sorting and removing duplicate block coordinates..." << std::endl;
            thrust::sort(thrust::device, block_coords.begin(), block_coords.end(),
                [] __device__ (const BlockCoordinates & a, const BlockCoordinates & b) {
                    // sort so that we get blocks in column-major order (e.g consecutive indices in same column)
                    if (a.col == b.col) {
                        return a.row < b.row;
                    }
                    return a.col < b.col;
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

        void compute_hessian_blocks(Graph<T, S>* graph, StreamPool &streams) {
            thrust::fill(thrust::device, d_hessian.begin(), d_hessian.end(), static_cast<S>(0.0));

            thrust::host_vector<size_t> h_block_offsets;
            thrust::device_vector<size_t> d_block_offsets;

            auto & f_desc = graph->get_factor_descriptors();
            for (auto & f: f_desc) {
                f->compute_hessian_blocks(block_indices, d_hessian, h_block_offsets, d_block_offsets, streams);
            }


        }

        // Block coords must be unique and in column-major order
        // Build block-csc style indices on the host that we can access on GPU for quickly
        // constructing a scalar CSC-style Hessian matrix (upper triangle only)

        void build_indices(Graph<T, S>* graph, const std::vector<BlockCoordinates>& block_coords) {
            const auto num_columns = graph->get_num_block_columns();
            thrust::host_vector<size_t> col_pointers(num_columns + 1);
            thrust::host_vector<size_t> row_indices(block_coords.size());
            thrust::host_vector<size_t> offsets(block_coords.size());
            thrust::fill(thrust::host, col_pointers.begin(), col_pointers.end(), 0);
            size_t current_col = 0;
            for (size_t i = 0; i < block_coords.size(); i++) {
                const auto & coord = block_coords[i];

                col_pointers[coord.col + 1]++;
                row_indices[i] = coord.row;
                offsets[i] = block_indices[coord];
            }


            thrust::exclusive_scan(
                thrust::device,
                d_col_pointers.begin(),
                d_col_pointers.end(),
                d_col_pointers.begin()
            );


            // Transfer to device
            d_col_pointers = col_pointers;
            d_row_indices = row_indices;
            d_offsets = offsets;
            d_hessian_offsets = graph->get_offset_vector();
 
        }


        void backup_diagonal(Graph<T, S>* graph, StreamPool &streams) {
            d_prev_diagonal.resize(graph->get_hessian_dimension());
            
            auto diag = d_prev_diagonal.data().get(); 
            const auto h_offsets = d_hessian_offsets.data().get();
            const auto p_col = d_col_pointers.data().get();
            const auto r_idx = d_row_indices.data().get();
            const auto block_locations = d_offsets.data().get();
            const auto h = d_hessian.data().get();

            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(graph->get_num_block_columns()),
                [=] __device__ (size_t block_col) {
                    const size_t hessian_col = h_offsets[block_col];
                    const size_t dim = h_offsets[block_col + 1] - hessian_col;

                    // find diagonal block in column where row == col
                    const auto start = p_col[block_col];
                    const auto end = p_col[block_col + 1];

                    for (size_t b = start; b < end; b++) {
                        if (r_idx[b] == block_col) {
                            // found diagonal block, copy elements
                            const auto block = h + block_locations[b];
                            for (size_t i = 0; i < dim; i++) {
                                diag[hessian_col + i] = block[i * dim + i];
                            }
                            break;
                        }
                    }
                }
            );
        }

        void restore_diagonal(Graph<T, S>* graph, StreamPool &streams) {
            d_prev_diagonal.resize(graph->get_hessian_dimension());
            
            const auto diag = d_prev_diagonal.data().get(); 
            const auto h_offsets = d_hessian_offsets.data().get();
            const auto p_col = d_col_pointers.data().get();
            const auto r_idx = d_row_indices.data().get();
            const auto block_locations = d_offsets.data().get();
            auto h = d_hessian.data().get();

            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(graph->get_num_block_columns()),
                [=] __device__ (size_t block_col) {
                    const size_t hessian_col = h_offsets[block_col];
                    const size_t dim = h_offsets[block_col + 1] - hessian_col;

                    // find diagonal block in column where row == col
                    const auto start = p_col[block_col];
                    const auto end = p_col[block_col + 1];

                    for (size_t b = start; b < end; b++) {
                        if (r_idx[b] == block_col) {
                            // found diagonal block, copy elements
                            const auto block = h + block_locations[b];
                            for (size_t i = 0; i < dim; i++) {
                                block[i * dim + i] = diag[hessian_col + i];
                            }
                            break;
                        }
                    }
                }
            );
        }
        

        void backup_diagonal_and_apply_damping(Graph<T, S>* graph, T damping_factor, StreamPool &streams) {
            d_prev_diagonal.resize(graph->get_hessian_dimension());
            
            auto diag = d_prev_diagonal.data().get(); 
            const auto h_offsets = d_hessian_offsets.data().get();
            const auto p_col = d_col_pointers.data().get();
            const auto r_idx = d_row_indices.data().get();
            const auto block_locations = d_offsets.data().get();
            auto h = d_hessian.data().get();

            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(graph->get_num_block_columns()),
                [=] __device__ (size_t block_col) {
                    const size_t hessian_col = h_offsets[block_col];
                    const size_t dim = h_offsets[block_col + 1] - hessian_col;

                    // find diagonal block in column where row == col
                    const auto start = p_col[block_col];
                    const auto end = p_col[block_col + 1];

                    for (size_t b = start; b < end; b++) {
                        if (r_idx[b] == block_col) {
                            // found diagonal block, backup then apply damping
                            const auto block = h + block_locations[b];
                            for (size_t i = 0; i < dim; i++) {
                                diag[hessian_col + i] = block[i * dim + i];
                                block[i*dim + i] = (S)((double)block[i*dim + i] * (1.0 + (double)damping_factor));
                            }
                            break;
                        }
                    }
                }
            );
        }

        // Data
        std::unordered_map<BlockCoordinates, size_t> block_indices;
        thrust::device_vector<S> d_hessian;
        thrust::device_vector<size_t> d_col_pointers;
        thrust::device_vector<size_t> d_row_indices;
        thrust::device_vector<size_t> d_offsets;
        thrust::device_vector<S> d_prev_diagonal;
        thrust::device_vector<size_t> d_hessian_offsets;


        public:

        Hessian() = default;


        void build(Graph<T, S>* graph, T damping_factor, StreamPool &streams) {
            // Implementation for building the Hessian matrix
            
            // Assume we don't have a GPU hash map.
            // First we need to count how many Hessian blocks we have
            // We'll create a set of Hessian block coordinates by iterating over descriptors
            // Ignore blocks not in the upper triangular part
            // std::cout << "Getting Hessian block coordinates..." << std::endl;
            // auto t0 = std::chrono::steady_clock::now();
            const auto block_coords = get_block_coordinates(graph);
            // auto t1 = std::chrono::steady_clock::now();
            // std::cout << "Time to get block coordinates: "
            //           << std::chrono::duration<double>(t1 - t0).count() << " seconds" << std::endl;
            // Then we need to allocate memory for each block
            // We can iterate the set and figure out the total memory,
            // then allocate a big chunk and assign pointers accordingly
            // TODO: Maybe we can use an GPU exclusive scan instead?
            // std::cout << "Allocating Hessian blocks..." << std::endl;
            // auto t2 = std::chrono::steady_clock::now();
            size_t num_values = 0;
            for (const auto & coord : block_coords) {
                block_indices[coord] = num_values;
                num_values += graph->get_variable_dimension(coord.row) * graph->get_variable_dimension(coord.col);
            }
            d_hessian.resize(num_values);
            // auto t3 = std::chrono::steady_clock::now();
            // std::cout << "Time to allocate Hessian blocks: "
            //           << std::chrono::duration<double>(t3 - t2).count() << " seconds" << std::endl;
            // Then for each GPU block, need to calculate the Hessian block values
            // for each descriptor combination in a constraint, we basically need a factor ID, each jacobian pointer for each vertex descriptor,
            // the precision matrix data pointer, and the output location (idx or pointer)
            // std::cout << "Computing Hessian blocks..." << std::endl;
            // auto t4 = std::chrono::steady_clock::now();
            compute_hessian_blocks(graph, streams);
            // auto t5 = std::chrono::steady_clock::now();
            // std::cout << "Time to compute Hessian blocks: "
            //           << std::chrono::duration<double>(t5 - t4).count() << " seconds" << std::endl;

            // We need to end up with a block CSC-style representation
            // where we can iterate down the blocks in each block columnn
            // and retrieve the data pointer for each block for the purpose of
            // constructing a scalar CSC-style representation.
            // std::cout << "Building Hessian indices..." << std::endl;
            // auto t6 = std::chrono::steady_clock::now();
            build_indices(graph, block_coords);
            // auto t7 = std::chrono::steady_clock::now();
            // std::cout << "Time to build Hessian indices: "
            //           << std::chrono::duration<double>(t7 - t6).count() << " seconds" << std::endl;

        }
    };

}