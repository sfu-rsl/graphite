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


    template <typename T, typename I>
    class CSRMatrix {
        public:

        thrust::device_vector<I> d_row_pointers;
        thrust::device_vector<I> d_col_indices;
        thrust::device_vector<T> d_values;

    };

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

                col_pointers[coord.col]++;
                row_indices[i] = coord.row;
                offsets[i] = block_indices[coord];
            }




            // Transfer to device
            d_col_pointers = col_pointers;
            d_row_indices = row_indices;
            d_offsets = offsets;
            d_hessian_offsets = graph->get_offset_vector();

            thrust::exclusive_scan(
                thrust::device,
                d_col_pointers.begin(),
                d_col_pointers.end(),
                d_col_pointers.begin()
            );

            // Also need a map of scalar column to block column for constructing CSR matrices
            scalar_to_block_map.resize(graph->get_hessian_dimension());
            auto s2b_map = scalar_to_block_map.data().get();
            const auto bc_offset = d_hessian_offsets.data().get();
            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(graph->get_num_block_columns()),
                [=] __device__ (size_t block_column) {
                    const size_t hessian_col = bc_offset[block_column];
                    const auto dim = bc_offset[block_column + 1] - hessian_col;

                    for (size_t i = 0; i < dim; i++) {
                        s2b_map[hessian_col + i] = block_column;
                    }
                    
                }
            );

            // thrust::host_vector<size_t> h_s2b = scalar_to_block_map;
            // std::cout << "Scalar to block column map: ";
            // for (size_t i = 0; i < h_s2b.size(); i++) {
            //     std::cout << h_s2b[i] << " ";
            // }
            // std::cout << std::endl;

            // // Print block column pointers
            // thrust::host_vector<size_t> h_col_ptrs = d_col_pointers;
            // std::cout << "Hessian block column pointers: ";
            // for (size_t i = 0; i < h_col_ptrs.size(); i++) {
            //     std::cout << h_col_ptrs[i] << " ";
            // }
            // std::cout << std::endl;

            // // Print block row indices
            // thrust::host_vector<size_t> h_row_idxs = d_row_indices;
            // std::cout << "Hessian block row indices: ";
            // for (size_t i = 0; i < h_row_idxs.size(); i++) {
            //     std::cout << h_row_idxs[i] << " ";
            // }
            // std::cout << std::endl;

 
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
        

        void apply_damping(Graph<T, S>* graph, T damping_factor, StreamPool &streams) {
            // Assume diagonal was already backed up when recomputing Hessian values
            auto diag = d_prev_diagonal.data().get(); 
            const auto h_offsets = d_hessian_offsets.data().get();
            const auto p_col = d_col_pointers.data().get();
            const auto r_idx = d_row_indices.data().get();
            auto block_locations = d_offsets.data().get();
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
                            auto block = h + block_locations[b];
                            for (size_t i = 0; i < dim; i++) {
                                const auto h_diag = diag[hessian_col + i];
                                block[i*dim + i] = (S)((double)h_diag * (1.0 + (double)damping_factor));
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
        thrust::device_vector<size_t> scalar_to_block_map;

        // For calculating Hessian blocks
        thrust::host_vector<size_t> h_block_offsets;
        thrust::device_vector<size_t> d_block_offsets;


        public:

        Hessian() = default;


        void build_structure(Graph<T, S>* graph, StreamPool &streams) {
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
            // compute_hessian_blocks(graph, streams);
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

        void update_values(Graph<T, S>* graph, StreamPool& streams) {
            thrust::fill(thrust::device, d_hessian.begin(), d_hessian.end(), static_cast<S>(0.0));




            auto & f_desc = graph->get_factor_descriptors();
            for (auto & f: f_desc) {
                h_block_offsets.clear();
                d_block_offsets.clear();
                f->compute_hessian_blocks(block_indices, d_hessian, h_block_offsets, d_block_offsets, streams);
            }

            // Back up diagonal for damping purposes
            backup_diagonal(graph, streams);
        }

        template <typename I>
        void build_csr_structure(Graph<T, S>* graph, CSRMatrix<S, I> & matrix) {
            
            // const auto nnz = d_hessian.size(); // this is an upper bound, since it stores all values of blocks on the diagonal
            const auto hessian_dim = graph->get_hessian_dimension();
            matrix.d_row_pointers.resize(hessian_dim + 1);
            thrust::fill(thrust::device, matrix.d_row_pointers.begin(), matrix.d_row_pointers.end(), 0);

            // Now we count how many values in each column.
            // Note that we're using a block CSC variant (each block is column major)
            // and we can use this to construct a triangular CSR matrix because
            // the Hessian is symmetric.

            // Iterate through each block column

            const auto h = d_hessian.data().get();
            const auto p_col = d_col_pointers.data().get();
            const auto r_idx = d_row_indices.data().get();
            const auto block_locations = d_offsets.data().get();
            const auto hessian_offsets = d_hessian_offsets.data().get(); // this is scalar offset
            const auto s2b_map = scalar_to_block_map.data().get();
            
            auto scalar_row_ptrs = matrix.d_row_pointers.data().get();

            // std::cout << "Hessian dimension: " << hessian_dim << std::endl;
            // std::cout << "Stored hessian non-zeroes: " << d_hessian.size() << std::endl;

            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(hessian_dim),
                [=] __device__ (const size_t hessian_col) {
                    const auto block_col = s2b_map[hessian_col];
                    const size_t ncols = hessian_offsets[block_col + 1] - hessian_col;

                    // find diagonal block in column where row == col
                    const auto start = p_col[block_col];
                    const auto end = p_col[block_col + 1];
                    size_t num_values = 0; // values in upper triangle scalar column
                    // Iterate through each block in the column
                    for (size_t b = start; b < end; b++) {

                        // stop if col == row
                        const auto block_row = r_idx[b];
                        const auto nrows = hessian_offsets[block_row + 1] - hessian_offsets[block_row];

                        auto scalar_row = hessian_offsets[block_row];

                        for (size_t r = 0; r < nrows; r++) {
                            if (scalar_row + r <= hessian_col) {
                                // only count upper triangle
                                num_values++;
                            }
                            else {
                                break;
                            }
                        }

                        if (block_row >= block_col) {
                            // reached diagonal block, stop
                            break;
                        }

                    }
                    scalar_row_ptrs[hessian_col] = num_values;
                }
            );

            // Now sum up everything
            thrust::exclusive_scan(
                thrust::device,
                matrix.d_row_pointers.begin(),
                matrix.d_row_pointers.end(),
                matrix.d_row_pointers.begin()
            );

            // // debug
            // thrust::host_vector<I> h_row_ptrs = matrix.d_row_pointers;
            // std::cout << "Hessian CSR row pointers: ";
            // for (size_t i = 0; i < h_row_ptrs.size(); i++) {
            //     std::cout << h_row_ptrs[i] << " ";
            // }
            // std::cout << std::endl;

            // Now we need to iterate through columns again and populate 
            // the column indices and values
            const size_t nnz = matrix.d_row_pointers[hessian_dim]; // expensive! we're accessing device memory
            matrix.d_col_indices.resize(nnz);
            matrix.d_values.resize(nnz);


            auto p_col_indices = matrix.d_col_indices.data().get();

            // Update indices
            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(hessian_dim),
                [=] __device__ (const size_t hessian_col) {
                    const auto block_col = s2b_map[hessian_col];
                    const size_t ncols = hessian_offsets[block_col + 1] - hessian_col;

                    // find diagonal block in column where row == col
                    const auto start = p_col[block_col];
                    const auto end = p_col[block_col + 1];
                    const size_t num_values = 0; // values in upper triangle scalar column
                    size_t write_idx = scalar_row_ptrs[hessian_col];
                    // Iterate through each block in the column
                    for (size_t b = start; b < end; b++) {

                        // stop if col == row
                        const auto block_row = r_idx[b];
                        const auto nrows = hessian_offsets[block_row + 1] - hessian_offsets[block_row];

                        auto scalar_row = hessian_offsets[block_row];

                        const auto col_in_block = hessian_col - hessian_offsets[block_col];


                        for (size_t r = 0; r < nrows; r++) {
                            if (scalar_row + r <= hessian_col) {
                                // only write out upper triangle
                                p_col_indices[write_idx] = scalar_row + r;
                                write_idx++;
                            }
                            else {
                                break;
                            }
                        }

                        if (block_row >= block_col) {
                            // reached diagonal block, stop
                            break;
                        }

                    }                
                }
            );

            // std::cout << "Hessian CSR column indices: ";
            // thrust::host_vector<I> h_col_idxs = matrix.d_col_indices;
            // for (size_t i = 0; i < h_col_idxs.size(); i++) {
            //     std::cout << h_col_idxs[i] << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "Raw Hessian data: ";
            // thrust::host_vector<S> h_hessian = d_hessian;
            // for (size_t i = 0; i < h_hessian.size(); i++)
            // {
            //     std::cout << h_hessian[i] << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "Raw Hessian offsets: ";
            // thrust::host_vector<size_t> h_offsets = d_offsets;
            // for (size_t i = 0; i < h_offsets.size(); i++)
            // {
            //     std::cout << h_offsets[i] << " ";
            // }
            // std::cout << std::endl;

        }

        template <typename I>
        void update_csr_values(Graph<T, S>* graph, CSRMatrix<S, I> & matrix) {

            const auto hessian_dim = graph->get_hessian_dimension();
            const auto h = d_hessian.data().get();
            const auto p_col = d_col_pointers.data().get();
            const auto r_idx = d_row_indices.data().get();
            const auto block_locations = d_offsets.data().get();
            const auto hessian_offsets = d_hessian_offsets.data().get(); // this is scalar offset
            const auto s2b_map = scalar_to_block_map.data().get();
            
            auto scalar_row_ptrs = matrix.d_row_pointers.data().get();

            auto p_values = matrix.d_values.data().get();

            thrust::for_each(
                thrust::device,
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(hessian_dim),
                [=] __device__ (const size_t hessian_col) {
                    const auto block_col = s2b_map[hessian_col];
                    const size_t ncols = hessian_offsets[block_col + 1] - hessian_col;

                    // find diagonal block in column where row == col
                    const auto start = p_col[block_col];
                    const auto end = p_col[block_col + 1];
                    const size_t num_values = 0; // values in upper triangle scalar column
                    size_t write_idx = scalar_row_ptrs[hessian_col];
                    // Iterate through each block in the column
                    for (size_t b = start; b < end; b++) {

                        // stop if col == row
                        const auto block_row = r_idx[b];
                        const auto nrows = hessian_offsets[block_row + 1] - hessian_offsets[block_row];

                        auto scalar_row = hessian_offsets[block_row];

                        const auto col_in_block = hessian_col - hessian_offsets[block_col];

                        const auto block = h + block_locations[b] + col_in_block * nrows;

                        for (size_t r = 0; r < nrows; r++) {
                            if (scalar_row + r <= hessian_col) {
                                // only write out upper triangle
                                p_values[write_idx] = block[r];
                                write_idx++;
                            }
                            else {
                                break;
                            }
                        }

                        if (block_row >= block_col) {
                            // reached diagonal block, stop
                            break;
                        }

                    }                
                }
            );

            // std::cout << "Hessian CSR values: ";
            // thrust::host_vector<S> h_values = matrix.d_values;
            // for (size_t i = 0; i < h_values.size(); i++) {
            //     std::cout << h_values[i] << " ";
            // }
            // std::cout << std::endl;

        }
    };



}