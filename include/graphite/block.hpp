#pragma once
#include <graphite/utils.hpp>

namespace graphite {
    class BlockCoordinates {
        public:
        size_t row;
        size_t col;

        bool operator==(const BlockCoordinates& other) const {
            return (row == other.row) && (col == other.col);
        }
    };

    using BlockDimension = BlockCoordinates;
}

namespace std {
    template <>
    struct hash<graphite::BlockCoordinates> {
        size_t operator()(const graphite::BlockCoordinates& bc) const {
            // Combine row and col into a single hash value
            // Requires C++ 20
            // return std::hash<std::pair<size_t, size_t>>{}(
            //     std::make_pair(bc.row, bc.col));
            size_t seed = 0;
            graphite::hash_combine(seed, bc.row);
            graphite::hash_combine(seed, bc.col);
            return seed;
        }
    };
}