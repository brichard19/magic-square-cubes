#ifndef MAGIC_SQUARE_CUBES_SEARCH_H
#define MAGIC_SQUARE_CUBES_SEARCH_H

#include <cstdint>
#include <vector>

using u32  = std::uint32_t;
using u64  = std::uint64_t;
using u128 = unsigned __int128;

struct search_partition {
    // Effective bounding range for this partition.
    u64 min_a;
    u64 min_b;
    u64 max_a;
    u64 max_b;

    // Half-open sum range:
    //
    //     s_min <= a^3 + b^3 < s_max
    //
    u128 s_min;
    u128 s_max;

    // Exact number of valid (a,b) pairs in this partition.
    u64 pair_count;
};

struct collision_signature {
    u128 u;
    u128 v;
};

struct collision_record {
    u32 a1;
    u32 b1;

    u32 a2;
    u32 b2;

    collision_signature signature;
};

// Every search backend implements this interface.
using partition_search_function =
    std::vector<collision_record>(
        const search_partition& partition);

#endif
