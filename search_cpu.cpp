#include "search_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

// Internal temporary record.
struct pair_record {
    // Storing the 128-bit sum directly would align the whole
    // record to 16 bytes and pad it from 24 to 32 bytes. Split
    // it into halves so each temporary pair occupies 24 bytes.
    u64 s_low;
    u64 s_high;
    u32 a;
    u32 b;

    pair_record(u128 sum, u32 a_value, u32 b_value)
        : s_low(static_cast<u64>(sum)),
          s_high(static_cast<u64>(sum >> 64)),
          a(std::min(a_value, b_value)),
          b(std::max(a_value, b_value))
    {}

    u128 sum() const
    {
        return
            (u128(s_high) << 64) |
            u128(s_low);
    }
};

struct b_bounds {
    u32 a;
    u32 min_b;
    u32 max_b;
};

static inline u128 cube(u64 x)
{
    return u128(x) * x * x;
}

// Floating-point estimate followed by exact correction.
static u64 floor_cuberoot(u128 n)
{
    const long double x =
        std::cbrt(static_cast<long double>(n));

    u64 r = static_cast<u64>(x);

    while (cube(r) > n)
        --r;

    while (cube(r + 1) <= n)
        ++r;

    return r;
}

static u64 ceil_cuberoot(u128 n)
{
    const u64 r = floor_cuberoot(n);

    if (cube(r) == n)
        return r;

    return r + 1;
}

static u128 gcd_u128(u128 x, u128 y)
{
    while (y != 0) {
        const u128 remainder = x % y;
        x = y;
        y = remainder;
    }

    return x;
}

static collision_signature make_collision_signature(
    const pair_record& first,
    const pair_record& second)
{
    const u128 x =
        cube(static_cast<u64>(first.b)) -
        cube(static_cast<u64>(first.a));

    const u128 y =
        cube(static_cast<u64>(second.b)) -
        cube(static_cast<u64>(second.a));

    if (x == 0 || y == 0) {
        throw std::logic_error(
            "collision signature requires distinct pair values");
    }

    const u128 divisor = gcd_u128(x, y);
    const u128 reduced_x = x / divisor;
    const u128 reduced_y = y / divisor;

    return {
        std::min(reduced_x, reduced_y),
        std::max(reduced_x, reduced_y)
    };
}

} // namespace

std::size_t cpu_pair_record_size() noexcept
{
    return sizeof(pair_record);
}

u64 cpu_pair_record_max_count() noexcept
{
    return static_cast<u64>(
        std::vector<pair_record>{}.max_size());
}

std::vector<collision_record> search_partition_collisions_cpu(
    const search_partition& partition)
{
    if (partition.pair_count == 0 ||
        partition.min_a > partition.max_a ||
        partition.min_b > partition.max_b ||
        partition.s_min >= partition.s_max)
    {
        return {};
    }

    const u64 maximum_stored_value =
        std::numeric_limits<u32>::max();

    if (partition.min_a > maximum_stored_value ||
        partition.max_a > maximum_stored_value ||
        partition.min_b > maximum_stored_value ||
        partition.max_b > maximum_stored_value)
    {
        throw std::out_of_range(
            "a and b values must fit in uint32_t");
    }

    // Pre-compute the valid b range for every a in this
    // S partition.
    std::vector<b_bounds> bounds;

    for (u64 a = partition.min_a;
         a <= partition.max_a;
         ++a)
    {
        const u128 a3 = cube(a);

        // If a^3 alone has reached s_max, no positive b
        // can bring the sum back into this partition.
        if (a3 >= partition.s_max)
            break;

        // Calculate the exact lower b bound:
        //
        //     s_min <= a^3 + b^3
        u64 first_b;

        if (a3 >= partition.s_min) {
            first_b =
                std::max(
                    partition.min_b,
                    a + 1);
        }
        else {
            const u128 required_b3 =
                partition.s_min - a3;

            first_b =
                std::max({
                    partition.min_b,
                    a + 1,
                    ceil_cuberoot(required_b3)
                });
        }

        // Calculate the exact upper b bound:
        //
        //     a^3 + b^3 < s_max
        //
        // equivalently:
        //
        //     b^3 <= s_max - 1 - a^3
        const u128 maximum_b3 =
            partition.s_max - 1 - a3;

        const u64 last_b =
            std::min(
                partition.max_b,
                floor_cuberoot(maximum_b3));

        if (first_b > last_b)
            continue;

        bounds.push_back({
            static_cast<u32>(a),
            static_cast<u32>(first_b),
            static_cast<u32>(last_b)
        });
    }

    // Generate every (a,b,s) belonging to the pre-computed
    // ranges for this S partition.
    std::vector<pair_record> pairs;

    if (partition.pair_count > pairs.max_size())
        throw std::length_error(
            "partition is too large for vector");

    pairs.reserve(
        static_cast<std::size_t>(
            partition.pair_count));

    for (const auto& bound : bounds) {
        const u128 a3 =
            cube(static_cast<u64>(bound.a));

        // Every pair generated here is guaranteed to belong
        // to this S partition.
        for (u64 b = bound.min_b;
             b <= bound.max_b;
             ++b)
        {
            pairs.push_back({
                a3 + cube(b),
                bound.a,
                static_cast<u32>(b)
            });
        }
    }

    // Useful correctness check during development.
    if (pairs.size() != partition.pair_count) {
        throw std::runtime_error(
            "generated pair count does not match "
            "partition.pair_count");
    }

    // Sort by S. a/b are included as tie-breakers so the
    // result is deterministic.
    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const pair_record& x,
           const pair_record& y)
        {
            // Check S first, then a, then b.
            const u128 x_sum = x.sum();
            const u128 y_sum = y.sum();

            if (x_sum != y_sum)
                return x_sum < y_sum;

            if (x.a != y.a)
                return x.a < y.a;

            return x.b < y.b;
        });

    // Find runs having identical S values.
    std::vector<collision_record> collisions;

    std::size_t i = 0;

    while (i < pairs.size()) {
        std::size_t j = i + 1;
        const u128 sum = pairs[i].sum();

        while (
            j < pairs.size() &&
            pairs[j].sum() == sum)
        {
            ++j;
        }

        const std::size_t representations =
            j - i;

        if (representations >= 2) {
            // If there are R representations of the same S,
            // emit every distinct pair of representations.
            //
            // R = 2 -> 1 collision
            // R = 3 -> 3 collisions
            // R = 4 -> 6 collisions
            for (std::size_t x = i; x < j; ++x) {
                for (std::size_t y = x + 1;
                     y < j;
                     ++y)
                {
                    collisions.push_back({
                        pairs[x].a,
                        pairs[x].b,

                        pairs[y].a,
                        pairs[y].b,

                        make_collision_signature(
                            pairs[x],
                            pairs[y])
                    });
                }
            }
        }

        i = j;
    }

    return collisions;
}
