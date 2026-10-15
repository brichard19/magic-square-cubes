#include "search_gpu.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct gpu_pair_record {
    u64 s_low;
    u64 s_high;
    u32 a;
    u32 b;
};

struct gpu_pair_span {
    u32 a;
    u32 min_b;
    u32 max_b;
    u64 output_offset;
};

struct gpu_collision_input {
    u32 a1;
    u32 b1;
    u32 a2;
    u32 b2;
};

static_assert(sizeof(gpu_pair_record) == 24);
static_assert(sizeof(gpu_collision_input) == 16);

static void check_hip(
    hipError_t result,
    const char* operation)
{
    if (result == hipSuccess)
        return;

    throw std::runtime_error(
        std::string(operation) + ": " +
        hipGetErrorString(result));
}

static unsigned int active_device_wavefront_size()
{
    int device = 0;
    check_hip(
        hipGetDevice(&device),
        "get active HIP device");

    hipDeviceProp_t properties{};
    check_hip(
        hipGetDeviceProperties(&properties, device),
        "get HIP device properties");

    if (properties.warpSize <= 0 ||
        properties.warpSize >
            properties.maxThreadsPerBlock)
    {
        throw std::runtime_error(
            "HIP device reported an invalid wavefront size");
    }

    return static_cast<unsigned int>(
        properties.warpSize);
}

template <typename T>
class device_buffer {
public:
    device_buffer() = default;

    explicit device_buffer(u64 count)
    {
        allocate(count);
    }

    ~device_buffer()
    {
        if (data_ != nullptr)
            (void)hipFree(data_);
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    device_buffer(device_buffer&& other) noexcept
        : data_(other.data_)
    {
        other.data_ = nullptr;
    }

    device_buffer& operator=(device_buffer&& other) noexcept
    {
        if (this == &other)
            return *this;

        if (data_ != nullptr)
            (void)hipFree(data_);

        data_ = other.data_;
        other.data_ = nullptr;
        return *this;
    }

    void allocate(u64 count)
    {
        if (data_ != nullptr)
            throw std::logic_error(
                "device buffer is already allocated");

        if (count == 0)
            return;

        if (count >
            std::numeric_limits<std::size_t>::max() /
                sizeof(T))
        {
            throw std::length_error(
                "GPU allocation size exceeds size_t");
        }

        const std::size_t byte_count =
            static_cast<std::size_t>(count) * sizeof(T);

        check_hip(
            hipMalloc(
                reinterpret_cast<void**>(&data_),
                byte_count),
            "hipMalloc");
    }

    T* get() const noexcept
    {
        return data_;
    }

private:
    T* data_ = nullptr;
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
    const gpu_collision_input& collision)
{
    const u128 x =
        cube(static_cast<u64>(collision.b1)) -
        cube(static_cast<u64>(collision.a1));

    const u128 y =
        cube(static_cast<u64>(collision.b2)) -
        cube(static_cast<u64>(collision.a2));

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

static std::vector<gpu_pair_span> make_pair_spans(
    const search_partition& partition)
{
    std::vector<gpu_pair_span> spans;
    u64 output_offset = 0;

    for (u64 a = partition.min_a;
         a <= partition.max_a;
         ++a)
    {
        const u128 a3 = cube(a);

        if (a3 >= partition.s_max)
            break;

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

        const u128 maximum_b3 =
            partition.s_max - 1 - a3;

        const u64 last_b =
            std::min(
                partition.max_b,
                floor_cuberoot(maximum_b3));

        if (first_b > last_b)
            continue;

        const u64 pair_count =
            last_b - first_b + 1;

        if (pair_count >
            std::numeric_limits<u64>::max() -
                output_offset)
        {
            throw std::overflow_error(
                "GPU pair prefix sum overflow");
        }

        spans.push_back({
            static_cast<u32>(a),
            static_cast<u32>(first_b),
            static_cast<u32>(last_b),
            output_offset
        });

        output_offset += pair_count;
    }

    if (output_offset != partition.pair_count) {
        throw std::runtime_error(
            "GPU pair prefix sum does not match "
            "partition.pair_count");
    }

    return spans;
}

static u64 next_power_of_two(u64 value)
{
    u64 result = 1;

    while (result < value) {
        if (result >
            std::numeric_limits<u64>::max() / 2)
        {
            throw std::overflow_error(
                "bitonic sort size overflow");
        }

        result *= 2;
    }

    return result;
}

static dim3 one_dimensional_grid(
    u64 item_count,
    unsigned int threads_per_block)
{
    const u64 block_count =
        (item_count + threads_per_block - 1) /
        threads_per_block;

    if (block_count >
        std::numeric_limits<unsigned int>::max())
    {
        throw std::length_error(
            "GPU grid requires too many blocks");
    }

    return dim3(static_cast<unsigned int>(block_count));
}

__device__ u128 cube_gpu(u32 x)
{
    return u128(x) * x * x;
}

__device__ gpu_pair_record add_cubes_gpu(
    u32 a,
    u32 b)
{
    const u128 sum =
        cube_gpu(a) + cube_gpu(b);

    return {
        static_cast<u64>(sum),
        static_cast<u64>(sum >> 64),
        a,
        b
    };
}

__device__ bool sums_equal(
    const gpu_pair_record& left,
    const gpu_pair_record& right)
{
    return
        left.s_high == right.s_high &&
        left.s_low == right.s_low;
}

__device__ bool pair_less(
    const gpu_pair_record& left,
    const gpu_pair_record& right)
{
    if (left.s_high != right.s_high)
        return left.s_high < right.s_high;

    if (left.s_low != right.s_low)
        return left.s_low < right.s_low;

    if (left.a != right.a)
        return left.a < right.a;

    return left.b < right.b;
}

__global__ void generate_pairs_kernel(
    const gpu_pair_span* spans,
    gpu_pair_record* pairs)
{
    const gpu_pair_span span = spans[blockIdx.x];
    const u64 pair_count =
        u64(span.max_b) - span.min_b + 1;

    for (u64 local_index = threadIdx.x;
         local_index < pair_count;
         local_index += blockDim.x)
    {
        const u32 b =
            span.min_b +
            static_cast<u32>(local_index);

        pairs[span.output_offset + local_index] =
            add_cubes_gpu(span.a, b);
    }
}

__global__ void bitonic_sort_step_kernel(
    gpu_pair_record* pairs,
    u64 pair_count,
    u64 sequence_length,
    u64 comparison_stride)
{
    const u64 index =
        u64(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index >= pair_count)
        return;

    const u64 partner = index ^ comparison_stride;

    if (partner <= index || partner >= pair_count)
        return;

    const bool ascending =
        (index & sequence_length) == 0;

    const gpu_pair_record left = pairs[index];
    const gpu_pair_record right = pairs[partner];

    const bool swap = ascending
        ? pair_less(right, left)
        : pair_less(left, right);

    if (swap) {
        pairs[index] = right;
        pairs[partner] = left;
    }
}

__global__ void count_collisions_kernel(
    const gpu_pair_record* pairs,
    u64 pair_count,
    unsigned long long* collision_count)
{
    const u64 index =
        u64(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index >= pair_count)
        return;

    if (index > 0 &&
        sums_equal(pairs[index - 1], pairs[index]))
    {
        return;
    }

    u64 end = index + 1;

    while (end < pair_count &&
           sums_equal(pairs[index], pairs[end]))
    {
        ++end;
    }

    const u64 representations = end - index;

    if (representations >= 2) {
        const u64 count =
            representations *
            (representations - 1) / 2;

        atomicAdd(
            collision_count,
            static_cast<unsigned long long>(count));
    }
}

__global__ void collect_collisions_kernel(
    const gpu_pair_record* pairs,
    u64 pair_count,
    gpu_collision_input* collisions,
    unsigned long long* next_collision)
{
    const u64 index =
        u64(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index >= pair_count)
        return;

    if (index > 0 &&
        sums_equal(pairs[index - 1], pairs[index]))
    {
        return;
    }

    u64 end = index + 1;

    while (end < pair_count &&
           sums_equal(pairs[index], pairs[end]))
    {
        ++end;
    }

    const u64 representations = end - index;

    if (representations < 2)
        return;

    const u64 run_collision_count =
        representations *
        (representations - 1) / 2;

    u64 output_index = atomicAdd(
        next_collision,
        static_cast<unsigned long long>(
            run_collision_count));

    for (u64 first = index; first < end; ++first) {
        for (u64 second = first + 1;
             second < end;
             ++second)
        {
            collisions[output_index++] = {
                pairs[first].a,
                pairs[first].b,
                pairs[second].a,
                pairs[second].b
            };
        }
    }
}

} // namespace

std::size_t gpu_pair_record_size() noexcept
{
    return sizeof(gpu_pair_record);
}

u64 gpu_pair_record_max_count() noexcept
{
    return static_cast<u64>(
        std::numeric_limits<std::size_t>::max() /
        sizeof(gpu_pair_record));
}

std::vector<collision_record> search_partition_collisions_gpu(
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

    const unsigned int threads_per_block =
        active_device_wavefront_size();

    const auto spans = make_pair_spans(partition);

    if (spans.empty())
        return {};

    if (spans.size() >
        std::numeric_limits<unsigned int>::max())
    {
        throw std::length_error(
            "GPU partition has too many a spans");
    }

    const u64 padded_pair_count =
        next_power_of_two(partition.pair_count);

    device_buffer<gpu_pair_span> device_spans(
        static_cast<u64>(spans.size()));
    device_buffer<gpu_pair_record> device_pairs(
        padded_pair_count);

    check_hip(
        hipMemcpy(
            device_spans.get(),
            spans.data(),
            spans.size() * sizeof(gpu_pair_span),
            hipMemcpyHostToDevice),
        "copy pair spans to GPU");

    // All-ones is a sentinel larger than every valid sum and
    // therefore sorts after the real pair records.
    check_hip(
        hipMemset(
            device_pairs.get(),
            0xff,
            static_cast<std::size_t>(padded_pair_count) *
                sizeof(gpu_pair_record)),
        "initialize padded GPU pair buffer");

    hipLaunchKernelGGL(
        generate_pairs_kernel,
        dim3(static_cast<unsigned int>(spans.size())),
        dim3(threads_per_block),
        0,
        0,
        device_spans.get(),
        device_pairs.get());
    check_hip(
        hipGetLastError(),
        "launch pair-generation kernel");

    const dim3 sort_grid =
        one_dimensional_grid(
            padded_pair_count,
            threads_per_block);

    for (u64 sequence_length = 2;
         sequence_length <= padded_pair_count;
         sequence_length *= 2)
    {
        for (u64 comparison_stride =
                 sequence_length / 2;
             comparison_stride > 0;
             comparison_stride /= 2)
        {
            hipLaunchKernelGGL(
                bitonic_sort_step_kernel,
                sort_grid,
                dim3(threads_per_block),
                0,
                0,
                device_pairs.get(),
                padded_pair_count,
                sequence_length,
                comparison_stride);
            check_hip(
                hipGetLastError(),
                "launch bitonic-sort kernel");
        }

        if (sequence_length == padded_pair_count)
            break;
    }

    device_buffer<unsigned long long>
        device_collision_count(1);

    check_hip(
        hipMemset(
            device_collision_count.get(),
            0,
            sizeof(unsigned long long)),
        "initialize GPU collision counter");

    const dim3 pair_grid =
        one_dimensional_grid(
            partition.pair_count,
            threads_per_block);

    hipLaunchKernelGGL(
        count_collisions_kernel,
        pair_grid,
        dim3(threads_per_block),
        0,
        0,
        device_pairs.get(),
        partition.pair_count,
        device_collision_count.get());
    check_hip(
        hipGetLastError(),
        "launch collision-count kernel");

    unsigned long long collision_count = 0;

    check_hip(
        hipMemcpy(
            &collision_count,
            device_collision_count.get(),
            sizeof(collision_count),
            hipMemcpyDeviceToHost),
        "copy collision count from GPU");

    if (collision_count == 0)
        return {};

    if (collision_count >
        std::vector<gpu_collision_input>{}.max_size())
    {
        throw std::length_error(
            "GPU collision result is too large");
    }

    device_buffer<gpu_collision_input>
        device_collisions(collision_count);

    check_hip(
        hipMemset(
            device_collision_count.get(),
            0,
            sizeof(unsigned long long)),
        "reset GPU collision counter");

    hipLaunchKernelGGL(
        collect_collisions_kernel,
        pair_grid,
        dim3(threads_per_block),
        0,
        0,
        device_pairs.get(),
        partition.pair_count,
        device_collisions.get(),
        device_collision_count.get());
    check_hip(
        hipGetLastError(),
        "launch collision-collection kernel");

    std::vector<gpu_collision_input> collision_inputs(
        static_cast<std::size_t>(collision_count));

    check_hip(
        hipMemcpy(
            collision_inputs.data(),
            device_collisions.get(),
            collision_inputs.size() *
                sizeof(gpu_collision_input),
            hipMemcpyDeviceToHost),
        "copy collisions from GPU");

    unsigned long long emitted_collision_count = 0;

    check_hip(
        hipMemcpy(
            &emitted_collision_count,
            device_collision_count.get(),
            sizeof(emitted_collision_count),
            hipMemcpyDeviceToHost),
        "copy emitted collision count from GPU");

    if (emitted_collision_count != collision_count) {
        throw std::runtime_error(
            "GPU collision count changed during collection");
    }

    std::vector<collision_record> collisions;
    collisions.reserve(collision_inputs.size());

    for (const auto& input : collision_inputs) {
        collisions.push_back({
            input.a1,
            input.b1,
            input.a2,
            input.b2,
            make_collision_signature(input)
        });
    }

    return collisions;
}
