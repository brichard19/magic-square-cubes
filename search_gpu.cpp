#include "search_gpu.h"

#include <hip/hip_runtime.h>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/iterator/zip_iterator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t gpu_pair_storage_size =
    sizeof(u128) + 2 * sizeof(u32);

// At a maximum load factor of 70%, the hash table uses about
// 45.7 bytes per pair: 16 bytes for the sum, 8 for its tag, and
// 4 each for a and b.  Round up to a conservative partitioning
// guideline that leaves a little room for the span array.
constexpr std::size_t gpu_hash_pair_storage_size = 48;
constexpr u64 hash_load_numerator = 7;
constexpr u64 hash_load_denominator = 10;
constexpr unsigned long long hash_empty_tag = 0;
constexpr unsigned long long hash_reserved_tag = 1;

int configured_hip_device = 0;

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

static_assert(gpu_pair_storage_size == 24);
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
    check_hip(
        hipSetDevice(configured_hip_device),
        "select HIP device");

    hipDeviceProp_t properties{};
    check_hip(
        hipGetDeviceProperties(
            &properties,
            configured_hip_device),
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

static std::vector<collision_record> make_collision_records(
    const std::vector<gpu_collision_input>& inputs)
{
    std::vector<collision_record> collisions;
    collisions.reserve(inputs.size());

    for (const auto& input : inputs) {
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

static u64 hash_table_capacity(u64 pair_count)
{
    const u128 scaled =
        u128(pair_count) * hash_load_denominator;

    const u128 capacity =
        (scaled + hash_load_numerator - 1) /
        hash_load_numerator;

    if (capacity > std::numeric_limits<u64>::max()) {
        throw std::length_error(
            "GPU hash-table capacity overflow");
    }

    return std::max<u64>(
        static_cast<u64>(capacity),
        2);
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

__device__ u128 add_cubes_gpu(
    u32 a,
    u32 b)
{
    return cube_gpu(a) + cube_gpu(b);
}

__global__ void generate_pairs_kernel(
    const gpu_pair_span* spans,
    u128* sums,
    u32* a_values,
    u32* b_values)
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

        const u64 output_index =
            span.output_offset + local_index;

        sums[output_index] =
            add_cubes_gpu(span.a, b);
        a_values[output_index] = span.a;
        b_values[output_index] = b;
    }
}

__device__ unsigned long long mix_hash_gpu(
    unsigned long long value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value =
        (value ^ (value >> 30)) *
        0xbf58476d1ce4e5b9ULL;
    value =
        (value ^ (value >> 27)) *
        0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ unsigned long long hash_sum_gpu(u128 sum)
{
    const unsigned long long low =
        static_cast<unsigned long long>(sum);
    const unsigned long long high =
        static_cast<unsigned long long>(sum >> 64);

    return mix_hash_gpu(
        low ^ mix_hash_gpu(
            high + 0x243f6a8885a308d3ULL));
}

__device__ unsigned long long hash_tag_gpu(
    unsigned long long hash)
{
    return hash <= hash_reserved_tag
        ? hash + 2
        : hash;
}

__device__ u64 next_hash_slot_gpu(
    u64 slot,
    u64 capacity)
{
    ++slot;
    return slot == capacity ? 0 : slot;
}

__global__ void generate_hash_table_kernel(
    const gpu_pair_span* spans,
    u64 table_capacity,
    unsigned long long* tags,
    u128* sums,
    u32* a_values,
    u32* b_values,
    unsigned int* error_flag)
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
        const u128 sum = add_cubes_gpu(span.a, b);
        const unsigned long long hash =
            hash_sum_gpu(sum);
        const unsigned long long tag =
            hash_tag_gpu(hash);
        u64 slot = hash % table_capacity;
        bool inserted = false;

        for (u64 probe = 0;
             probe < table_capacity;
             ++probe)
        {
            const unsigned long long old_tag =
                atomicCAS(
                    tags + slot,
                    hash_empty_tag,
                    hash_reserved_tag);

            if (old_tag == hash_empty_tag) {
                sums[slot] = sum;
                a_values[slot] = span.a;
                b_values[slot] = b;

                // Publish the fingerprint only after the entry data.
                __threadfence();
                atomicExch(tags + slot, tag);
                inserted = true;
                break;
            }

            // A reserved slot is treated as occupied instead of being
            // waited on.  This avoids an intra-wavefront spinlock
            // deadlock; the next kernel runs only after all entries
            // have been published.
            slot = next_hash_slot_gpu(
                slot,
                table_capacity);
        }

        if (!inserted)
            atomicExch(error_flag, 1U);
    }
}

__global__ void count_hash_collisions_kernel(
    const unsigned long long* tags,
    const u128* sums,
    u64 table_capacity,
    unsigned long long* collision_count,
    unsigned int* error_flag)
{
    const u64 entry_slot =
        u64(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (entry_slot >= table_capacity)
        return;

    const unsigned long long entry_tag =
        tags[entry_slot];

    if (entry_tag == hash_empty_tag)
        return;

    if (entry_tag == hash_reserved_tag) {
        atomicExch(error_flag, 1U);
        return;
    }

    const u128 sum = sums[entry_slot];
    u64 slot =
        hash_sum_gpu(sum) % table_capacity;
    unsigned long long matches = 0;

    for (u64 probe = 0;
         probe < table_capacity;
         ++probe)
    {
        if (slot == entry_slot) {
            if (matches != 0) {
                atomicAdd(
                    collision_count,
                    matches);
            }
            return;
        }

        const unsigned long long candidate_tag =
            tags[slot];

        if (candidate_tag == hash_empty_tag ||
            candidate_tag == hash_reserved_tag)
        {
            atomicExch(error_flag, 1U);
            return;
        }

        if (candidate_tag == entry_tag &&
            sums[slot] == sum)
        {
            ++matches;
        }

        slot = next_hash_slot_gpu(
            slot,
            table_capacity);
    }

    atomicExch(error_flag, 1U);
}

__global__ void collect_hash_collisions_kernel(
    const unsigned long long* tags,
    const u128* sums,
    const u32* a_values,
    const u32* b_values,
    u64 table_capacity,
    gpu_collision_input* collisions,
    u64 collision_capacity,
    unsigned long long* next_collision,
    unsigned int* error_flag)
{
    const u64 entry_slot =
        u64(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (entry_slot >= table_capacity)
        return;

    const unsigned long long entry_tag =
        tags[entry_slot];

    if (entry_tag == hash_empty_tag)
        return;

    if (entry_tag == hash_reserved_tag) {
        atomicExch(error_flag, 1U);
        return;
    }

    const u128 sum = sums[entry_slot];
    u64 slot =
        hash_sum_gpu(sum) % table_capacity;

    for (u64 probe = 0;
         probe < table_capacity;
         ++probe)
    {
        if (slot == entry_slot)
            return;

        const unsigned long long candidate_tag =
            tags[slot];

        if (candidate_tag == hash_empty_tag ||
            candidate_tag == hash_reserved_tag)
        {
            atomicExch(error_flag, 1U);
            return;
        }

        if (candidate_tag == entry_tag &&
            sums[slot] == sum)
        {
            const unsigned long long output_index =
                atomicAdd(next_collision, 1ULL);

            if (output_index < collision_capacity) {
                collisions[output_index] = {
                    a_values[slot],
                    b_values[slot],
                    a_values[entry_slot],
                    b_values[entry_slot]
                };
            }
            else {
                atomicExch(error_flag, 1U);
            }
        }

        slot = next_hash_slot_gpu(
            slot,
            table_capacity);
    }

    atomicExch(error_flag, 1U);
}

__global__ void count_collisions_kernel(
    const u128* sums,
    u64 pair_count,
    unsigned long long* collision_count)
{
    const u64 index =
        u64(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index >= pair_count)
        return;

    if (index > 0 &&
        sums[index - 1] == sums[index])
    {
        return;
    }

    u64 end = index + 1;

    while (end < pair_count &&
           sums[index] == sums[end])
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
    const u128* sums,
    const u32* a_values,
    const u32* b_values,
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
        sums[index - 1] == sums[index])
    {
        return;
    }

    u64 end = index + 1;

    while (end < pair_count &&
           sums[index] == sums[end])
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
                a_values[first],
                b_values[first],
                a_values[second],
                b_values[second]
            };
        }
    }
}

} // namespace

std::size_t gpu_pair_record_size() noexcept
{
    return gpu_pair_storage_size;
}

u64 gpu_pair_record_max_count() noexcept
{
    return static_cast<u64>(
        std::numeric_limits<std::size_t>::max() /
        gpu_pair_storage_size);
}

std::size_t gpu_hash_pair_record_size() noexcept
{
    return gpu_hash_pair_storage_size;
}

u64 gpu_hash_pair_record_max_count() noexcept
{
    return static_cast<u64>(
        std::numeric_limits<std::size_t>::max() /
        gpu_hash_pair_storage_size);
}

void set_hip_device(int device)
{
    if (device < 0) {
        throw std::invalid_argument(
            "HIP device index must not be negative");
    }

    configured_hip_device = device;
}

std::string hip_device_identifier(int device)
{
    if (device < 0) {
        throw std::invalid_argument(
            "HIP device index must not be negative");
    }

    hipDeviceProp_t properties{};
    check_hip(
        hipGetDeviceProperties(&properties, device),
        "get HIP device properties");

    bool uuid_is_nonzero = false;
    bool uuid_is_safe_text = true;

    for (const char byte : properties.uuid.bytes) {
        if (byte != 0) {
            uuid_is_nonzero = true;
        }

        const unsigned char character =
            static_cast<unsigned char>(byte);

        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'z') ||
              (character >= 'A' && character <= 'Z') ||
              character == '-' || character == '_'))
        {
            uuid_is_safe_text = false;
        }
    }

    // AMD's HIP runtime commonly exposes the GPU serial number as 16
    // printable bytes (for example, "295cea3615cef364").  Preserve that
    // representation so directory names match the other GPU tools.
    if (uuid_is_nonzero && uuid_is_safe_text) {
        return std::string(
            properties.uuid.bytes,
            sizeof(properties.uuid.bytes));
    }

    if (uuid_is_nonzero) {
        static constexpr char hex[] =
            "0123456789abcdef";

        std::string identifier;
        identifier.reserve(
            2 * sizeof(properties.uuid.bytes));

        for (std::size_t i = 0;
             i < sizeof(properties.uuid.bytes);
             ++i)
        {
            const unsigned int byte =
                static_cast<unsigned char>(
                    properties.uuid.bytes[i]);

            identifier.push_back(hex[byte >> 4]);
            identifier.push_back(hex[byte & 0x0f]);
        }

        return identifier;
    }

    char identifier[64];
    std::snprintf(
        identifier,
        sizeof(identifier),
        "pci-%04x-%02x-%02x",
        static_cast<unsigned int>(
            properties.pciDomainID),
        static_cast<unsigned int>(
            properties.pciBusID),
        static_cast<unsigned int>(
            properties.pciDeviceID));

    return identifier;
}

std::vector<collision_record>
search_partition_collisions_gpu_radix(
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

    if (partition.pair_count >
        std::numeric_limits<std::size_t>::max())
    {
        throw std::length_error(
            "GPU partition exceeds the host size_t range");
    }

    const std::size_t pair_count =
        static_cast<std::size_t>(
            partition.pair_count);

    const dim3 generation_grid(
        static_cast<unsigned int>(spans.size()));
    const dim3 collision_grid =
        one_dimensional_grid(
            partition.pair_count,
            threads_per_block);

    device_buffer<gpu_pair_span> device_spans(
        static_cast<u64>(spans.size()));
    device_buffer<u128> device_sums(partition.pair_count);
    device_buffer<u32> device_a_values(partition.pair_count);
    device_buffer<u32> device_b_values(partition.pair_count);

    device_buffer<u128> device_sorted_sums(
        partition.pair_count);
    device_buffer<u32> device_sorted_a_values(
        partition.pair_count);
    device_buffer<u32> device_sorted_b_values(
        partition.pair_count);

    const auto input_values =
        rocprim::make_zip_iterator(
            rocprim::make_tuple(
                device_a_values.get(),
                device_b_values.get()));

    const auto output_values =
        rocprim::make_zip_iterator(
            rocprim::make_tuple(
                device_sorted_a_values.get(),
                device_sorted_b_values.get()));

    std::size_t radix_temporary_storage_size = 0;

    check_hip(
        rocprim::radix_sort_pairs(
            nullptr,
            radix_temporary_storage_size,
            device_sums.get(),
            device_sorted_sums.get(),
            input_values,
            output_values,
            pair_count,
            0,
            8 * sizeof(u128)),
        "query rocPRIM radix-sort storage");

    device_buffer<unsigned char> radix_temporary_storage(
        static_cast<u64>(
            std::max<std::size_t>(
                radix_temporary_storage_size,
                1)));

    std::printf(
        "GPU pair generation: grid=%u blocks, "
        "block=%u threads\n"
        "GPU radix sort: rocPRIM-managed geometry, "
        "pairs=%llu, temporary_storage=%zu bytes\n"
        "GPU collision scan: grid=%u blocks, "
        "block=%u threads\n",
        generation_grid.x,
        threads_per_block,
        (unsigned long long)partition.pair_count,
        radix_temporary_storage_size,
        collision_grid.x,
        threads_per_block);
    std::fflush(stdout);

    check_hip(
        hipMemcpy(
            device_spans.get(),
            spans.data(),
            spans.size() * sizeof(gpu_pair_span),
            hipMemcpyHostToDevice),
        "copy pair spans to GPU");

    hipLaunchKernelGGL(
        generate_pairs_kernel,
        generation_grid,
        dim3(threads_per_block),
        0,
        0,
        device_spans.get(),
        device_sums.get(),
        device_a_values.get(),
        device_b_values.get());
    check_hip(
        hipGetLastError(),
        "launch pair-generation kernel");

    check_hip(
        rocprim::radix_sort_pairs(
            radix_temporary_storage.get(),
            radix_temporary_storage_size,
            device_sums.get(),
            device_sorted_sums.get(),
            input_values,
            output_values,
            pair_count,
            0,
            8 * sizeof(u128)),
        "run rocPRIM radix sort");

    device_buffer<unsigned long long>
        device_collision_count(1);

    check_hip(
        hipMemset(
            device_collision_count.get(),
            0,
            sizeof(unsigned long long)),
        "initialize GPU collision counter");

    hipLaunchKernelGGL(
        count_collisions_kernel,
        collision_grid,
        dim3(threads_per_block),
        0,
        0,
        device_sorted_sums.get(),
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
        collision_grid,
        dim3(threads_per_block),
        0,
        0,
        device_sorted_sums.get(),
        device_sorted_a_values.get(),
        device_sorted_b_values.get(),
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

    return make_collision_records(collision_inputs);
}

std::vector<collision_record>
search_partition_collisions_gpu_hash(
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

    if (partition.pair_count >
        gpu_hash_pair_record_max_count())
    {
        throw std::length_error(
            "GPU hash partition exceeds the host size_t range");
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

    const u64 table_capacity =
        hash_table_capacity(partition.pair_count);

    if (table_capacity >
        std::numeric_limits<std::size_t>::max() /
            sizeof(u128))
    {
        throw std::length_error(
            "GPU hash table exceeds the host size_t range");
    }

    const dim3 insertion_grid(
        static_cast<unsigned int>(spans.size()));
    const dim3 table_grid =
        one_dimensional_grid(
            table_capacity,
            threads_per_block);

    device_buffer<gpu_pair_span> device_spans(
        static_cast<u64>(spans.size()));
    device_buffer<unsigned long long> device_tags(
        table_capacity);
    device_buffer<u128> device_sums(table_capacity);
    device_buffer<u32> device_a_values(table_capacity);
    device_buffer<u32> device_b_values(table_capacity);
    device_buffer<unsigned long long>
        device_collision_count(1);
    device_buffer<unsigned int> device_error_flag(1);

    const long double table_bytes =
        static_cast<long double>(table_capacity) *
        (sizeof(unsigned long long) + sizeof(u128) +
         2 * sizeof(u32));
    const double load_factor =
        static_cast<double>(partition.pair_count) /
        static_cast<double>(table_capacity);

    std::printf(
        "GPU hash insertion: grid=%u blocks, "
        "block=%u threads\n"
        "GPU hash table: pairs=%llu, slots=%llu, "
        "load=%.2f%%, storage=%.0Lf bytes\n"
        "GPU hash scan: grid=%u blocks, "
        "block=%u threads\n",
        insertion_grid.x,
        threads_per_block,
        (unsigned long long)partition.pair_count,
        (unsigned long long)table_capacity,
        100.0 * load_factor,
        table_bytes,
        table_grid.x,
        threads_per_block);
    std::fflush(stdout);

    check_hip(
        hipMemcpy(
            device_spans.get(),
            spans.data(),
            spans.size() * sizeof(gpu_pair_span),
            hipMemcpyHostToDevice),
        "copy pair spans to GPU");

    check_hip(
        hipMemset(
            device_tags.get(),
            0,
            static_cast<std::size_t>(table_capacity) *
                sizeof(unsigned long long)),
        "initialize GPU hash tags");
    check_hip(
        hipMemset(
            device_error_flag.get(),
            0,
            sizeof(unsigned int)),
        "initialize GPU hash error flag");

    hipLaunchKernelGGL(
        generate_hash_table_kernel,
        insertion_grid,
        dim3(threads_per_block),
        0,
        0,
        device_spans.get(),
        table_capacity,
        device_tags.get(),
        device_sums.get(),
        device_a_values.get(),
        device_b_values.get(),
        device_error_flag.get());
    check_hip(
        hipGetLastError(),
        "launch GPU hash-insertion kernel");

    unsigned int error_flag = 0;
    check_hip(
        hipMemcpy(
            &error_flag,
            device_error_flag.get(),
            sizeof(error_flag),
            hipMemcpyDeviceToHost),
        "copy GPU hash insertion status");

    if (error_flag != 0) {
        throw std::runtime_error(
            "GPU hash insertion failed");
    }

    check_hip(
        hipMemset(
            device_collision_count.get(),
            0,
            sizeof(unsigned long long)),
        "initialize GPU hash collision counter");

    hipLaunchKernelGGL(
        count_hash_collisions_kernel,
        table_grid,
        dim3(threads_per_block),
        0,
        0,
        device_tags.get(),
        device_sums.get(),
        table_capacity,
        device_collision_count.get(),
        device_error_flag.get());
    check_hip(
        hipGetLastError(),
        "launch GPU hash collision-count kernel");

    unsigned long long collision_count = 0;
    check_hip(
        hipMemcpy(
            &collision_count,
            device_collision_count.get(),
            sizeof(collision_count),
            hipMemcpyDeviceToHost),
        "copy GPU hash collision count");
    check_hip(
        hipMemcpy(
            &error_flag,
            device_error_flag.get(),
            sizeof(error_flag),
            hipMemcpyDeviceToHost),
        "copy GPU hash scan status");

    if (error_flag != 0) {
        throw std::runtime_error(
            "GPU hash collision scan found an invalid probe chain");
    }

    if (collision_count == 0)
        return {};

    if (collision_count >
        std::vector<gpu_collision_input>{}.max_size())
    {
        throw std::length_error(
            "GPU hash collision result is too large");
    }

    device_buffer<gpu_collision_input>
        device_collisions(collision_count);

    check_hip(
        hipMemset(
            device_collision_count.get(),
            0,
            sizeof(unsigned long long)),
        "reset GPU hash collision counter");

    hipLaunchKernelGGL(
        collect_hash_collisions_kernel,
        table_grid,
        dim3(threads_per_block),
        0,
        0,
        device_tags.get(),
        device_sums.get(),
        device_a_values.get(),
        device_b_values.get(),
        table_capacity,
        device_collisions.get(),
        collision_count,
        device_collision_count.get(),
        device_error_flag.get());
    check_hip(
        hipGetLastError(),
        "launch GPU hash collision-collection kernel");

    std::vector<gpu_collision_input> collision_inputs(
        static_cast<std::size_t>(collision_count));

    check_hip(
        hipMemcpy(
            collision_inputs.data(),
            device_collisions.get(),
            collision_inputs.size() *
                sizeof(gpu_collision_input),
            hipMemcpyDeviceToHost),
        "copy hash collisions from GPU");

    unsigned long long emitted_collision_count = 0;
    check_hip(
        hipMemcpy(
            &emitted_collision_count,
            device_collision_count.get(),
            sizeof(emitted_collision_count),
            hipMemcpyDeviceToHost),
        "copy emitted GPU hash collision count");
    check_hip(
        hipMemcpy(
            &error_flag,
            device_error_flag.get(),
            sizeof(error_flag),
            hipMemcpyDeviceToHost),
        "copy GPU hash collection status");

    if (error_flag != 0) {
        throw std::runtime_error(
            "GPU hash collision collection found an invalid "
            "probe chain");
    }

    if (emitted_collision_count != collision_count) {
        throw std::runtime_error(
            "GPU hash collision count changed during collection");
    }

    return make_collision_records(collision_inputs);
}
