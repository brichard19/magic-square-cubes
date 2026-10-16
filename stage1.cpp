#include "search.h"
#include "search_cpu.h"
#ifdef MAGIC_SQUARE_CUBES_ENABLE_HIP
#include "search_gpu.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

struct partition_stats {
    u64 min_a = 0;
    u64 min_b = 0;
    u64 max_a = 0;
    u64 max_b = 0;
    u64 pair_count = 0;
};

static bool parse_range(
    std::string_view text,
    u64& minimum,
    u64& maximum)
{
    const std::size_t separator = text.find(':');

    if (separator == std::string_view::npos ||
        separator == 0 ||
        separator + 1 == text.size() ||
        text.find(':', separator + 1) !=
            std::string_view::npos)
    {
        return false;
    }

    const char* const begin = text.data();
    const char* const middle = begin + separator;
    const char* const end = begin + text.size();

    u64 parsed_minimum = 0;
    u64 parsed_maximum = 0;

    const auto minimum_result =
        std::from_chars(begin, middle, parsed_minimum);

    const auto maximum_result =
        std::from_chars(middle + 1, end, parsed_maximum);

    if (minimum_result.ec != std::errc{} ||
        minimum_result.ptr != middle ||
        maximum_result.ec != std::errc{} ||
        maximum_result.ptr != end ||
        parsed_minimum > parsed_maximum)
    {
        return false;
    }

    minimum = parsed_minimum;
    maximum = parsed_maximum;
    return true;
}

static bool parse_buffer_size(
    std::string_view text,
    u64& size_bytes)
{
    if (text.empty())
        return false;

    u64 value = 0;

    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result =
        std::from_chars(begin, end, value);

    if (result.ec != std::errc{} ||
        result.ptr == begin ||
        value == 0)
    {
        return false;
    }

    const std::string_view suffix(
        result.ptr,
        static_cast<std::size_t>(end - result.ptr));

    u64 multiplier = 0;

    if (suffix.empty() || suffix == "B" || suffix == "b") {
        multiplier = 1;
    }
    else if (suffix == "K" || suffix == "k" ||
             suffix == "KB" || suffix == "kb" ||
             suffix == "KiB" || suffix == "kib")
    {
        multiplier = u64(1) << 10;
    }
    else if (suffix == "M" || suffix == "m" ||
             suffix == "MB" || suffix == "mb" ||
             suffix == "MiB" || suffix == "mib")
    {
        multiplier = u64(1) << 20;
    }
    else if (suffix == "G" || suffix == "g" ||
             suffix == "GB" || suffix == "gb" ||
             suffix == "GiB" || suffix == "gib")
    {
        multiplier = u64(1) << 30;
    }
    else {
        return false;
    }

    if (value >
        std::numeric_limits<u64>::max() / multiplier)
    {
        return false;
    }

    size_bytes = value * multiplier;
    return true;
}

static bool parse_thread_count(
    std::string_view text,
    std::size_t& thread_count)
{
    if (text.empty())
        return false;

    u64 value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result =
        std::from_chars(begin, end, value);

    if (result.ec != std::errc{} ||
        result.ptr != end ||
        value == 0 ||
        value > std::numeric_limits<std::size_t>::max())
    {
        return false;
    }

    thread_count = static_cast<std::size_t>(value);
    return true;
}

static bool parse_hip_device(
    std::string_view text,
    int& device)
{
    if (text.empty())
        return false;

    u64 value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result =
        std::from_chars(begin, end, value);

    if (result.ec != std::errc{} ||
        result.ptr != end ||
        value > static_cast<u64>(
            std::numeric_limits<int>::max()))
    {
        return false;
    }

    device = static_cast<int>(value);
    return true;
}

static void print_usage(const char* program)
{
    std::fprintf(
        stderr,
        "Usage: %s [-a MIN:MAX] [-b MIN:MAX] [-m SIZE] [-j N] [-d N] "
        "[--backend cpu|gpu]\n"
        "  -m, --buffer-size SIZE  target pair buffer size; B, KB, MB, GB use powers of 1024\n"
        "  -j, --threads N         worker threads; pair buffer applies to each worker\n"
        "  -d, --hip-device N      HIP device index for the GPU backend\n"
        "      --backend TYPE      search backend: cpu or gpu\n"
        "Defaults: -a 1:100000 -b 2:100000 -m 1GB -j 1 -d 0 --backend cpu\n",
        program);
}

static inline u128 cube(u64 x)
{
    return u128(x) * x * x;
}

static u64 mix_u64(u64 value)
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

static u64 hash_collision_signature(
    const collision_signature& signature)
{
    const u64 u_low = static_cast<u64>(signature.u);
    const u64 u_high = static_cast<u64>(signature.u >> 64);
    const u64 v_low = static_cast<u64>(signature.v);
    const u64 v_high = static_cast<u64>(signature.v >> 64);

    u64 hash = mix_u64(u_low);
    hash ^= mix_u64(u_high + 0x243f6a8885a308d3ULL);
    hash ^= mix_u64(v_low + 0x13198a2e03707344ULL);
    hash ^= mix_u64(v_high + 0xa4093822299f31d0ULL);
    return mix_u64(hash);
}

static std::filesystem::path collision_output_directory(
    std::string_view backend_directory)
{
    std::array<char, 256> hostname_buffer{};

    if (gethostname(
            hostname_buffer.data(),
            hostname_buffer.size()) != 0)
    {
        throw std::system_error(
            errno,
            std::generic_category(),
            "unable to determine hostname");
    }

    hostname_buffer.back() = '\0';
    std::string hostname(hostname_buffer.data());

    if (hostname.empty())
        hostname = "unknown-host";

    for (char& character : hostname) {
        const bool is_letter =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z');

        const bool is_digit =
            character >= '0' && character <= '9';

        if (!is_letter && !is_digit &&
            character != '-' &&
            character != '_' &&
            character != '.')
        {
            character = '_';
        }
    }

    return
        std::filesystem::path(hostname) /
        backend_directory;
}

class collision_bucket_writer {
public:
    static constexpr std::size_t bucket_count = 32;
    static constexpr std::size_t bucket_size_bytes =
        std::size_t(1) << 20;
    static constexpr std::size_t records_per_bucket =
        bucket_size_bytes / sizeof(collision_record);

    explicit collision_bucket_writer(
        std::string_view backend_directory)
        : output_directory_(
            collision_output_directory(
                backend_directory))
    {
        static_assert(records_per_bucket > 0);
        static_assert(
            std::is_trivially_copyable_v<collision_record>);

        std::filesystem::create_directories(
            output_directory_);

        for (std::size_t i = 0; i < bucket_count; ++i) {
            buckets_[i].reserve(records_per_bucket);

            const std::filesystem::path filename =
                output_directory_ /
                bucket_filename(i);

            files_[i].open(
                filename,
                std::ios::binary |
                std::ios::app);

            if (!files_[i]) {
                throw std::runtime_error(
                    "unable to open " +
                    filename.string());
            }
        }
    }

    const std::filesystem::path& output_directory() const
    {
        return output_directory_;
    }

    void add(const collision_record& collision)
    {
        const std::size_t index =
            static_cast<std::size_t>(
                hash_collision_signature(
                    collision.signature) %
                bucket_count);

        auto& bucket = buckets_[index];
        bucket.push_back(collision);

        if (bucket.size() >= records_per_bucket)
            flush_bucket(index);
    }

    void flush_all()
    {
        for (std::size_t i = 0; i < bucket_count; ++i)
            flush_bucket(i);
    }

private:
    static std::string bucket_filename(std::size_t index)
    {
        char filename[64];

        std::snprintf(
            filename,
            sizeof(filename),
            "collision-bucket-u32-v1-%02zu.bin",
            index);

        return filename;
    }

    void flush_bucket(std::size_t index)
    {
        auto& bucket = buckets_[index];

        if (bucket.empty())
            return;

        auto& file = files_[index];
        const std::size_t byte_count =
            bucket.size() * sizeof(collision_record);

        file.write(
            reinterpret_cast<const char*>(bucket.data()),
            static_cast<std::streamsize>(byte_count));
        file.flush();

        if (!file) {
            throw std::runtime_error(
                "unable to write " +
                (output_directory_ /
                    bucket_filename(index)).string());
        }

        bucket.clear();
    }

    std::filesystem::path output_directory_;

    std::array<
        std::vector<collision_record>,
        bucket_count> buckets_;

    std::array<std::ofstream, bucket_count> files_;
};

struct partition_result {
    std::size_t partition_index = 0;
    std::vector<collision_record> collisions;
};

class bounded_partition_result_queue {
public:
    bounded_partition_result_queue(
        std::size_t capacity,
        std::size_t active_workers)
        : capacity_(capacity),
          active_workers_(active_workers)
    {
        if (capacity_ == 0 || active_workers_ == 0) {
            throw std::invalid_argument(
                "result queue requires positive capacity "
                "and worker count");
        }
    }

    bool push(partition_result&& result)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        not_full_.wait(
            lock,
            [this]
            {
                return cancelled_ ||
                    results_.size() < capacity_;
            });

        if (cancelled_)
            return false;

        results_.push_back(std::move(result));
        not_empty_.notify_one();
        return true;
    }

    bool pop(partition_result& result)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_.wait(
            lock,
            [this]
            {
                return cancelled_ ||
                    !results_.empty() ||
                    active_workers_ == 0;
            });

        if (cancelled_ || results_.empty())
            return false;

        result = std::move(results_.front());
        results_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void worker_finished()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (active_workers_ > 0)
            --active_workers_;

        not_empty_.notify_all();
    }

    void cancel()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    const std::size_t capacity_;
    std::size_t active_workers_;
    bool cancelled_ = false;

    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<partition_result> results_;
};

// Count all pairs satisfying:
//
//     min_a <= a <= max_a
//     min_b <= b <= max_b
//     a < b
//
static u64 count_all_pairs(
    u64 min_a,
    u64 min_b,
    u64 max_a,
    u64 max_b)
{
    if (min_a > max_a || min_b > max_b)
        return 0;

    if (max_b == 0)
        return 0;

    const u64 last_a =
        std::min(max_a, max_b - 1);

    if (min_a > last_a)
        return 0;

    u64 count = 0;

    for (u64 a = min_a; a <= last_a; ++a) {
        const u64 first_b =
            std::max(min_b, a + 1);

        if (first_b <= max_b)
            count += max_b - first_b + 1;
    }

    return count;
}


// Deterministically sample sums uniformly across the flattened
// (a,b) pair space.
//
// This is used only to estimate good partition boundaries.
// It does NOT affect completeness of the search.
//
static std::vector<u128> sample_sum_distribution(
    const std::vector<u128>& cubes,
    u64 min_a,
    u64 min_b,
    u64 max_a,
    u64 max_b,
    u64 total_pairs,
    u64 sample_count)
{
    std::vector<u128> samples;
    samples.reserve(sample_count);

    if (total_pairs == 0 || sample_count == 0)
        return samples;

    u64 a = min_a;
    u128 row_start = 0;

    for (u64 i = 0; i < sample_count; ++i) {

        // Midpoint of this sample's slice of the flattened
        // pair space.
        const u128 target =
            (u128(2) * i + 1) *
            total_pairs /
            (u128(2) * sample_count);

        while (a <= max_a) {

            const u64 first_b =
                std::max(min_b, a + 1);

            u64 row_count = 0;

            if (first_b <= max_b)
                row_count = max_b - first_b + 1;

            if (target < row_start + row_count) {

                const u64 offset =
                    static_cast<u64>(
                        target - row_start);

                const u64 b =
                    first_b + offset;

                samples.push_back(
                    cubes[a] + cubes[b]);

                break;
            }

            row_start += row_count;
            ++a;
        }
    }

    return samples;
}


// Examine one S partition and determine:
//
//   * exact pair count
//   * smallest/largest a that contributes
//   * smallest/largest b that contributes
//
// Valid pairs satisfy:
//
//     global_min_a <= a <= global_max_a
//     global_min_b <= b <= global_max_b
//     a < b
//
// and:
//
//     s_min <= a^3 + b^3 < s_max
//
// This runs in O(A + B), because both b pointers only move
// downward.
//
static partition_stats analyze_partition(
    const std::vector<u128>& cubes,
    u64 global_min_a,
    u64 global_min_b,
    u64 global_max_a,
    u64 global_max_b,
    u128 s_min,
    u128 s_max)
{
    partition_stats stats;

    if (global_min_a > global_max_a ||
        global_min_b > global_max_b ||
        s_min >= s_max ||
        global_max_b == 0)
    {
        return stats;
    }

    const u64 last_a =
        std::min(global_max_a, global_max_b - 1);

    if (global_min_a > last_a)
        return stats;

    /*
     * These represent the first b for which:
     *
     *     a^3 + b^3 >= s_min
     *
     * and:
     *
     *     a^3 + b^3 >= s_max
     *
     * respectively.
     *
     * max_b + 1 means "no b satisfies it".
     */
    u64 b_min_boundary = global_max_b + 1;
    u64 b_max_boundary = global_max_b + 1;

    bool found_any = false;

    for (u64 a = global_min_a; a <= last_a; ++a) {

        const u64 lowest_b =
            std::max(global_min_b, a + 1);

        if (lowest_b > global_max_b)
            break;

        /*
         * Find the first b such that:
         *
         *     a^3 + b^3 >= s_min
         */
        while (
            b_min_boundary > lowest_b &&
            cubes[a] +
                cubes[b_min_boundary - 1] >= s_min)
        {
            --b_min_boundary;
        }

        /*
         * Find the first b such that:
         *
         *     a^3 + b^3 >= s_max
         */
        while (
            b_max_boundary > lowest_b &&
            cubes[a] +
                cubes[b_max_boundary - 1] >= s_max)
        {
            --b_max_boundary;
        }

        /*
         * Valid b values are therefore:
         *
         *     first_b <= b < end_b
         */
        const u64 first_b =
            std::max(
                lowest_b,
                b_min_boundary);

        const u64 end_b =
            std::min(
                global_max_b + 1,
                b_max_boundary);

        if (first_b >= end_b)
            continue;

        const u64 count =
            end_b - first_b;

        stats.pair_count += count;

        if (!found_any) {
            stats.min_a = a;
            stats.max_a = a;

            stats.min_b = first_b;
            stats.max_b = end_b - 1;

            found_any = true;
        }
        else {
            stats.max_a = a;

            stats.min_b =
                std::min(
                    stats.min_b,
                    first_b);

            stats.max_b =
                std::max(
                    stats.max_b,
                    end_b - 1);
        }
    }

    return stats;
}

std::vector<search_partition> make_partitions(
    u64 min_a,
    u64 min_b,
    u64 max_a,
    u64 max_b,
    u64 target_pair_records)
{
    if (target_pair_records == 0)
        throw std::invalid_argument(
            "pair buffer must hold at least one record");

    if (min_a == 0 || min_b == 0)
        throw std::invalid_argument(
            "a and b must be positive");

    if (min_a > max_a || min_b > max_b)
        throw std::invalid_argument(
            "invalid a/b range");

    if (max_b <= min_a)
        return {};

    /*
     * Precompute cubes.
     *
     * unsigned __int128 is required once the roots get large;
     * e.g. 5,000,000^3 does not fit comfortably in the
     * overall two-cube sum inside uint64_t.
     */
    const u64 max_value =
        std::max(max_a, max_b);

    std::vector<u128> cubes(max_value + 1);

    for (u64 i = 0; i <= max_value; ++i)
        cubes[i] = cube(i);

    const u64 total_pairs =
        count_all_pairs(
            min_a,
            min_b,
            max_a,
            max_b);

    if (total_pairs == 0)
        return {};

    /*
     * Choose the initial number of S partitions from the
     * target pair-record capacity of the configured buffer.
     * Sampled S boundaries are approximate, so exact partition
     * counts are allowed to be above or below this target.
     */
    const u64 partition_count =
        total_pairs / target_pair_records +
        (total_pairs % target_pair_records != 0);

    if (partition_count >
        std::numeric_limits<std::size_t>::max() - 1)
    {
        throw std::length_error(
            "too many partitions for this platform");
    }

    const std::size_t p =
        static_cast<std::size_t>(partition_count);

    /*
     * Find smallest valid pair.
     */
    u64 first_a = min_a;

    while (
        first_a <= max_a &&
        std::max(min_b, first_a + 1) > max_b)
    {
        ++first_a;
    }

    if (first_a > max_a)
        return {};

    const u64 first_b =
        std::max(min_b, first_a + 1);

    /*
     * Largest valid pair.
     */
    const u64 last_a =
        std::min(max_a, max_b - 1);

    const u64 last_b =
        max_b;

    /*
     * Full half-open sum range:
     *
     *     global_s_min <= s < global_s_max
     */
    const u128 global_s_min =
        cubes[first_a] +
        cubes[first_b];

    const u128 global_s_max =
        cubes[last_a] +
        cubes[last_b] + 1;

    /*
     * Estimate S quantiles using a deterministic sample.
     *
     * Sampling only controls load balancing. It cannot make
     * the eventual search miss a pair.
     */
    u64 sample_count =
        std::min<u64>(partition_count, 20) *
        100000ULL;

    sample_count =
        std::max<u64>(
            sample_count,
            250000ULL);

    sample_count =
        std::min<u64>(
            sample_count,
            2000000ULL);

    sample_count =
        std::min<u64>(
            sample_count,
            total_pairs);

    auto samples =
        sample_sum_distribution(
            cubes,
            min_a,
            min_b,
            max_a,
            max_b,
            total_pairs,
            sample_count);

    std::sort(
        samples.begin(),
        samples.end());

    /*
     * Build P+1 S boundaries.
     */
    std::vector<u128> boundaries(p + 1);

    boundaries[0] = global_s_min;
    boundaries[p] = global_s_max;

    for (std::size_t i = 1; i < p; ++i) {

        const std::size_t sample_index =
            static_cast<std::size_t>(
                (u128(samples.size()) * i) / p);

        boundaries[i] =
            samples[
                std::min(
                    sample_index,
                    samples.size() - 1)];
    }

    /*
     * Analyze each partition exactly.
     *
     * This calculates:
     *
     *   - effective min/max a
     *   - effective min/max b
     *   - exact pair_count
     */
    std::vector<search_partition> partitions;
    partitions.reserve(p);

    for (std::size_t i = 0; i < p; ++i) {

        const u128 s_min =
            boundaries[i];

        const u128 s_max =
            boundaries[i + 1];

        const auto stats =
            analyze_partition(
                cubes,
                min_a,
                min_b,
                max_a,
                max_b,
                s_min,
                s_max);

        if (stats.pair_count == 0)
            continue;

        partitions.push_back({
            stats.min_a,
            stats.min_b,
            stats.max_a,
            stats.max_b,
            s_min,
            s_max,
            stats.pair_count
        });
    }

    return partitions;
}


int main(int argc, char* argv[])
{
    u64 min_a = 1;
    u64 min_b = 2;
    u64 max_a = 100000;
    u64 max_b = 100000;
    u64 buffer_size_bytes = u64(1) << 30;
    std::size_t requested_thread_count = 1;
    int requested_hip_device = 0;
    bool use_gpu = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];

        if (option == "-h" || option == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        const bool is_range_option =
            option == "-a" || option == "-b";

        const bool is_buffer_option =
            option == "-m" ||
            option == "--buffer-size";

        const bool is_thread_option =
            option == "-j" ||
            option == "--threads";

        const bool is_backend_option =
            option == "--backend";

        const bool is_hip_device_option =
            option == "-d" ||
            option == "--hip-device";

        if (!is_range_option &&
            !is_buffer_option &&
            !is_thread_option &&
            !is_backend_option &&
            !is_hip_device_option)
        {
            std::fprintf(
                stderr,
                "Unknown option: %s\n",
                argv[i]);
            print_usage(argv[0]);
            return 1;
        }

        if (i + 1 == argc) {
            std::fprintf(
                stderr,
                "Missing value after %s\n",
                argv[i]);
            print_usage(argv[0]);
            return 1;
        }

        const char* const value = argv[++i];

        if (is_backend_option) {
            const std::string_view backend = value;

            if (backend == "cpu") {
                use_gpu = false;
            }
            else if (backend == "gpu") {
                use_gpu = true;
            }
            else {
                std::fprintf(
                    stderr,
                    "Invalid backend: %s\n",
                    value);
                print_usage(argv[0]);
                return 1;
            }

            continue;
        }

        if (is_buffer_option) {
            if (!parse_buffer_size(
                    value,
                    buffer_size_bytes))
            {
                std::fprintf(
                    stderr,
                    "Invalid buffer size: %s\n",
                    value);
                print_usage(argv[0]);
                return 1;
            }

            continue;
        }

        if (is_thread_option) {
            if (!parse_thread_count(
                    value,
                    requested_thread_count))
            {
                std::fprintf(
                    stderr,
                    "Invalid thread count: %s\n",
                    value);
                print_usage(argv[0]);
                return 1;
            }

            continue;
        }

        if (is_hip_device_option) {
            if (!parse_hip_device(
                    value,
                    requested_hip_device))
            {
                std::fprintf(
                    stderr,
                    "Invalid HIP device index: %s\n",
                    value);
                print_usage(argv[0]);
                return 1;
            }

            continue;
        }

        u64& minimum = option == "-a" ? min_a : min_b;
        u64& maximum = option == "-a" ? max_a : max_b;

        if (!parse_range(value, minimum, maximum)) {
            std::fprintf(
                stderr,
                "Invalid range for %.*s: %s\n",
                static_cast<int>(option.size()),
                option.data(),
                value);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (min_a == 0 || min_b == 0) {
        std::fprintf(
            stderr,
            "The a and b ranges must contain positive integers.\n");
        return 1;
    }

    const u64 maximum_stored_value =
        std::numeric_limits<u32>::max();

    if (max_a > maximum_stored_value ||
        max_b > maximum_stored_value)
    {
        std::fprintf(
            stderr,
            "The a and b ranges must not exceed %llu "
            "(UINT32_MAX).\n",
            (unsigned long long)maximum_stored_value);
        return 1;
    }

    const char* backend_name = "CPU";
    std::string backend_directory = "CPU";
    std::string device_identifier;
    partition_search_function* search_backend_function =
        search_partition_collisions_cpu;
    std::size_t pair_record_size =
        cpu_pair_record_size();
    u64 pair_record_max_count =
        cpu_pair_record_max_count();

    if (use_gpu) {
#ifdef MAGIC_SQUARE_CUBES_ENABLE_HIP
        backend_name = "HIP GPU";
        search_backend_function =
            search_partition_collisions_gpu;
        pair_record_size = gpu_pair_record_size();
        pair_record_max_count =
            gpu_pair_record_max_count();

        try {
            set_hip_device(requested_hip_device);
            device_identifier =
                hip_device_identifier(
                    requested_hip_device);
            backend_directory =
                "GPU-" + device_identifier;
        }
        catch (const std::exception& error) {
            std::fprintf(
                stderr,
                "Unable to configure HIP device %d: %s\n",
                requested_hip_device,
                error.what());
            return 1;
        }
#else
        std::fprintf(
            stderr,
            "The GPU backend is unavailable in this build. "
            "Run 'make hip' and use the resulting "
            "magic-square-cubes-stage1-hip executable.\n");
        return 1;
#endif
    }

    const u64 target_pair_records =
        buffer_size_bytes / pair_record_size;

    if (target_pair_records == 0) {
        std::fprintf(
            stderr,
            "The buffer must be at least %zu bytes.\n",
            pair_record_size);
        return 1;
    }

    if (target_pair_records >
        pair_record_max_count)
    {
        std::fprintf(
            stderr,
            "The requested buffer exceeds the maximum "
            "supported vector size.\n");
        return 1;
    }

    std::printf(
        "Search backend: %s\n",
        backend_name);

    if (use_gpu) {
        std::printf(
            "HIP device: %d (%s)\n",
            requested_hip_device,
            device_identifier.c_str());
    }

    std::printf(
        "Search ranges: a[%llu, %llu], b[%llu, %llu]\n",
        (unsigned long long)min_a,
        (unsigned long long)max_a,
        (unsigned long long)min_b,
        (unsigned long long)max_b);

    std::printf(
        "Pair buffer target: %llu bytes (%llu records, %zu bytes each)\n",
        (unsigned long long)buffer_size_bytes,
        (unsigned long long)target_pair_records,
        pair_record_size);

    if (use_gpu) {
        std::printf(
            "GPU sorting: device-wide radix sort of "
            "unsigned 128-bit sums\n");
    }

    std::fflush(stdout);

    std::vector<search_partition> partitions;

    try {
        partitions =
            make_partitions(
                min_a,
                min_b,
                max_a,
                max_b,
                target_pair_records);
    }
    catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Unable to partition the search: %s\n",
            error.what());
        return 1;
    }

    if (partitions.empty()) {
        std::printf(
            "No pairs satisfy the specified ranges and a < b.\n");
        return 0;
    }

    for (std::size_t i = 0;
         i < partitions.size();
         ++i)
    {
        const auto& partition = partitions[i];

        printf("Partition %zu/%zu: a[%llu, %llu], b[%llu, %llu], s[%llu, %llu), pair_count=%llu\n",
               i + 1,
               partitions.size(),
               (unsigned long long)partition.min_a,
               (unsigned long long)partition.max_a,
               (unsigned long long)partition.min_b,
               (unsigned long long)partition.max_b,
               (unsigned long long)partition.s_min,
               (unsigned long long)partition.s_max,
               (unsigned long long)partition.pair_count);
    }

    std::printf(
        "Collision buckets: %zu files, %zu bytes "
        "(%zu records of %zu bytes) per in-memory bucket\n",
        collision_bucket_writer::bucket_count,
        collision_bucket_writer::bucket_size_bytes,
        collision_bucket_writer::records_per_bucket,
        sizeof(collision_record));

    const std::size_t backend_worker_count =
        use_gpu ? 1 : requested_thread_count;

    const std::size_t worker_count =
        std::min(
            backend_worker_count,
            partitions.size());

    if (use_gpu) {
        std::printf(
            "Worker threads: %zu requested, %zu active; "
            "the GPU backend uses one host worker\n",
            requested_thread_count,
            worker_count);
    }
    else {
        std::printf(
            "Worker threads: %zu requested, %zu active; "
            "pair buffer target applies to each active worker\n",
            requested_thread_count,
            worker_count);
    }

    try {
        collision_bucket_writer collision_buckets(
            backend_directory);
        u64 total_collision_count = 0;

        std::printf(
            "Collision output directory: %s\n",
            collision_buckets
                .output_directory()
                .string()
                .c_str());
        std::fflush(stdout);

        bounded_partition_result_queue results(
            worker_count,
            worker_count);

        std::atomic<std::size_t> next_partition{0};
        std::atomic<bool> stop_workers{false};

        std::mutex progress_mutex;
        std::mutex error_mutex;
        std::exception_ptr worker_error;

        const auto record_worker_error =
            [&](std::exception_ptr error)
            {
                {
                    std::lock_guard<std::mutex> lock(
                        error_mutex);

                    if (!worker_error)
                        worker_error = error;
                }

                stop_workers.store(
                    true,
                    std::memory_order_release);
                results.cancel();
            };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);

        std::size_t started_workers = 0;

        try {
            for (std::size_t worker_index = 0;
                 worker_index < worker_count;
                 ++worker_index)
            {
                workers.emplace_back(
                    [&, worker_index]
                    {
                        try {
                            while (!stop_workers.load(
                                std::memory_order_acquire))
                            {
                                const std::size_t partition_index =
                                    next_partition.fetch_add(
                                        1,
                                        std::memory_order_relaxed);

                                if (partition_index >=
                                    partitions.size())
                                {
                                    break;
                                }

                                const auto& partition =
                                    partitions[partition_index];

                                {
                                    std::lock_guard<std::mutex> lock(
                                        progress_mutex);

                                    std::printf(
                                        "Worker %zu/%zu processing "
                                        "partition %zu/%zu "
                                        "(pair_count=%llu)...\n",
                                        worker_index + 1,
                                        worker_count,
                                        partition_index + 1,
                                        partitions.size(),
                                        (unsigned long long)
                                            partition.pair_count);
                                    std::fflush(stdout);
                                }

                                const auto search_start =
                                    std::chrono::steady_clock::now();

                                auto collisions =
                                    search_backend_function(
                                        partition);

                                const auto search_end =
                                    std::chrono::steady_clock::now();

                                const double elapsed_seconds =
                                    std::chrono::duration<double>(
                                        search_end - search_start)
                                        .count();

                                const double operations_per_second =
                                    elapsed_seconds > 0.0
                                        ? static_cast<double>(
                                            partition.pair_count) /
                                            elapsed_seconds
                                        : 0.0;

                                const double million_operations_per_second =
                                    operations_per_second / 1'000'000.0;

                                const std::size_t collision_count =
                                    collisions.size();

                                {
                                    std::lock_guard<std::mutex> lock(
                                        progress_mutex);

                                    std::printf(
                                        "Worker %zu/%zu completed "
                                        "partition %zu/%zu: "
                                        "%zu collisions found, "
                                        "%.2f Mops/sec "
                                        "(%.3f s)\n",
                                        worker_index + 1,
                                        worker_count,
                                        partition_index + 1,
                                        partitions.size(),
                                        collision_count,
                                        million_operations_per_second,
                                        elapsed_seconds);
                                    std::fflush(stdout);
                                }

                                if (!results.push({
                                        partition_index,
                                        std::move(collisions)}))
                                {
                                    break;
                                }
                            }
                        }
                        catch (...) {
                            record_worker_error(
                                std::current_exception());
                        }

                        results.worker_finished();
                    });

                ++started_workers;
            }
        }
        catch (...) {
            stop_workers.store(
                true,
                std::memory_order_release);
            results.cancel();

            for (std::size_t i = started_workers;
                 i < worker_count;
                 ++i)
            {
                results.worker_finished();
            }

            for (auto& worker : workers) {
                if (worker.joinable())
                    worker.join();
            }

            throw;
        }

        std::exception_ptr writer_error;
        std::size_t processed_partitions = 0;

        try {
            while (true) {
                partition_result result;

                if (!results.pop(result))
                    break;

                for (const auto& collision :
                     result.collisions)
                {
                    collision_buckets.add(collision);
                }

                if (result.collisions.size() >
                    std::numeric_limits<u64>::max() -
                        total_collision_count)
                {
                    throw std::overflow_error(
                        "total collision count overflow");
                }

                total_collision_count +=
                    static_cast<u64>(
                        result.collisions.size());
                ++processed_partitions;

                {
                    std::lock_guard<std::mutex> lock(
                        progress_mutex);

                    std::printf(
                        "Bucketed partition %zu/%zu: "
                        "%zu collisions, %llu total\n",
                        result.partition_index + 1,
                        partitions.size(),
                        result.collisions.size(),
                        (unsigned long long)
                            total_collision_count);
                    std::fflush(stdout);
                }
            }
        }
        catch (...) {
            writer_error = std::current_exception();
            stop_workers.store(
                true,
                std::memory_order_release);
            results.cancel();
        }

        for (auto& worker : workers) {
            if (worker.joinable())
                worker.join();
        }

        if (writer_error)
            std::rethrow_exception(writer_error);

        {
            std::lock_guard<std::mutex> lock(error_mutex);

            if (worker_error)
                std::rethrow_exception(worker_error);
        }

        if (processed_partitions != partitions.size()) {
            throw std::runtime_error(
                "not all partitions produced a result");
        }

        collision_buckets.flush_all();

        std::printf(
            "Appended %llu collisions from all partitions "
            "to the bucket files.\n",
            (unsigned long long)total_collision_count);
    }
    catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Parallel collision search failed: %s\n",
            error.what());
        return 1;
    }

    return 0;
}
