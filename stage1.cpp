#include "search.h"
#include "search_cpu.h"
#ifdef MAGIC_SQUARE_CUBES_ENABLE_HIP
#include "search_gpu.h"
#endif
#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstring>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

static volatile std::sig_atomic_t stop_requested = 0;

extern "C" void request_stop_signal_handler(int)
{
    stop_requested = 1;
}

#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI

static void check_mpi(int result, const char* operation)
{
    if (result == MPI_SUCCESS)
        return;

    char message[MPI_MAX_ERROR_STRING]{};
    int message_length = 0;
    MPI_Error_string(
        result,
        message,
        &message_length);

    throw std::runtime_error(
        std::string(operation) + ": " +
        std::string(message, message_length));
}

class mpi_environment {
public:
    mpi_environment(int& argc, char**& argv)
    {
        int provided = MPI_THREAD_SINGLE;
        const int initialization_result =
            MPI_Init_thread(
                &argc,
                &argv,
                MPI_THREAD_FUNNELED,
                &provided);

        if (initialization_result != MPI_SUCCESS) {
            throw std::runtime_error(
                "MPI_Init_thread failed");
        }

        initialized_ = true;

        try {
            if (provided < MPI_THREAD_FUNNELED) {
                throw std::runtime_error(
                    "MPI does not provide MPI_THREAD_FUNNELED");
            }

            check_mpi(
                MPI_Comm_set_errhandler(
                    MPI_COMM_WORLD,
                    MPI_ERRORS_RETURN),
                "MPI_Comm_set_errhandler");
            check_mpi(
                MPI_Comm_rank(MPI_COMM_WORLD, &rank_),
                "MPI_Comm_rank");
            check_mpi(
                MPI_Comm_size(MPI_COMM_WORLD, &size_),
                "MPI_Comm_size");
        }
        catch (...) {
            (void)MPI_Finalize();
            initialized_ = false;
            throw;
        }
    }

    ~mpi_environment()
    {
        if (initialized_)
            (void)MPI_Finalize();
    }

    mpi_environment(const mpi_environment&) = delete;
    mpi_environment& operator=(const mpi_environment&) = delete;

    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }

    void abort(int error_code) const noexcept
    {
        if (initialized_)
            (void)MPI_Abort(MPI_COMM_WORLD, error_code);
    }

private:
    bool initialized_ = false;
    int rank_ = 0;
    int size_ = 1;
};

#endif

struct partition_stats {
    u64 min_a = 0;
    u64 min_b = 0;
    u64 max_a = 0;
    u64 max_b = 0;
    u64 pair_count = 0;
};

using partition_preparation_progress =
    std::function<void(unsigned int)>;

static search_partition materialize_partition(
    const search_partition& planned_partition,
    const partition_preparation_progress& progress = {});

class coarse_progress_reporter {
public:
    coarse_progress_reporter(
        const char* phase,
        u64 total)
        : phase_(phase), total_(total)
    {
        update_threshold();
    }

    void update(u64 completed)
    {
        if (total_ < 10 || next_percent_ > 100 ||
            completed < next_completed_)
        {
            return;
        }

        std::printf(
            "%s: %u%% (%llu/%llu)\n",
            phase_,
            next_percent_,
            (unsigned long long)completed,
            (unsigned long long)total_);
        std::fflush(stdout);

        next_percent_ += 10;
        update_threshold();
    }

private:
    void update_threshold()
    {
        if (next_percent_ > 100 || total_ == 0) {
            next_completed_ =
                std::numeric_limits<u64>::max();
            return;
        }

        next_completed_ = static_cast<u64>(
            (u128(total_) * next_percent_ + 99) / 100);
    }

    const char* phase_;
    u64 total_;
    unsigned int next_percent_ = 10;
    u64 next_completed_ = 0;
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

static bool valid_project_name(std::string_view name)
{
    if (name.empty() || name.size() > 255 ||
        name == "." || name == "..")
    {
        return false;
    }

    for (const char character : name) {
        const bool is_letter =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z');
        const bool is_digit =
            character >= '0' && character <= '9';

        if (!is_letter && !is_digit &&
            character != '-' && character != '_' &&
            character != '.')
        {
            return false;
        }
    }

    return true;
}

static void print_usage(const char* program)
{
    std::fprintf(
        stderr,
        "Usage: %s -p NAME [-a MIN:MAX] [-b MIN:MAX] [-m SIZE] [-j N] [-d N] "
        "[--backend cpu|gpu] [--gpu-method hash|radix]\n"
        "  -p, --project NAME      required project/output directory name\n"
        "  -m, --buffer-size SIZE  target pair buffer size; B, KB, MB, GB use powers of 1024\n"
        "  -j, --threads N         worker threads; pair buffer applies to each worker\n"
        "  -d, --hip-device N      HIP device index; first local device under MPI\n"
        "      --backend TYPE      search backend: cpu or gpu\n"
        "      --gpu-method TYPE   GPU collision method: hash or radix\n"
        "Defaults: -a 1:100000 -b 2:100000 -m 1GB -j 1 -d 0 --backend cpu --gpu-method hash\n",
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
    std::string_view project_name)
{
    return std::filesystem::path(std::string(project_name)) /
        "buckets";
}

class collision_bucket_writer {
public:
    static constexpr std::size_t bucket_count = 32;
    static constexpr std::size_t bucket_size_bytes =
        std::size_t(1) << 20;
    static constexpr std::size_t records_per_bucket =
        bucket_size_bytes / sizeof(collision_record);

    explicit collision_bucket_writer(
        std::string_view project_name)
        : output_directory_(
            collision_output_directory(
                project_name))
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

struct progress_fingerprint {
    u64 first;
    u64 second;
};

static progress_fingerprint make_progress_fingerprint(
    const std::vector<search_partition>& partitions,
    u64 min_a,
    u64 min_b,
    u64 max_a,
    u64 max_b,
    u64 target_pair_records)
{
    u64 first = 0x6a09e667f3bcc909ULL;
    u64 second = 0xbb67ae8584caa73bULL;

    const auto add = [&](u64 value) {
        first = mix_u64(first ^ mix_u64(value));
        second = mix_u64(
            second ^ mix_u64(
                value + 0x9e3779b97f4a7c15ULL));
    };

    add(min_a);
    add(min_b);
    add(max_a);
    add(max_b);
    add(target_pair_records);
    add(static_cast<u64>(partitions.size()));

    for (const auto& partition : partitions) {
        add(partition.min_a);
        add(partition.min_b);
        add(partition.max_a);
        add(partition.max_b);
        add(static_cast<u64>(partition.s_min));
        add(static_cast<u64>(partition.s_min >> 64));
        add(static_cast<u64>(partition.s_max));
        add(static_cast<u64>(partition.s_max >> 64));
        add(partition.pair_count);
    }

    return {first, second};
}

class partition_progress {
public:
    partition_progress(
        std::string_view project_name,
        const std::vector<search_partition>& partitions,
        progress_fingerprint fingerprint)
        : path_(
            std::filesystem::path(std::string(project_name)) /
            "stage1-progress-v1.bin"),
          distributed_(partitions.size(), false),
          completed_(partitions.size(), false)
    {
        static_assert(
            std::is_trivially_copyable_v<file_header>);
        static_assert(
            std::is_trivially_copyable_v<event_record>);

        std::filesystem::create_directories(
            path_.parent_path());

        const file_header expected_header =
            make_header(partitions.size(), fingerprint);

        if (std::filesystem::exists(path_)) {
            load(expected_header);
        }
        else {
            create(expected_header);
        }

        descriptor_ = ::open(
            path_.c_str(),
            O_WRONLY | O_APPEND);

        if (descriptor_ < 0) {
            throw std::system_error(
                errno,
                std::generic_category(),
                "unable to open progress file " +
                    path_.string());
        }
    }

    ~partition_progress()
    {
        if (descriptor_ >= 0)
            (void)::close(descriptor_);
    }

    partition_progress(const partition_progress&) = delete;
    partition_progress& operator=(
        const partition_progress&) = delete;

    std::vector<std::size_t> pending_indices() const
    {
        std::vector<std::size_t> pending;
        pending.reserve(
            completed_.size() - completed_count_);

        for (std::size_t i = 0;
             i < completed_.size();
             ++i)
        {
            if (!completed_[i])
                pending.push_back(i);
        }

        return pending;
    }

    std::size_t completed_count() const noexcept
    {
        return completed_count_;
    }

    std::size_t incomplete_distributed_count() const noexcept
    {
        std::size_t count = 0;

        for (std::size_t i = 0;
             i < distributed_.size();
             ++i)
        {
            if (distributed_[i] && !completed_[i])
                ++count;
        }

        return count;
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

    void record_distributed(std::size_t partition_index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        check_index(partition_index);

        if (completed_[partition_index] ||
            distributed_[partition_index])
        {
            return;
        }

        append_event({
            static_cast<u64>(partition_index),
            event_distributed,
            0
        });
        distributed_[partition_index] = true;
    }

    void record_completed(std::size_t partition_index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        check_index(partition_index);

        if (completed_[partition_index])
            return;

        append_event({
            static_cast<u64>(partition_index),
            event_completed,
            0
        });
        distributed_[partition_index] = true;
        completed_[partition_index] = true;
        ++completed_count_;
    }

private:
    static constexpr std::array<char, 8> file_magic{
        'M', 'S', 'C', 'S', '1', 'P', 'R', '1'
    };
    static constexpr u32 file_version = 1;
    static constexpr u32 event_distributed = 1;
    static constexpr u32 event_completed = 2;

    struct file_header {
        std::array<char, 8> magic;
        u32 version;
        u32 header_size;
        u64 partition_count;
        u64 fingerprint_first;
        u64 fingerprint_second;
    };

    struct event_record {
        u64 partition_index;
        u32 type;
        u32 reserved;
    };

    static file_header make_header(
        std::size_t partition_count,
        progress_fingerprint fingerprint)
    {
        if (partition_count >
            std::numeric_limits<u64>::max())
        {
            throw std::length_error(
                "too many partitions for the progress file");
        }

        return {
            file_magic,
            file_version,
            static_cast<u32>(sizeof(file_header)),
            static_cast<u64>(partition_count),
            fingerprint.first,
            fingerprint.second
        };
    }

    void create(const file_header& header)
    {
        std::ofstream file(
            path_,
            std::ios::binary |
            std::ios::trunc);

        file.write(
            reinterpret_cast<const char*>(&header),
            sizeof(header));
        file.flush();

        if (!file) {
            throw std::runtime_error(
                "unable to create progress file " +
                path_.string());
        }
    }

    void load(const file_header& expected_header)
    {
        std::ifstream file(path_, std::ios::binary);
        file_header header{};

        file.read(
            reinterpret_cast<char*>(&header),
            sizeof(header));

        if (!file ||
            header.magic != expected_header.magic ||
            header.version != expected_header.version ||
            header.header_size != expected_header.header_size ||
            header.partition_count !=
                expected_header.partition_count ||
            header.fingerprint_first !=
                expected_header.fingerprint_first ||
            header.fingerprint_second !=
                expected_header.fingerprint_second)
        {
            throw std::runtime_error(
                "progress file does not match this search: " +
                path_.string());
        }

        std::uintmax_t valid_size = sizeof(header);

        while (true) {
            event_record event{};
            file.read(
                reinterpret_cast<char*>(&event),
                sizeof(event));

            if (file.gcount() == 0)
                break;

            if (file.gcount() !=
                static_cast<std::streamsize>(sizeof(event)))
            {
                file.close();
                std::filesystem::resize_file(
                    path_,
                    valid_size);
                break;
            }

            valid_size += sizeof(event);

            if (event.reserved != 0 ||
                event.partition_index >= completed_.size())
            {
                throw std::runtime_error(
                    "invalid record in progress file " +
                    path_.string());
            }

            const std::size_t index =
                static_cast<std::size_t>(
                    event.partition_index);

            if (event.type == event_distributed) {
                distributed_[index] = true;
            }
            else if (event.type == event_completed) {
                distributed_[index] = true;

                if (!completed_[index]) {
                    completed_[index] = true;
                    ++completed_count_;
                }
            }
            else {
                throw std::runtime_error(
                    "unknown record in progress file " +
                    path_.string());
            }
        }
    }

    void check_index(std::size_t partition_index) const
    {
        if (partition_index >= completed_.size()) {
            throw std::out_of_range(
                "partition progress index is out of range");
        }
    }

    void append_event(const event_record& event)
    {
        const auto* data =
            reinterpret_cast<const unsigned char*>(&event);
        std::size_t remaining = sizeof(event);

        while (remaining != 0) {
            const ssize_t written = ::write(
                descriptor_,
                data,
                remaining);

            if (written < 0 && errno == EINTR)
                continue;

            if (written <= 0) {
                throw std::system_error(
                    errno,
                    std::generic_category(),
                    "unable to append progress file " +
                        path_.string());
            }

            data += written;
            remaining -= static_cast<std::size_t>(written);
        }

        if (::fsync(descriptor_) != 0) {
            throw std::system_error(
                errno,
                std::generic_category(),
                "unable to sync progress file " +
                    path_.string());
        }
    }

    std::filesystem::path path_;
    std::vector<bool> distributed_;
    std::vector<bool> completed_;
    std::size_t completed_count_ = 0;
    int descriptor_ = -1;
    std::mutex mutex_;
};

static std::size_t remove_nonprimitive_collisions(
    std::vector<collision_record>& collisions)
{
    const std::size_t original_size = collisions.size();

    collisions.erase(
        std::remove_if(
            collisions.begin(),
            collisions.end(),
            [](const collision_record& collision)
            {
                u32 divisor = std::gcd(
                    collision.a1,
                    collision.b1);
                divisor = std::gcd(
                    divisor,
                    collision.a2);
                divisor = std::gcd(
                    divisor,
                    collision.b2);
                return divisor > 1;
            }),
        collisions.end());

    return original_size - collisions.size();
}

struct partition_result {
    std::size_t partition_index = 0;
    std::vector<collision_record> collisions;
    std::size_t removed_collision_count = 0;
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

#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI

enum mpi_message_tag : int {
    mpi_partition_tag = 1,
    mpi_stop_tag = 2,
    mpi_result_header_tag = 3,
    mpi_result_records_tag = 4
};

struct mpi_partition_message {
    u64 partition_index;
    u64 partition_count;
    u64 min_a;
    u64 min_b;
    u64 max_a;
    u64 max_b;
    u64 s_min_low;
    u64 s_min_high;
    u64 s_max_low;
    u64 s_max_high;
    u64 pair_count;
};

struct mpi_result_header {
    u64 partition_index;
    u64 pair_count;
    u64 collision_count;
    u64 removed_collision_count;
    double elapsed_seconds;
};

static_assert(
    std::is_trivially_copyable_v<mpi_partition_message>);
static_assert(
    std::is_trivially_copyable_v<mpi_result_header>);
static_assert(
    std::is_trivially_copyable_v<collision_record>);

template <typename T>
static void mpi_send_object(
    const T& value,
    int destination,
    int tag)
{
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) <= INT_MAX);

    check_mpi(
        MPI_Send(
            &value,
            static_cast<int>(sizeof(T)),
            MPI_BYTE,
            destination,
            tag,
            MPI_COMM_WORLD),
        "MPI_Send");
}

template <typename T>
static void mpi_receive_object(
    T& value,
    int source,
    int tag,
    MPI_Status* status = MPI_STATUS_IGNORE)
{
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) <= INT_MAX);

    check_mpi(
        MPI_Recv(
            &value,
            static_cast<int>(sizeof(T)),
            MPI_BYTE,
            source,
            tag,
            MPI_COMM_WORLD,
            status),
        "MPI_Recv");
}

static void mpi_send_bytes(
    const void* data,
    std::size_t byte_count,
    int destination,
    int tag)
{
    const auto* bytes =
        static_cast<const unsigned char*>(data);

    while (byte_count != 0) {
        const int chunk_size = static_cast<int>(
            std::min<std::size_t>(
                byte_count,
                static_cast<std::size_t>(INT_MAX)));

        check_mpi(
            MPI_Send(
                bytes,
                chunk_size,
                MPI_BYTE,
                destination,
                tag,
                MPI_COMM_WORLD),
            "MPI_Send collision records");

        bytes += chunk_size;
        byte_count -= static_cast<std::size_t>(chunk_size);
    }
}

static void mpi_receive_bytes(
    void* data,
    std::size_t byte_count,
    int source,
    int tag)
{
    auto* bytes = static_cast<unsigned char*>(data);

    while (byte_count != 0) {
        const int chunk_size = static_cast<int>(
            std::min<std::size_t>(
                byte_count,
                static_cast<std::size_t>(INT_MAX)));

        check_mpi(
            MPI_Recv(
                bytes,
                chunk_size,
                MPI_BYTE,
                source,
                tag,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE),
            "MPI_Recv collision records");

        bytes += chunk_size;
        byte_count -= static_cast<std::size_t>(chunk_size);
    }
}

static mpi_partition_message make_mpi_partition_message(
    const search_partition& partition,
    std::size_t partition_index,
    std::size_t partition_count)
{
    return {
        static_cast<u64>(partition_index),
        static_cast<u64>(partition_count),
        partition.min_a,
        partition.min_b,
        partition.max_a,
        partition.max_b,
        static_cast<u64>(partition.s_min),
        static_cast<u64>(partition.s_min >> 64),
        static_cast<u64>(partition.s_max),
        static_cast<u64>(partition.s_max >> 64),
        partition.pair_count
    };
}

static search_partition search_partition_from_mpi(
    const mpi_partition_message& message)
{
    return {
        message.min_a,
        message.min_b,
        message.max_a,
        message.max_b,
        (u128(message.s_min_high) << 64) |
            message.s_min_low,
        (u128(message.s_max_high) << 64) |
            message.s_max_low,
        message.pair_count
    };
}

static void mpi_send_partition(
    const std::vector<search_partition>& partitions,
    std::size_t partition_index,
    int worker_rank)
{
    const auto message = make_mpi_partition_message(
        partitions[partition_index],
        partition_index,
        partitions.size());

    mpi_send_object(
        message,
        worker_rank,
        mpi_partition_tag);

    std::printf(
        "Assigned partition %zu/%zu to MPI worker rank %d.\n",
        partition_index + 1,
        partitions.size(),
        worker_rank);
    std::fflush(stdout);
}

static void mpi_send_stop(int worker_rank)
{
    const unsigned char stop = 0;
    mpi_send_object(stop, worker_rank, mpi_stop_tag);
}

static void run_mpi_coordinator(
    const std::vector<search_partition>& partitions,
    const std::vector<std::size_t>& pending_partitions,
    partition_progress& progress,
    int world_size,
    std::string_view project_name)
{
    if (pending_partitions.empty()) {
        for (int worker_rank = 1;
             worker_rank < world_size;
             ++worker_rank)
        {
            mpi_send_stop(worker_rank);
        }
        return;
    }

    collision_bucket_writer collision_buckets(
        project_name);

    std::printf(
        "Collision output directory: %s\n",
        collision_buckets.output_directory()
            .string().c_str());
    std::fflush(stdout);

    std::size_t next_pending_partition = 0;
    std::size_t active_workers = 0;
    std::vector<bool> assigned(
        partitions.size(), false);

    const auto assign_partition =
        [&](int worker_rank)
        {
            const std::size_t partition_index =
                pending_partitions[
                    next_pending_partition++];

            progress.record_distributed(
                partition_index);
            assigned[partition_index] = true;
            mpi_send_partition(
                partitions,
                partition_index,
                worker_rank);
        };

    for (int worker_rank = 1;
         worker_rank < world_size;
         ++worker_rank)
    {
        if (!stop_requested &&
            next_pending_partition <
                pending_partitions.size())
        {
            assign_partition(worker_rank);
            ++active_workers;
        }
        else {
            mpi_send_stop(worker_rank);
        }
    }

    std::vector<bool> completed_this_run(
        partitions.size(), false);
    std::size_t completed_this_run_count = 0;
    u64 total_collision_count = 0;
    u64 total_removed_collision_count = 0;

    while (active_workers != 0) {
        mpi_result_header header{};
        MPI_Status status{};

        mpi_receive_object(
            header,
            MPI_ANY_SOURCE,
            mpi_result_header_tag,
            &status);

        if (header.partition_index >= partitions.size()) {
            throw std::runtime_error(
                "MPI worker returned an invalid partition index");
        }

        const std::size_t partition_index =
            static_cast<std::size_t>(
                header.partition_index);

        if (!assigned[partition_index] ||
            completed_this_run[partition_index])
        {
            throw std::runtime_error(
                "MPI worker returned an unassigned partition "
                "or returned a partition more than once");
        }

        if (header.collision_count >
            std::vector<collision_record>().max_size())
        {
            throw std::length_error(
                "MPI collision result is too large");
        }

        std::vector<collision_record> collisions(
            static_cast<std::size_t>(
                header.collision_count));

        if (collisions.size() >
            std::numeric_limits<std::size_t>::max() /
                sizeof(collision_record))
        {
            throw std::length_error(
                "MPI collision byte count overflow");
        }

        mpi_receive_bytes(
            collisions.data(),
            collisions.size() * sizeof(collision_record),
            status.MPI_SOURCE,
            mpi_result_records_tag);

        for (const auto& collision : collisions)
            collision_buckets.add(collision);

        // A partition is checkpointed only after every bucket write
        // for that partition has been flushed.
        collision_buckets.flush_all();
        progress.record_completed(partition_index);

        if (header.collision_count >
            std::numeric_limits<u64>::max() -
                total_collision_count)
        {
            throw std::overflow_error(
                "total collision count overflow");
        }

        total_collision_count += header.collision_count;

        if (header.removed_collision_count >
            std::numeric_limits<u64>::max() -
                total_removed_collision_count)
        {
            throw std::overflow_error(
                "total removed collision count overflow");
        }

        total_removed_collision_count +=
            header.removed_collision_count;
        completed_this_run[partition_index] = true;
        ++completed_this_run_count;

        const double operations_per_second =
            header.elapsed_seconds > 0.0
                ? static_cast<double>(
                    header.pair_count) /
                    header.elapsed_seconds
                : 0.0;

        std::printf(
            "MPI worker rank %d completed partition %zu/%zu: "
            "%llu primitive collisions retained, %llu "
            "non-primitive collisions removed, %.2f Mops/sec "
            "(%.3f s); "
            "%zu/%zu total partitions complete, "
            "%llu collisions this run\n",
            status.MPI_SOURCE,
            partition_index + 1,
            partitions.size(),
            (unsigned long long)header.collision_count,
            (unsigned long long)header.removed_collision_count,
            operations_per_second / 1'000'000.0,
            header.elapsed_seconds,
            progress.completed_count(),
            partitions.size(),
            (unsigned long long)total_collision_count);
        std::fflush(stdout);

        if (!stop_requested &&
            next_pending_partition <
                pending_partitions.size())
        {
            assign_partition(status.MPI_SOURCE);
        }
        else {
            mpi_send_stop(status.MPI_SOURCE);
            --active_workers;
        }
    }

    collision_buckets.flush_all();

    if (stop_requested &&
        progress.completed_count() < partitions.size())
    {
        std::printf(
            "Stop requested. Checkpointed %zu partitions this "
            "run; %zu/%zu total partitions are complete.\n",
            completed_this_run_count,
            progress.completed_count(),
            partitions.size());
    }
    else {
        std::printf(
            "Appended %llu collisions this run; all %zu "
            "partitions are complete (%llu non-primitive "
            "collisions removed).\n",
            (unsigned long long)total_collision_count,
            partitions.size(),
            (unsigned long long)total_removed_collision_count);
    }
    std::fflush(stdout);
}

static void run_mpi_worker(
    partition_search_function* search_backend_function,
    int world_rank)
{
    while (true) {
        MPI_Status status{};
        check_mpi(
            MPI_Probe(
                0,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                &status),
            "MPI_Probe");

        if (status.MPI_TAG == mpi_stop_tag) {
            unsigned char stop = 0;
            mpi_receive_object(
                stop,
                0,
                mpi_stop_tag);
            break;
        }

        if (status.MPI_TAG != mpi_partition_tag) {
            throw std::runtime_error(
                "MPI worker received an unexpected message");
        }

        mpi_partition_message message{};
        mpi_receive_object(
            message,
            0,
            mpi_partition_tag);

        const search_partition planned_partition =
            search_partition_from_mpi(message);

        std::printf(
            "MPI worker rank %d preparing partition %llu/%llu...\n",
            world_rank,
            (unsigned long long)message.partition_index + 1,
            (unsigned long long)message.partition_count);
        std::fflush(stdout);

        const auto preparation_start =
            std::chrono::steady_clock::now();

        const search_partition partition =
            materialize_partition(
                planned_partition,
                [&](unsigned int percent)
                {
                    if (std::chrono::steady_clock::now() -
                            preparation_start <
                        std::chrono::seconds(1))
                    {
                        return;
                    }

                    std::printf(
                        "MPI worker rank %d partition %llu "
                        "preparation: %u%%\n",
                        world_rank,
                        (unsigned long long)
                            message.partition_index + 1,
                        percent);
                    std::fflush(stdout);
                });

        std::printf(
            "MPI worker rank %d processing partition %llu/%llu "
            "(pair_count=%llu)...\n",
            world_rank,
            (unsigned long long)message.partition_index + 1,
            (unsigned long long)message.partition_count,
            (unsigned long long)partition.pair_count);
        std::fflush(stdout);

        const auto search_start =
            std::chrono::steady_clock::now();

        auto collisions =
            search_backend_function(partition);

        const auto search_end =
            std::chrono::steady_clock::now();

        const double elapsed_seconds =
            std::chrono::duration<double>(
                search_end - search_start).count();

        const std::size_t removed_collision_count =
            remove_nonprimitive_collisions(collisions);

        const mpi_result_header header{
            message.partition_index,
            partition.pair_count,
            static_cast<u64>(collisions.size()),
            static_cast<u64>(removed_collision_count),
            elapsed_seconds
        };

        mpi_send_object(
            header,
            0,
            mpi_result_header_tag);

        mpi_send_bytes(
            collisions.data(),
            collisions.size() * sizeof(collision_record),
            0,
            mpi_result_records_tag);
    }
}

static int mpi_worker_device_index(
    int world_rank,
    int first_device)
{
    MPI_Comm local_communicator = MPI_COMM_NULL;
    check_mpi(
        MPI_Comm_split_type(
            MPI_COMM_WORLD,
            MPI_COMM_TYPE_SHARED,
            world_rank,
            MPI_INFO_NULL,
            &local_communicator),
        "MPI_Comm_split_type");

    int local_rank = 0;
    check_mpi(
        MPI_Comm_rank(
            local_communicator,
            &local_rank),
        "MPI_Comm_rank for shared-memory communicator");

    const int coordinator_candidate =
        world_rank == 0 ? local_rank : -1;
    int coordinator_local_rank = -1;

    check_mpi(
        MPI_Allreduce(
            &coordinator_candidate,
            &coordinator_local_rank,
            1,
            MPI_INT,
            MPI_MAX,
            local_communicator),
        "MPI_Allreduce coordinator local rank");

    int worker_local_index = local_rank;
    if (coordinator_local_rank >= 0 &&
        local_rank > coordinator_local_rank)
    {
        --worker_local_index;
    }

    check_mpi(
        MPI_Comm_free(&local_communicator),
        "MPI_Comm_free");

    if (first_device >
        std::numeric_limits<int>::max() -
            worker_local_index)
    {
        throw std::overflow_error(
            "MPI worker HIP device index overflow");
    }

    return first_device + worker_local_index;
}

#endif

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

    u128 count = 0;

    if (min_a < min_b) {
        const u64 constant_end =
            std::min(last_a, min_b - 1);

        if (min_a <= constant_end) {
            count +=
                u128(constant_end - min_a + 1) *
                (max_b - min_b + 1);
        }
    }

    const u64 sloped_start =
        std::max(min_a, min_b);

    if (sloped_start <= last_a) {
        const u128 row_count =
            last_a - sloped_start + 1;
        const u128 a_sum =
            (u128(sloped_start) + last_a) *
            row_count / 2;

        count +=
            row_count * max_b -
            a_sum;
    }

    if (count > std::numeric_limits<u64>::max()) {
        throw std::overflow_error(
            "pair count exceeds uint64_t");
    }

    return static_cast<u64>(count);
}


// Deterministically sample sums uniformly across the flattened
// (a,b) pair space.
//
// This is used only to estimate good partition boundaries.
// It does NOT affect completeness of the search.
//
static std::vector<u128> sample_sum_distribution(
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

    coarse_progress_reporter progress(
        "Sampling sum distribution",
        sample_count);

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
                    cube(a) + cube(b));

                break;
            }

            row_start += row_count;
            ++a;
        }

        progress.update(i + 1);
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
    u64 global_min_a,
    u64 global_min_b,
    u64 global_max_a,
    u64 global_max_b,
    u128 s_min,
    u128 s_max,
    const partition_preparation_progress& progress)
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
    const u64 a_count =
        last_a - global_min_a + 1;
    unsigned int next_progress_percent =
        a_count < 10 ? 100 : 10;
    u64 next_progress_a = static_cast<u64>(
        (u128(a_count) * next_progress_percent + 99) /
        100);

    for (u64 a = global_min_a; a <= last_a; ++a) {

        const u128 a3 = cube(a);

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
            a3 + cube(b_min_boundary - 1) >= s_min)
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
            a3 + cube(b_max_boundary - 1) >= s_max)
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

        const u64 completed_a =
            a - global_min_a + 1;

        if (progress &&
            next_progress_percent <= 100 &&
            completed_a >= next_progress_a)
        {
            progress(next_progress_percent);
            next_progress_percent += 10;

            if (next_progress_percent <= 100) {
                next_progress_a = static_cast<u64>(
                    (u128(a_count) *
                         next_progress_percent +
                     99) /
                    100);
            }
        }

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

static search_partition materialize_partition(
    const search_partition& planned_partition,
    const partition_preparation_progress& progress)
{
    const auto stats = analyze_partition(
        planned_partition.min_a,
        planned_partition.min_b,
        planned_partition.max_a,
        planned_partition.max_b,
        planned_partition.s_min,
        planned_partition.s_max,
        progress);

    if (stats.pair_count == 0) {
        search_partition empty = planned_partition;
        empty.pair_count = 0;
        return empty;
    }

    return {
        stats.min_a,
        stats.min_b,
        stats.max_a,
        stats.max_b,
        planned_partition.s_min,
        planned_partition.s_max,
        stats.pair_count
    };
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
        cube(first_a) +
        cube(first_b);

    const u128 global_s_max =
        cube(last_a) +
        cube(last_b) + 1;

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

    std::printf(
        "Partition planning: %llu pairs, %llu target "
        "partitions, %llu distribution samples.\n",
        (unsigned long long)total_pairs,
        (unsigned long long)partition_count,
        (unsigned long long)sample_count);
    std::fflush(stdout);

    auto samples =
        sample_sum_distribution(
            min_a,
            min_b,
            max_a,
            max_b,
            total_pairs,
            sample_count);

    std::printf(
        "Sorting %zu distribution samples...\n",
        samples.size());
    std::fflush(stdout);

    std::sort(
        samples.begin(),
        samples.end());

    std::printf("Distribution sample sort complete.\n");
    std::fflush(stdout);

    /*
     * Build P+1 S boundaries.
     */
    std::vector<u128> boundaries(p + 1);

    boundaries[0] = global_s_min;
    boundaries[p] = global_s_max;

    coarse_progress_reporter boundary_progress(
        "Selecting sum boundaries",
        partition_count > 1
            ? partition_count - 1
            : 0);

    for (std::size_t i = 1; i < p; ++i) {

        const std::size_t sample_index =
            static_cast<std::size_t>(
                (u128(samples.size()) * i) / p);

        boundaries[i] =
            samples[
                std::min(
                    sample_index,
                    samples.size() - 1)];

        boundary_progress.update(
            static_cast<u64>(i));
    }

    // Exact bounds and pair counts are intentionally deferred to
    // workers. Analyzing every partition here costs O(P * (A+B))
    // and prevents any CPU or GPU work from starting.
    std::vector<search_partition> partitions;
    partitions.reserve(p);

    coarse_progress_reporter plan_progress(
        "Building partition plan",
        partition_count);

    for (std::size_t i = 0; i < p; ++i) {

        const u128 s_min =
            boundaries[i];

        const u128 s_max =
            boundaries[i + 1];

        plan_progress.update(
            static_cast<u64>(i + 1));

        if (s_min >= s_max)
            continue;

        partitions.push_back({
            min_a,
            min_b,
            max_a,
            max_b,
            s_min,
            s_max,
            0
        });
    }

    return partitions;
}


static int run_stage1(
    int argc,
    char* argv[],
    int mpi_rank,
    int mpi_size)
{
    const bool distributed = mpi_size > 1;
    const bool coordinator = mpi_rank == 0;

    u64 min_a = 1;
    u64 min_b = 2;
    u64 max_a = 100000;
    u64 max_b = 100000;
    u64 buffer_size_bytes = u64(1) << 30;
    std::size_t requested_thread_count = 1;
    int requested_hip_device = 0;
    bool use_gpu = false;
    bool use_gpu_hash_table = true;
    std::string project_name;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];

        if (option == "-h" || option == "--help") {
            if (coordinator)
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

        const bool is_gpu_method_option =
            option == "--gpu-method";

        const bool is_project_option =
            option == "-p" ||
            option == "--project";

        if (!is_range_option &&
            !is_buffer_option &&
            !is_thread_option &&
            !is_backend_option &&
            !is_hip_device_option &&
            !is_gpu_method_option &&
            !is_project_option)
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

        if (is_project_option) {
            if (!valid_project_name(value)) {
                std::fprintf(
                    stderr,
                    "Invalid project name: %s\n"
                    "Project names may contain only letters, "
                    "digits, '.', '-', and '_', and must be at "
                    "most 255 characters.\n",
                    value);
                print_usage(argv[0]);
                return 1;
            }

            project_name = value;
            continue;
        }

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

        if (is_gpu_method_option) {
            const std::string_view method = value;

            if (method == "hash") {
                use_gpu_hash_table = true;
            }
            else if (method == "radix") {
                use_gpu_hash_table = false;
            }
            else {
                std::fprintf(
                    stderr,
                    "Invalid GPU method: %s\n",
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

    if (project_name.empty()) {
        if (coordinator) {
            std::fprintf(
                stderr,
                "A project name is required.\n");
            print_usage(argv[0]);
        }
        return 1;
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
    std::string device_identifier;
    partition_search_function* search_backend_function =
        search_partition_collisions_cpu;
    std::size_t pair_record_size =
        cpu_pair_record_size();
    u64 pair_record_max_count =
        cpu_pair_record_max_count();
    int effective_hip_device = requested_hip_device;

#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
    if (use_gpu && distributed) {
        effective_hip_device =
            mpi_worker_device_index(
                mpi_rank,
                requested_hip_device);
    }
#endif

    if (use_gpu) {
#ifdef MAGIC_SQUARE_CUBES_ENABLE_HIP
        backend_name = "HIP GPU";
        search_backend_function =
            use_gpu_hash_table
                ? search_partition_collisions_gpu_hash
                : search_partition_collisions_gpu_radix;
        pair_record_size =
            use_gpu_hash_table
                ? gpu_hash_pair_record_size()
                : gpu_pair_record_size();
        pair_record_max_count =
            use_gpu_hash_table
                ? gpu_hash_pair_record_max_count()
                : gpu_pair_record_max_count();

        if (!distributed || !coordinator) {
            try {
                set_hip_device(effective_hip_device);
                device_identifier =
                    hip_device_identifier(
                        effective_hip_device);
            }
            catch (const std::exception& error) {
                std::fprintf(
                    stderr,
                    "MPI rank %d unable to configure HIP "
                    "device %d: %s\n",
                    mpi_rank,
                    effective_hip_device,
                    error.what());
#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
                if (distributed)
                    (void)MPI_Abort(MPI_COMM_WORLD, 1);
#endif
                return 1;
            }
        }
#else
        if (coordinator) {
            std::fprintf(
                stderr,
                "The GPU backend is unavailable in this build. "
                "Run 'make hip' and use the resulting "
                "magic-square-cubes-stage1-hip executable.\n");
        }
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

    if (coordinator) {
        try {
            std::filesystem::create_directories(
                std::filesystem::path(project_name));
        }
        catch (const std::exception& error) {
            std::fprintf(
                stderr,
                "Unable to create project directory '%s': %s\n",
                project_name.c_str(),
                error.what());
#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
            if (distributed)
                (void)MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }
    }

    (void)std::signal(
        SIGINT,
        request_stop_signal_handler);
    (void)std::signal(
        SIGTERM,
        request_stop_signal_handler);

    if (coordinator) {
        std::printf(
            "Project: %s\n",
            project_name.c_str());
        std::printf(
            "Search backend: %s%s\n",
            backend_name,
            distributed ? " with MPI" : "");
    }

    if (use_gpu && !distributed) {
        std::printf(
            "HIP device: %d (%s)\n",
            effective_hip_device,
            device_identifier.c_str());
    }
    else if (use_gpu && !coordinator) {
        std::printf(
            "MPI worker rank %d using HIP device %d (%s)\n",
            mpi_rank,
            effective_hip_device,
            device_identifier.c_str());
    }

    if (coordinator) {
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
            if (use_gpu_hash_table) {
                std::printf(
                    "GPU collision method: open-addressed "
                    "hash table of unsigned 128-bit sums\n");
            }
            else {
                std::printf(
                    "GPU collision method: device-wide radix "
                    "sort of unsigned 128-bit sums\n");
            }
        }

        if (distributed) {
            std::printf(
                "MPI ranks: %d total, 1 coordinator, %d workers\n",
                mpi_size,
                mpi_size - 1);
        }
    }

    std::fflush(stdout);

    std::vector<search_partition> partitions;

    if (coordinator) {
        std::printf("Planning sum partitions...\n");
        std::fflush(stdout);

        const auto planning_start =
            std::chrono::steady_clock::now();

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
#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
            if (distributed)
                (void)MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        const double planning_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                planning_start).count();

        std::printf(
            "Planned %zu sum partitions in %.3f s.\n",
            partitions.size(),
            planning_seconds);
        std::fflush(stdout);
    }

    if (coordinator && partitions.empty()) {
        std::printf(
            "No pairs satisfy the specified ranges and a < b.\n");
    }

    if (coordinator) {
        const auto print_partition =
            [&](std::size_t i)
            {
                const auto& partition = partitions[i];

                std::printf(
                    "Partition %zu/%zu: a[%llu, %llu], "
                    "b[%llu, %llu], s[%llu, %llu)\n",
                    i + 1,
                    partitions.size(),
                    (unsigned long long)partition.min_a,
                    (unsigned long long)partition.max_a,
                    (unsigned long long)partition.min_b,
                    (unsigned long long)partition.max_b,
                    (unsigned long long)partition.s_min,
                    (unsigned long long)partition.s_max);
            };

        constexpr std::size_t preview_count = 10;

        if (partitions.size() <= preview_count) {
            for (std::size_t i = 0;
                 i < partitions.size();
                 ++i)
            {
                print_partition(i);
            }
        }
        else {
            const std::size_t edge_count =
                preview_count / 2;

            for (std::size_t i = 0;
                 i < edge_count;
                 ++i)
            {
                print_partition(i);
            }

            std::printf(
                "... %zu partitions omitted from startup "
                "preview ...\n",
                partitions.size() - 2 * edge_count);

            for (std::size_t i =
                     partitions.size() - edge_count;
                 i < partitions.size();
                 ++i)
            {
                print_partition(i);
            }
        }

        if (!partitions.empty()) {
            std::printf(
                "Exact partition bounds and pair counts are "
                "calculated by workers on assignment.\n");
        }

        if (!partitions.empty()) {
            std::printf(
                "Collision buckets: %zu files, %zu bytes "
                "(%zu records of %zu bytes) per in-memory bucket\n",
                collision_bucket_writer::bucket_count,
                collision_bucket_writer::bucket_size_bytes,
                collision_bucket_writer::records_per_bucket,
                sizeof(collision_record));
        }
    }

    std::unique_ptr<partition_progress> progress;
    std::vector<std::size_t> pending_partitions;

    if (coordinator) {
        try {
            progress = std::make_unique<partition_progress>(
                project_name,
                partitions,
                make_progress_fingerprint(
                    partitions,
                    min_a,
                    min_b,
                    max_a,
                    max_b,
                    target_pair_records));
            pending_partitions =
                progress->pending_indices();

            std::printf(
                "Progress file: %s\n"
                "Resume status: %zu/%zu partitions complete, "
                "%zu pending",
                progress->path().string().c_str(),
                progress->completed_count(),
                partitions.size(),
                pending_partitions.size());

            const std::size_t previously_distributed =
                progress->incomplete_distributed_count();

            if (previously_distributed != 0) {
                std::printf(
                    " (%zu previously distributed but incomplete)",
                    previously_distributed);
            }

            std::printf(".\n");
            std::fflush(stdout);
        }
        catch (const std::exception& error) {
            std::fprintf(
                stderr,
                "Unable to initialize partition progress: %s\n",
                error.what());
#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
            if (distributed)
                (void)MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }
    }

#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
    if (distributed) {
        try {
            if (coordinator) {
                run_mpi_coordinator(
                    partitions,
                    pending_partitions,
                    *progress,
                    mpi_size,
                    project_name);
            }
            else {
                run_mpi_worker(
                    search_backend_function,
                    mpi_rank);
            }

            check_mpi(
                MPI_Barrier(MPI_COMM_WORLD),
                "MPI_Barrier");
        }
        catch (const std::exception& error) {
            std::fprintf(
                stderr,
                "MPI rank %d failed: %s\n",
                mpi_rank,
                error.what());
            (void)MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        return 0;
    }
#endif

    if (pending_partitions.empty()) {
        if (!partitions.empty()) {
            std::printf(
                "All partitions were already completed; "
                "nothing to do.\n");
        }
        return 0;
    }

    const std::size_t backend_worker_count =
        use_gpu ? 1 : requested_thread_count;

    const std::size_t worker_count =
        std::min(
            backend_worker_count,
            pending_partitions.size());

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
            project_name);
        u64 total_collision_count = 0;
        u64 total_removed_collision_count = 0;

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

        std::atomic<std::size_t> next_pending_partition{0};
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
                                       std::memory_order_acquire) &&
                                   !stop_requested)
                            {
                                const std::size_t pending_index =
                                    next_pending_partition.fetch_add(
                                        1,
                                        std::memory_order_relaxed);

                                if (pending_index >=
                                    pending_partitions.size())
                                {
                                    break;
                                }

                                const std::size_t partition_index =
                                    pending_partitions[pending_index];

                                progress->record_distributed(
                                    partition_index);

                                const auto& planned_partition =
                                    partitions[partition_index];

                                {
                                    std::lock_guard<std::mutex> lock(
                                        progress_mutex);

                                    std::printf(
                                        "Worker %zu/%zu preparing "
                                        "partition %zu/%zu...\n",
                                        worker_index + 1,
                                        worker_count,
                                        partition_index + 1,
                                        partitions.size());
                                    std::fflush(stdout);
                                }

                                const auto preparation_start =
                                    std::chrono::steady_clock::now();

                                const search_partition partition =
                                    materialize_partition(
                                        planned_partition,
                                        [&](unsigned int percent)
                                        {
                                            if (std::chrono::steady_clock::now() -
                                                    preparation_start <
                                                std::chrono::seconds(1))
                                            {
                                                return;
                                            }

                                            std::lock_guard<std::mutex> lock(
                                                progress_mutex);

                                            std::printf(
                                                "Worker %zu/%zu partition "
                                                "%zu preparation: %u%%\n",
                                                worker_index + 1,
                                                worker_count,
                                                partition_index + 1,
                                                percent);
                                            std::fflush(stdout);
                                        });

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

                                const std::size_t removed_collision_count =
                                    remove_nonprimitive_collisions(
                                        collisions);

                                const std::size_t collision_count =
                                    collisions.size();

                                {
                                    std::lock_guard<std::mutex> lock(
                                        progress_mutex);

                                    std::printf(
                                        "Worker %zu/%zu completed "
                                        "partition %zu/%zu: "
                                        "%zu primitive collisions retained, "
                                        "%zu non-primitive collisions removed, "
                                        "%.2f Mops/sec "
                                        "(%.3f s)\n",
                                        worker_index + 1,
                                        worker_count,
                                        partition_index + 1,
                                        partitions.size(),
                                        collision_count,
                                        removed_collision_count,
                                        million_operations_per_second,
                                        elapsed_seconds);
                                    std::fflush(stdout);
                                }

                                if (!results.push({
                                        partition_index,
                                        std::move(collisions),
                                        removed_collision_count}))
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

                // Do not mark the partition complete until its bucket
                // records have been flushed.
                collision_buckets.flush_all();
                progress->record_completed(
                    result.partition_index);

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

                if (result.removed_collision_count >
                    std::numeric_limits<u64>::max() -
                        total_removed_collision_count)
                {
                    throw std::overflow_error(
                        "total removed collision count overflow");
                }

                total_removed_collision_count +=
                    static_cast<u64>(
                        result.removed_collision_count);
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

        if (!stop_requested &&
            processed_partitions != pending_partitions.size())
        {
            throw std::runtime_error(
                "not all partitions produced a result");
        }

        collision_buckets.flush_all();

        if (stop_requested &&
            progress->completed_count() < partitions.size())
        {
            std::printf(
                "Stop requested. Checkpointed %zu partitions this "
                "run; %zu/%zu total partitions are complete.\n",
                processed_partitions,
                progress->completed_count(),
                partitions.size());
        }
        else {
            std::printf(
                "Appended %llu collisions this run; all %zu "
                "partitions are complete (%llu non-primitive "
                "collisions removed).\n",
                (unsigned long long)total_collision_count,
                partitions.size(),
                (unsigned long long)total_removed_collision_count);
        }
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

int main(int argc, char* argv[])
{
#ifdef MAGIC_SQUARE_CUBES_ENABLE_MPI
    try {
        mpi_environment mpi(argc, argv);

        try {
            return run_stage1(
                argc,
                argv,
                mpi.rank(),
                mpi.size());
        }
        catch (const std::exception& error) {
            std::fprintf(
                stderr,
                "MPI stage 1 failed: %s\n",
                error.what());

            if (mpi.size() > 1)
                mpi.abort(1);

            return 1;
        }
    }
    catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Unable to initialize MPI: %s\n",
            error.what());
        return 1;
    }
#else
    return run_stage1(argc, argv, 0, 1);
#endif
}
