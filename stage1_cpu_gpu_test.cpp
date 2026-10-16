#include "search.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct normalized_collision {
    u32 a1;
    u32 b1;
    u32 a2;
    u32 b2;
    u128 signature_u;
    u128 signature_v;
};

static bool operator<(
    const normalized_collision& left,
    const normalized_collision& right)
{
    if (left.a1 != right.a1)
        return left.a1 < right.a1;
    if (left.b1 != right.b1)
        return left.b1 < right.b1;
    if (left.a2 != right.a2)
        return left.a2 < right.a2;
    if (left.b2 != right.b2)
        return left.b2 < right.b2;
    if (left.signature_u != right.signature_u)
        return left.signature_u < right.signature_u;
    return left.signature_v < right.signature_v;
}

static bool operator==(
    const normalized_collision& left,
    const normalized_collision& right)
{
    return
        left.a1 == right.a1 &&
        left.b1 == right.b1 &&
        left.a2 == right.a2 &&
        left.b2 == right.b2 &&
        left.signature_u == right.signature_u &&
        left.signature_v == right.signature_v;
}

static normalized_collision normalize(
    const collision_record& collision)
{
    std::pair<u32, u32> first{
        collision.a1,
        collision.b1
    };
    std::pair<u32, u32> second{
        collision.a2,
        collision.b2
    };

    if (second < first)
        std::swap(first, second);

    return {
        first.first,
        first.second,
        second.first,
        second.second,
        collision.signature.u,
        collision.signature.v
    };
}

static std::string to_decimal(u128 value)
{
    if (value == 0)
        return "0";

    std::string result;

    while (value != 0) {
        result.push_back(
            static_cast<char>('0' + value % 10));
        value /= 10;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

static void print_collision(
    const char* label,
    const normalized_collision& collision)
{
    std::fprintf(
        stderr,
        "%s: (%u,%u),(%u,%u), signature=(%s,%s)\n",
        label,
        collision.a1,
        collision.b1,
        collision.a2,
        collision.b2,
        to_decimal(collision.signature_u).c_str(),
        to_decimal(collision.signature_v).c_str());
}

static std::filesystem::path make_test_directory()
{
    std::array<char, 64> path{};
    const int length = std::snprintf(
        path.data(),
        path.size(),
        "/tmp/magic-square-cubes-cpu-gpu-XXXXXX");

    if (length < 0 ||
        static_cast<std::size_t>(length) >= path.size())
    {
        throw std::runtime_error(
            "unable to construct temporary-directory template");
    }

    if (::mkdtemp(path.data()) == nullptr) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "unable to create temporary test directory");
    }

    return path.data();
}

static std::string quote_for_display(std::string_view argument)
{
    std::string quoted("'");

    for (const char character : argument) {
        if (character == '\'')
            quoted += "'\\''";
        else
            quoted.push_back(character);
    }

    quoted.push_back('\'');
    return quoted;
}

static void run_stage1(
    const std::filesystem::path& executable,
    const std::filesystem::path& working_directory,
    std::string_view project,
    std::string_view backend,
    const std::vector<std::string>& forwarded_arguments)
{
    std::vector<std::string> arguments{
        executable.string(),
        "--project",
        std::string(project),
        "--backend",
        std::string(backend)
    };

    arguments.insert(
        arguments.end(),
        forwarded_arguments.begin(),
        forwarded_arguments.end());

    std::printf("Running:");
    for (const auto& argument : arguments)
        std::printf(" %s", quote_for_display(argument).c_str());
    std::printf("\n");
    std::fflush(stdout);

    const pid_t child = ::fork();

    if (child < 0) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "unable to fork stage 1");
    }

    if (child == 0) {
        if (::chdir(working_directory.c_str()) != 0) {
            std::fprintf(
                stderr,
                "Unable to enter %s: %s\n",
                working_directory.string().c_str(),
                std::strerror(errno));
            _exit(126);
        }

        std::vector<char*> child_arguments;
        child_arguments.reserve(arguments.size() + 1);

        for (auto& argument : arguments)
            child_arguments.push_back(argument.data());

        child_arguments.push_back(nullptr);
        ::execv(executable.c_str(), child_arguments.data());

        std::fprintf(
            stderr,
            "Unable to execute %s: %s\n",
            executable.string().c_str(),
            std::strerror(errno));
        _exit(127);
    }

    int status = 0;

    while (::waitpid(child, &status, 0) < 0) {
        if (errno == EINTR)
            continue;

        throw std::system_error(
            errno,
            std::generic_category(),
            "unable to wait for stage 1");
    }

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
        return;

    if (WIFEXITED(status)) {
        throw std::runtime_error(
            std::string(backend) +
            " stage-1 run exited with status " +
            std::to_string(WEXITSTATUS(status)));
    }

    if (WIFSIGNALED(status)) {
        throw std::runtime_error(
            std::string(backend) +
            " stage-1 run was terminated by signal " +
            std::to_string(WTERMSIG(status)));
    }

    throw std::runtime_error(
        std::string(backend) +
        " stage-1 run did not exit normally");
}

static std::filesystem::path find_bucket_directory(
    const std::filesystem::path& project_directory)
{
    std::vector<std::filesystem::path> candidates;

    for (const auto& entry :
         std::filesystem::directory_iterator(project_directory))
    {
        if (!entry.is_directory())
            continue;

        const auto buckets = entry.path() / "buckets";

        if (std::filesystem::is_directory(buckets))
            candidates.push_back(buckets);
    }

    if (candidates.size() != 1) {
        throw std::runtime_error(
            "expected exactly one hostname bucket directory beneath " +
            project_directory.string());
    }

    return candidates.front();
}

static std::vector<std::string> bucket_filenames(
    const std::filesystem::path& directory)
{
    std::vector<std::string> filenames;

    for (const auto& entry :
         std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file()) {
            filenames.push_back(
                entry.path().filename().string());
        }
    }

    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

static std::vector<normalized_collision> read_bucket(
    const std::filesystem::path& path)
{
    const std::uintmax_t byte_count =
        std::filesystem::file_size(path);

    if (byte_count % sizeof(collision_record) != 0) {
        throw std::runtime_error(
            "bucket size is not a multiple of collision_record: " +
            path.string());
    }

    const std::uintmax_t record_count =
        byte_count / sizeof(collision_record);

    if (record_count >
        std::vector<collision_record>().max_size())
    {
        throw std::length_error(
            "bucket is too large to compare: " +
            path.string());
    }

    std::vector<collision_record> records(
        static_cast<std::size_t>(record_count));
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "unable to open bucket " + path.string());
    }

    if (byte_count >
        static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamsize>::max()))
    {
        throw std::length_error(
            "bucket exceeds streamsize: " + path.string());
    }

    file.read(
        reinterpret_cast<char*>(records.data()),
        static_cast<std::streamsize>(byte_count));

    if (!file && byte_count != 0) {
        throw std::runtime_error(
            "unable to read bucket " + path.string());
    }

    std::vector<normalized_collision> normalized;
    normalized.reserve(records.size());

    for (const auto& record : records)
        normalized.push_back(normalize(record));

    std::sort(normalized.begin(), normalized.end());
    return normalized;
}

static std::size_t compare_buckets(
    const std::filesystem::path& cpu_directory,
    const std::filesystem::path& gpu_directory)
{
    const auto cpu_filenames =
        bucket_filenames(cpu_directory);
    const auto gpu_filenames =
        bucket_filenames(gpu_directory);

    constexpr std::size_t expected_bucket_count = 32;

    if (cpu_filenames.size() != expected_bucket_count ||
        gpu_filenames.size() != expected_bucket_count)
    {
        throw std::runtime_error(
            "expected 32 bucket files from each backend");
    }

    if (cpu_filenames != gpu_filenames) {
        throw std::runtime_error(
            "CPU and GPU bucket filename sets differ");
    }

    std::size_t total_records = 0;

    for (const auto& filename : cpu_filenames) {
        const auto cpu_records =
            read_bucket(cpu_directory / filename);
        const auto gpu_records =
            read_bucket(gpu_directory / filename);

        if (cpu_records != gpu_records) {
            std::fprintf(
                stderr,
                "Bucket %s differs: CPU has %zu records, "
                "GPU has %zu records.\n",
                filename.c_str(),
                cpu_records.size(),
                gpu_records.size());

            const std::size_t shared_count =
                std::min(cpu_records.size(), gpu_records.size());
            std::size_t difference = 0;

            while (difference < shared_count &&
                   cpu_records[difference] ==
                       gpu_records[difference])
            {
                ++difference;
            }

            if (difference < cpu_records.size()) {
                print_collision(
                    "First differing CPU record",
                    cpu_records[difference]);
            }

            if (difference < gpu_records.size()) {
                print_collision(
                    "First differing GPU record",
                    gpu_records[difference]);
            }

            throw std::runtime_error(
                "CPU and GPU collision buckets do not match");
        }

        total_records += cpu_records.size();
        std::printf(
            "Matched %s: %zu records\n",
            filename.c_str(),
            cpu_records.size());
    }

    return total_records;
}

static bool forbidden_forwarded_option(
    std::string_view option)
{
    return
        option == "-p" ||
        option == "--project" ||
        option == "--backend";
}

static void print_usage(const char* program)
{
    std::fprintf(
        stderr,
        "Usage: %s [--stage1 PATH] [--keep-output] "
        "[-- STAGE1_OPTIONS...]\n"
        "\n"
        "Runs the HIP-enabled stage-1 executable once with the CPU "
        "backend and once with the GPU backend, then compares every "
        "bucket as an unordered multiset of collision records.\n"
        "\n"
        "Defaults passed to stage 1 when no options follow --:\n"
        "  -a 1:2000 -b 1:2000 -m 16MB --gpu-method hash -j 1\n",
        program);
}

} // namespace

int main(int argc, char* argv[])
{
    std::filesystem::path stage1_executable =
        "./magic-square-cubes-stage1-hip";
    bool keep_output = false;
    bool arguments_parsed = false;
    std::vector<std::string> forwarded_arguments;

    try {
        int i = 1;

        for (; i < argc; ++i) {
            const std::string_view option = argv[i];

            if (option == "--") {
                ++i;
                break;
            }

            if (option == "-h" || option == "--help") {
                print_usage(argv[0]);
                return 0;
            }

            if (option == "--keep-output") {
                keep_output = true;
                continue;
            }

            if (option == "--stage1") {
                if (++i == argc) {
                    throw std::invalid_argument(
                        "missing path after --stage1");
                }

                stage1_executable = argv[i];
                continue;
            }

            throw std::invalid_argument(
                "unknown test option before --: " +
                std::string(option));
        }

        for (; i < argc; ++i) {
            if (forbidden_forwarded_option(argv[i])) {
                throw std::invalid_argument(
                    std::string(argv[i]) +
                    " is controlled by the test program");
            }

            forwarded_arguments.emplace_back(argv[i]);
        }

        if (forwarded_arguments.empty()) {
            forwarded_arguments = {
                "-a", "1:2000",
                "-b", "1:2000",
                "-m", "16MB",
                "--gpu-method", "hash",
                "-j", "1"
            };
        }

        arguments_parsed = true;

        stage1_executable =
            std::filesystem::absolute(stage1_executable);

        if (!std::filesystem::is_regular_file(
                stage1_executable))
        {
            throw std::runtime_error(
                "stage-1 executable does not exist: " +
                stage1_executable.string());
        }

        const auto test_directory = make_test_directory();
        bool succeeded = false;

        try {
            std::printf(
                "Test workspace: %s\n",
                test_directory.string().c_str());

            run_stage1(
                stage1_executable,
                test_directory,
                "cpu",
                "cpu",
                forwarded_arguments);
            run_stage1(
                stage1_executable,
                test_directory,
                "gpu",
                "gpu",
                forwarded_arguments);

            const auto cpu_buckets = find_bucket_directory(
                test_directory / "cpu");
            const auto gpu_buckets = find_bucket_directory(
                test_directory / "gpu");

            const std::size_t record_count = compare_buckets(
                cpu_buckets,
                gpu_buckets);

            std::printf(
                "PASS: all buckets match exactly (%zu collision "
                "records).\n",
                record_count);
            succeeded = true;
        }
        catch (...) {
            std::fprintf(
                stderr,
                "Test output retained in %s\n",
                test_directory.string().c_str());
            throw;
        }

        if (succeeded && !keep_output) {
            std::error_code error;
            std::filesystem::remove_all(
                test_directory,
                error);

            if (error) {
                std::fprintf(
                    stderr,
                    "Warning: unable to remove %s: %s\n",
                    test_directory.string().c_str(),
                    error.message().c_str());
            }
        }
        else if (succeeded) {
            std::printf(
                "Test output retained in %s\n",
                test_directory.string().c_str());
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::fprintf(stderr, "CPU/GPU bucket test failed: %s\n", error.what());

        if (!arguments_parsed)
            print_usage(argv[0]);

        return 1;
    }
}
