#include "search.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

constexpr std::size_t bucket_count = 32;
constexpr std::size_t default_matching_groups_to_print = 10;
constexpr std::size_t default_records_per_group_to_print = 3;

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
        "Usage: %s -p NAME [--show-matches]\n"
        "  -p, --project NAME  stage-1 project directory\n"
        "  --show-matches      print every repeated signature and record\n"
        "  -h, --help          show this help\n",
        program);
}

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

static std::filesystem::path find_bucket_directory(
    const std::filesystem::path& project_directory)
{
    if (!std::filesystem::is_directory(project_directory)) {
        throw std::runtime_error(
            "project directory does not exist: " +
            project_directory.string());
    }

    const auto directory = project_directory / "buckets";

    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error(
            "bucket directory does not exist: " +
            directory.string());
    }

    for (std::size_t i = 0; i < bucket_count; ++i) {
        const auto path = directory / bucket_filename(i);

        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error(
                "missing bucket file " + path.string());
        }
    }

    return directory;
}

static void read_file_into(
    const std::filesystem::path& path,
    collision_record* destination,
    std::size_t byte_count)
{
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "unable to open bucket " + path.string());
    }

    auto* output = reinterpret_cast<char*>(destination);
    std::size_t remaining = byte_count;
    constexpr std::size_t maximum_chunk =
        std::size_t(1) << 30;

    while (remaining != 0) {
        const std::size_t chunk =
            std::min(remaining, maximum_chunk);

        file.read(
            output,
            static_cast<std::streamsize>(chunk));

        if (!file) {
            throw std::runtime_error(
                "unable to read bucket " + path.string());
        }

        output += chunk;
        remaining -= chunk;
    }
}

static std::vector<collision_record> read_bucket(
    const std::filesystem::path& directory,
    std::size_t bucket_index)
{
    static_assert(
        std::is_trivially_copyable_v<collision_record>);

    const std::string filename =
        bucket_filename(bucket_index);
    const auto path = directory / filename;
    const std::uintmax_t file_bytes =
        std::filesystem::file_size(path);

    if (file_bytes % sizeof(collision_record) != 0) {
        throw std::runtime_error(
            "bucket size is not a multiple of collision_record: " +
            path.string());
    }

    if (file_bytes >
        std::numeric_limits<std::size_t>::max())
    {
        throw std::length_error(
            "bucket byte count overflows size_t: " +
            path.string());
    }

    const std::size_t byte_count =
        static_cast<std::size_t>(file_bytes);
    const std::size_t total_records =
        byte_count / sizeof(collision_record);

    std::vector<collision_record> records;

    if (total_records > records.max_size()) {
        throw std::length_error(
            "combined bucket is too large for memory");
    }

    records.resize(total_records);

    read_file_into(
        path,
        records.empty() ? nullptr : records.data(),
        byte_count);

    return records;
}

static bool signature_less(
    const collision_record& left,
    const collision_record& right)
{
    if (left.signature.u != right.signature.u)
        return left.signature.u < right.signature.u;
    if (left.signature.v != right.signature.v)
        return left.signature.v < right.signature.v;
    if (left.a1 != right.a1)
        return left.a1 < right.a1;
    if (left.b1 != right.b1)
        return left.b1 < right.b1;
    if (left.a2 != right.a2)
        return left.a2 < right.a2;
    return left.b2 < right.b2;
}

static bool equal_signature(
    const collision_record& left,
    const collision_record& right)
{
    return
        left.signature.u == right.signature.u &&
        left.signature.v == right.signature.v;
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

static void print_matching_group(
    std::size_t bucket_index,
    const std::vector<collision_record>& records,
    std::size_t begin,
    std::size_t end,
    bool show_records)
{
    const auto& signature = records[begin].signature;

    std::printf(
        "Bucket %02zu signature (%s, %s): %zu records\n",
        bucket_index,
        to_decimal(signature.u).c_str(),
        to_decimal(signature.v).c_str(),
        end - begin);

    const std::size_t records_to_print = show_records
        ? end - begin
        : std::min(
            end - begin,
            default_records_per_group_to_print);

    for (std::size_t i = begin;
         i < begin + records_to_print;
         ++i)
    {
        const auto& record = records[i];

        std::printf(
            "  (%u, %u) <-> (%u, %u)\n",
            record.a1,
            record.b1,
            record.a2,
            record.b2);
    }

    if (records_to_print < end - begin) {
        std::printf(
            "  ... %zu additional records not shown\n",
            end - begin - records_to_print);
    }
}

struct bucket_summary {
    std::size_t record_count = 0;
    std::size_t unique_signature_count = 0;
    std::size_t matching_signature_count = 0;
    std::size_t matching_record_count = 0;
};

static bucket_summary process_bucket(
    const std::filesystem::path& directory,
    std::size_t bucket_index,
    bool show_matches)
{
    auto records = read_bucket(directory, bucket_index);
    const auto start = std::chrono::steady_clock::now();

    std::sort(
        records.begin(),
        records.end(),
        signature_less);

    bucket_summary summary;
    summary.record_count = records.size();
    std::size_t printed_matching_groups = 0;
    std::size_t i = 0;

    while (i < records.size()) {
        std::size_t end = i + 1;

        while (end < records.size() &&
               equal_signature(records[i], records[end]))
        {
            ++end;
        }

        ++summary.unique_signature_count;

        if (end - i >= 2) {
            ++summary.matching_signature_count;
            summary.matching_record_count += end - i;

            if (show_matches ||
                printed_matching_groups <
                    default_matching_groups_to_print)
            {
                print_matching_group(
                    bucket_index,
                    records,
                    i,
                    end,
                    show_matches);
                ++printed_matching_groups;
            }
        }

        i = end;
    }

    if (summary.matching_signature_count >
        printed_matching_groups)
    {
        std::printf(
            "Bucket %02zu: %zu additional repeated signatures "
            "not shown (use --show-matches to print all)\n",
            bucket_index,
            summary.matching_signature_count -
                printed_matching_groups);
    }

    const double elapsed_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start)
            .count();

    std::printf(
        "Bucket %02zu complete: %zu records, %zu unique signatures, "
        "%zu repeated signatures containing %zu records (%.3f s)\n",
        bucket_index,
        summary.record_count,
        summary.unique_signature_count,
        summary.matching_signature_count,
        summary.matching_record_count,
        elapsed_seconds);
    std::fflush(stdout);

    return summary;
}

static void add_checked(
    std::size_t& destination,
    std::size_t value,
    const char* description)
{
    if (value >
        std::numeric_limits<std::size_t>::max() - destination)
    {
        throw std::overflow_error(description);
    }

    destination += value;
}

} // namespace

int main(int argc, char* argv[])
{
    std::string project_name;
    bool show_matches = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];

        if (option == "-h" || option == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        if (option == "--show-matches") {
            show_matches = true;
            continue;
        }

        if (option != "-p" && option != "--project") {
            std::fprintf(
                stderr,
                "Unknown option: %s\n",
                argv[i]);
            print_usage(argv[0]);
            return 1;
        }

        if (++i == argc) {
            std::fprintf(
                stderr,
                "Missing project name after %.*s\n",
                static_cast<int>(option.size()),
                option.data());
            print_usage(argv[0]);
            return 1;
        }

        project_name = argv[i];
    }

    if (!valid_project_name(project_name)) {
        std::fprintf(
            stderr,
            "A valid project name is required. Project names may "
            "contain only letters, digits, '.', '-', and '_'.\n");
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path project_directory(
            project_name);
        const auto bucket_directory =
            find_bucket_directory(project_directory);

        std::printf(
            "Project: %s\n"
            "Collision record size: %zu bytes\n"
            "Bucket directory: %s\n",
            project_name.c_str(),
            sizeof(collision_record),
            bucket_directory.string().c_str());

        std::fflush(stdout);

        bucket_summary total;

        for (std::size_t i = 0; i < bucket_count; ++i) {
            const auto summary = process_bucket(
                bucket_directory,
                i,
                show_matches);

            add_checked(
                total.record_count,
                summary.record_count,
                "total record count overflow");
            add_checked(
                total.unique_signature_count,
                summary.unique_signature_count,
                "total signature count overflow");
            add_checked(
                total.matching_signature_count,
                summary.matching_signature_count,
                "total matching signature count overflow");
            add_checked(
                total.matching_record_count,
                summary.matching_record_count,
                "total matching record count overflow");
        }

        std::printf(
            "Stage 2 complete: %zu records, %zu unique signatures, "
            "%zu repeated signatures containing %zu records.\n",
            total.record_count,
            total.unique_signature_count,
            total.matching_signature_count,
            total.matching_record_count);
    }
    catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Stage 2 failed: %s\n",
            error.what());
        return 1;
    }

    return 0;
}
