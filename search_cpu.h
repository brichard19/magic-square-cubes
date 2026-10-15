#ifndef MAGIC_SQUARE_CUBES_SEARCH_CPU_H
#define MAGIC_SQUARE_CUBES_SEARCH_CPU_H

#include "search.h"

#include <cstddef>

std::size_t cpu_pair_record_size() noexcept;
u64 cpu_pair_record_max_count() noexcept;

partition_search_function search_partition_collisions_cpu;

#endif
