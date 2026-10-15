#ifndef MAGIC_SQUARE_CUBES_SEARCH_GPU_H
#define MAGIC_SQUARE_CUBES_SEARCH_GPU_H

#include "search.h"

#include <cstddef>

std::size_t gpu_pair_record_size() noexcept;
u64 gpu_pair_record_max_count() noexcept;

partition_search_function search_partition_collisions_gpu;

#endif
