#ifndef MAGIC_SQUARE_CUBES_SEARCH_GPU_H
#define MAGIC_SQUARE_CUBES_SEARCH_GPU_H

#include "search.h"

#include <cstddef>
#include <string>

std::size_t gpu_pair_record_size() noexcept;
u64 gpu_pair_record_max_count() noexcept;
std::size_t gpu_hash_pair_record_size() noexcept;
u64 gpu_hash_pair_record_max_count() noexcept;
void set_hip_device(int device);
std::string hip_device_identifier(int device);

partition_search_function search_partition_collisions_gpu_radix;
partition_search_function search_partition_collisions_gpu_hash;

#endif
