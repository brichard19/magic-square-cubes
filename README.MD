# Magic Square Cubes

The stage 1 tool searches for integers with multiple representations as a sum
of two positive cubes:

```text
a^3 + b^3 = c^3 + d^3
```

Only canonical pairs with `a < b` and `c < d` are generated. The search space
is divided into ranges of the sum `S = a^3 + b^3`, allowing partitions to be
processed independently on multiple CPU cores or by the optional HIP GPU
backend.

Each collision is assigned a reduced difference signature. For the two pairs,
define:

```text
x = b^3 - a^3
y = d^3 - c^3
g = gcd(x, y)
```

The signature is:

```text
(min(x/g, y/g), max(x/g, y/g))
```

Collision records are distributed among binary bucket files using a hash of
this signature. These files are intended as input to later stages of the magic
square search.

## Requirements

The CPU build requires:

- A C++17 compiler such as `g++`
- POSIX threads

The GPU build additionally requires AMD ROCm with `hipcc` and a
HIP-compatible GPU.

## Building

Build the optimized CPU executable:

```sh
make
```

This creates `magic-square-cubes-stage1`.

Other build targets are:

```sh
make debug       # CPU debug build
make hip         # Optimized HIP build
make hip-debug   # HIP debug build
make clean       # Remove generated executables
```

The HIP targets create `magic-square-cubes-stage1-hip` and
`magic-square-cubes-stage1-hip-debug`.

## Usage

```text
magic-square-cubes-stage1 [options]

  -a MIN:MAX           inclusive range for a
  -b MIN:MAX           inclusive range for b
  -m SIZE              target pair-buffer size
  --buffer-size SIZE   same as -m
  -j N                 number of CPU worker threads
  --threads N          same as -j
  --backend cpu|gpu    select the search backend
  -h, --help           show command-line help
```

Defaults:

```text
-a 1:100000 -b 2:100000 -m 1GB -j 1 --backend cpu
```

Buffer sizes may use `B`, `KB`, `MB`, or `GB`. These suffixes use powers of
1024, so `1GB` means 1,073,741,824 bytes.

Both ranges must contain positive integers no larger than `UINT32_MAX`.
Regardless of the supplied ranges, only pairs satisfying `a < b` are searched.

### CPU example

Search `a` and `b` from 1 through 50,000 using eight CPU workers and a 1 GiB
pair buffer per worker:

```sh
./magic-square-cubes-stage1 \
    -a 1:50000 \
    -b 1:50000 \
    -m 1GB \
    -j 8 \
    --backend cpu
```

The CPU executable does not contain the GPU backend. Asking it to use
`--backend gpu` prints instructions to build the HIP executable.

### GPU example

Build and run the HIP backend:

```sh
make hip

./magic-square-cubes-stage1-hip \
    -a 1:50000 \
    -b 1:50000 \
    -m 1GB \
    --backend gpu
```

The HIP executable also contains the CPU backend, so it may be run with
`--backend cpu`.

The GPU backend uses one host worker. The `-j` value therefore does not
increase GPU concurrency.

## Partitioning and memory

The pair-buffer size is a partitioning guideline, not a hard memory limit.
Partitions are formed from half-open sum ranges:

```text
s_min <= a^3 + b^3 < s_max
```

The program may allow a partition to exceed the target when it cannot be split
without dividing an equal-sum group. On the CPU, the buffer target applies to
each active worker.

The HIP backend performs these stages for every partition:

1. The CPU calculates each valid `[a, b0..bN]` span and its exact prefix-sum
   output offset.
2. One GPU thread block generates the pairs for each span.
3. The pair buffer is padded to the next power of two and bitonic-sorted by its
   128-bit sum.
4. GPU kernels count and compact equal-sum pair combinations.
5. Compact collision inputs are copied to the CPU, where their GCD-based
   signatures are calculated.

Power-of-two padding means the GPU pair allocation can approach twice the
un-padded pair-buffer size. Additional space is required for span and collision
buffers.

## Progress output

At startup, the program reports:

- The selected backend
- The requested `a` and `b` ranges
- Pair-buffer size and pair-record count
- Every generated partition
- Bucket configuration and output directory
- Active worker count

After each partition, it reports the collision count, elapsed search time, and
throughput in `Mops/sec`. One operation means one `(a,b)` pair processed. The
reported search time excludes writing collisions to bucket files.

## Collision files

Results are written beneath the current directory:

```text
<hostname>/CPU/
<hostname>/GPU/
```

There are 32 bucket files named:

```text
collision-bucket-u32-v1-00.bin
...
collision-bucket-u32-v1-31.bin
```

The files are unsorted and opened in append mode. Running the same search more
than once in the same directory appends duplicate results unless the existing
bucket files are moved or removed first.

Each file is a raw sequence of `collision_record` objects:

```cpp
struct collision_record {
    uint32_t a1;
    uint32_t b1;
    uint32_t a2;
    uint32_t b2;
    collision_signature signature; // two unsigned 128-bit values
};
```

The common sum `S` is not stored because it can be recomputed as
`a1^3 + b1^3`.

The current binary format is native to the compiler ABI and machine byte
order; it has no file header or portability conversion. The executable reports
the collision-record size at startup. Do not combine bucket files produced by
builds with different record layouts.
