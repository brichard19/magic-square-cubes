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

MPI builds require an MPI implementation that provides `mpicxx` and `mpirun`
(for example, Open MPI). GPU builds additionally require AMD ROCm with
`hipcc` and a HIP-compatible GPU.

## Building

Build the optimized CPU executable:

```sh
make
```

This creates `magic-square-cubes-stage1`.

Other build targets are:

```sh
make debug       # CPU debug build
make mpi         # Optimized CPU + MPI build
make mpi-debug   # CPU + MPI debug build
make hip         # Optimized HIP build
make hip-debug   # HIP debug build
make hip-mpi     # Optimized HIP + MPI build
make hip-mpi-debug # HIP + MPI debug build
make clean       # Remove generated executables
```

The CPU MPI targets create `magic-square-cubes-stage1-mpi` and
`magic-square-cubes-stage1-mpi-debug`. The HIP targets create
`magic-square-cubes-stage1-hip` and `magic-square-cubes-stage1-hip-debug`.
The combined HIP and MPI targets create `magic-square-cubes-stage1-hip-mpi`
and `magic-square-cubes-stage1-hip-mpi-debug`.

## Usage

```text
magic-square-cubes-stage1 [options]

  -a MIN:MAX           inclusive range for a
  -b MIN:MAX           inclusive range for b
  -m SIZE              target pair-buffer size
  --buffer-size SIZE   same as -m
  -j N                 number of CPU worker threads
  --threads N          same as -j
  -d N                 HIP device index (first local device under MPI)
  --hip-device N       same as -d
  --backend cpu|gpu    select the search backend
  --gpu-method TYPE    GPU method: hash or radix
  -h, --help           show command-line help
```

Defaults:

```text
-a 1:100000 -b 2:100000 -m 1GB -j 1 -d 0 --backend cpu --gpu-method hash
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
    --hip-device 0 \
    --gpu-method hash \
    --backend gpu
```

The HIP executable also contains the CPU backend, so it may be run with
`--backend cpu`.

The GPU backend uses one host worker. The `-j` value therefore does not
increase GPU concurrency. Use `-d` or `--hip-device` to select a device by its
zero-based HIP index. The GPU worker selects that device before querying its
properties or launching kernels.

The GPU backend offers two collision methods:

- `--gpu-method hash` is the default. It inserts every pair into an
  open-addressed GPU hash table and then scans the completed probe chains for
  exact equal-sum matches.
- `--gpu-method radix` generates structure-of-arrays pair buffers, sorts them
  by sum with rocPRIM, and scans adjacent equal sums.

Keeping both methods makes it possible to compare throughput and memory use on
the target GPU without rebuilding the program.

### MPI examples

The MPI scheduler also works with the CPU backend. For four CPU worker
processes plus the coordinator:

```sh
make mpi

mpirun -np 5 ./magic-square-cubes-stage1-mpi \
    -a 1:50000 \
    -b 1:50000 \
    -m 1GB \
    --backend cpu
```

The MPI build uses rank 0 as a coordinator and every other rank as a worker.
Consequently, launch one more rank than the number of GPUs. For four GPUs on
one machine:

```sh
make hip-mpi

mpirun -np 5 ./magic-square-cubes-stage1-hip-mpi \
    -a 1:50000 \
    -b 1:50000 \
    -m 1GB \
    --backend gpu
```

Rank 0 creates the partitions, assigns them dynamically, receives each
completed collision vector, and is the only process that writes bucket files.
Workers request work implicitly by returning a result, so faster GPUs receive
more partitions. Collision-vector transfers are split into chunks when needed
to stay within MPI's per-message count limit.

Worker ranks are mapped to GPUs by their node-local rank. On the coordinator's
node, rank 0 is omitted from that numbering; on every other node, the first
local worker uses GPU 0. `-d N` changes GPU 0 in this description to GPU `N`,
which is useful when reserving lower-numbered devices. The `-j` option is not
used in distributed mode because each MPI worker processes one partition at a
time.

The HIP MPI executable also contains the CPU backend and may be run with
`--backend cpu`, but the CPU-only MPI executable does not require ROCm.

## Partitioning and memory

The pair-buffer size is a partitioning guideline, not a hard memory limit.
Partitions are formed from half-open sum ranges:

```text
s_min <= a^3 + b^3 < s_max
```

The program may allow a partition to exceed the target when it cannot be split
without dividing an equal-sum group. On the CPU, the buffer target applies to
each active worker.

With `--gpu-method hash`, the HIP backend performs these stages for every
partition:

1. The CPU calculates each valid `[a, b0..bN]` span.
2. One GPU thread block processes each span and inserts its pairs into an
   open-addressed table with a maximum load factor of 70%.
3. A 64-bit atomic tag claims each slot. The full unsigned 128-bit sum remains
   in the table and is always compared before reporting a collision, so tag
   collisions do not affect correctness.
4. GPU kernels first count and then collect equal-sum pairs by scanning probe
   chains. Every pair is retained in the table, so a sum with more than two
   representations produces every pairwise collision.
5. Compact collision inputs are copied to the CPU, where their GCD-based
   signatures are calculated.

The count pass determines the exact collision-buffer size; no maximum
collision estimate is required. Hash-table partitioning budgets 48 bytes per
input pair. The allocated structure-of-arrays table uses 32 bytes per slot and
has approximately 10 slots for every 7 pairs.

With `--gpu-method radix`, the HIP backend performs these stages:

1. The CPU calculates each valid `[a, b0..bN]` span and its exact prefix-sum
   output offset.
2. One GPU thread block generates the pairs for each span. Its thread count is
   set to the active GPU's reported wavefront size.
3. rocPRIM performs a stable device-wide radix sort using the unsigned 128-bit
   sums as keys and the separate `a` and `b` arrays as values.
4. GPU kernels count and compact equal-sum pair combinations.
5. Compact collision inputs are copied to the CPU, where their GCD-based
   signatures are calculated.

Radix-sort pair data uses a structure-of-arrays layout: sums are stored as
contiguous unsigned 128-bit values, with separate contiguous `uint32_t` arrays
for `a` and `b`. This keeps each field naturally aligned and allows adjacent
GPU threads to issue coalesced accesses.

Radix sorting does not require power-of-two padding. It does require a second
set of sum, `a`, and `b` arrays for its output, plus rocPRIM temporary storage.
The GPU therefore needs at least twice the target pair-buffer space, along with
additional space for radix-sort workspace, spans, and collisions.

## Progress output

At startup, the program reports:

- The selected backend
- The requested `a` and `b` ranges
- Pair-buffer size and pair-record count
- Every generated partition
- Bucket configuration and output directory
- Active worker count
- GPU grid and block sizes for generation and collision scanning
- Hash-table slot count, load factor, and storage, or the rocPRIM radix-sort
  item count and temporary-storage requirement

After each partition, it reports the collision count, elapsed search time, and
throughput in `Mops/sec`. One operation means one `(a,b)` pair processed. The
reported search time excludes writing collisions to bucket files.

## Collision files

Results are written beneath the current directory:

```text
<hostname>/CPU/
<hostname>/GPU-<device-ID>/
<hostname>/MPI-CPU/
<hostname>/MPI-GPU/
```

GPU directories use the 16-byte UUID reported in `hipDeviceProp_t`. AMD HIP
commonly reports this as a printable serial number, producing a directory such
as `GPU-295cea3615cef364`. A binary UUID is formatted as 32 hexadecimal digits.
If the runtime reports an all-zero UUID, the program instead uses
`GPU-pci-<domain>-<bus>-<device>`.

MPI runs aggregate every worker's collisions on rank 0, so they use one
`MPI-CPU` or `MPI-GPU` directory beneath the coordinator's hostname rather
than a directory for each worker device.

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
