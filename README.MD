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

Before bucket insertion, stage 1 removes non-primitive collisions for which
`gcd(a1, b1, a2, b2) > 1`. This eliminates collisions that are integer-scaled
copies of a smaller collision.

## Requirements

The CPU build requires:

- A C++17 compiler such as `g++`
- POSIX threads

MPI builds require an MPI implementation that provides `mpicxx` and `mpirun`
(for example, Open MPI). GPU builds additionally require AMD ROCm with
`hipcc` and a HIP-compatible GPU.

## Building

Build the optimized CPU stage-1, HIP stage-1, and stage-2 executables:

```sh
make
```

This creates `magic-square-cubes-stage1`, `magic-square-cubes-stage1-hip`, and
`magic-square-cubes-stage2`.

Other build targets are:

```sh
make debug       # CPU stage-1 and stage-2 debug builds
make stage1      # Optimized CPU stage-1 build
make stage2      # Optimized stage-2 build
make stage2-debug # Stage-2 debug build
make mpi         # Optimized CPU + MPI build
make mpi-debug   # CPU + MPI debug build
make hip         # Optimized HIP build
make hip-debug   # HIP debug build
make hip-mpi     # Optimized HIP + MPI build
make hip-mpi-debug # HIP + MPI debug build
make cpu-gpu-test # Build the CPU/GPU bucket comparison utility
make test-cpu-gpu # Build and run the default CPU/GPU comparison
make clean       # Remove generated executables
```

The CPU MPI targets create `magic-square-cubes-stage1-mpi` and
`magic-square-cubes-stage1-mpi-debug`. The HIP targets create
`magic-square-cubes-stage1-hip` and `magic-square-cubes-stage1-hip-debug`.
The combined HIP and MPI targets create `magic-square-cubes-stage1-hip-mpi`
and `magic-square-cubes-stage1-hip-mpi-debug`.

## CPU/GPU bucket comparison

`magic-square-cubes-stage1-cpu-gpu-test` runs the HIP-enabled stage-1 binary
twice in a temporary directory: once with `--backend cpu` and once with
`--backend gpu`. It then compares corresponding bucket files as unordered
multisets of complete collision records. Record multiplicity, both `(a,b)`
pairs, and both 128-bit signature values must match. Only the order of records
within a bucket and the order of the two representations within a collision
are canonicalized before comparison.

Build and run the default hash-table test with:

```sh
make test-cpu-gpu
```

The default search uses `a,b = 1:2000` and a 16 MiB buffer. To specify stage-1
options, place them after `--`:

```sh
make cpu-gpu-test hip

./magic-square-cubes-stage1-cpu-gpu-test \
    --stage1 ./magic-square-cubes-stage1-hip \
    -- -a 1:10000 -b 1:10000 -m 64MB --gpu-method hash -d 0
```

Use `--gpu-method radix` to compare the radix implementation instead. The
test removes its temporary workspace after a successful comparison. It keeps
the workspace after a failure for diagnosis; `--keep-output` also preserves a
successful run.

## Stage 1 usage

```text
magic-square-cubes-stage1 -p NAME [options]

  -p NAME              required project/output directory name
  --project NAME       same as -p
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

There is no default project name. It must be supplied for every run. Project
names are single directory components of at most 255 characters and may
contain letters, digits, `.`, `-`, and `_`.

Buffer sizes may use `B`, `KB`, `MB`, or `GB`. These suffixes use powers of
1024, so `1GB` means 1,073,741,824 bytes.

Both ranges must contain positive integers no larger than `UINT32_MAX`.
Regardless of the supplied ranges, only pairs satisfying `a < b` are searched.

### CPU example

Search `a` and `b` from 1 through 50,000 using eight CPU workers and a 1 GiB
pair buffer per worker:

```sh
./magic-square-cubes-stage1 \
    --project example-search \
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
    --project example-search \
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
    --project example-search \
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
    --project example-search \
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
- The progress-file path and resume status
- Bucket configuration and output directory
- Active worker count
- GPU grid and block sizes for generation and collision scanning
- Hash-table slot count, load factor, and storage, or the rocPRIM radix-sort
  item count and temporary-storage requirement

After each partition, it reports the collision count, elapsed search time, and
throughput in `Mops/sec`. One operation means one `(a,b)` pair processed. The
reported search time excludes writing collisions to bucket files.

## Collision files

The program creates the named project directory beneath the current working
directory. Results are written as:

```text
<project>/buckets/
```

There is one bucket set per project, regardless of the backend or the number
of CPU or GPU workers. In an MPI run, rank 0 receives every collision list and
writes this shared bucket set. GPU identifiers and hostnames are not part of
the output path.

There are 32 bucket files named:

```text
collision-bucket-u32-v1-00.bin
...
collision-bucket-u32-v1-31.bin
```

The files are unsorted and opened in append mode. Running the same search more
than once with the same project name resumes from the project's progress file;
partitions already recorded as complete are not searched or appended again.

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

## Stopping and resuming

Partition progress is stored in:

```text
<project>/stage1-progress-v1.bin
```

The coordinator records when each partition is distributed and when it is
complete. A partition is recorded as complete only after its collision records
have been flushed to the bucket files. On restart, completed partitions are
skipped. A partition that was distributed but did not complete is searched
again.

Press Ctrl-C, or send `SIGTERM`, to request a graceful stop. The coordinator
stops distributing new work, waits for currently active partitions, flushes
their collision records, and checkpoints them. Run the same command again to
continue the remaining work. MPI jobs use the same behavior: rank 0 owns the
progress file and stops assigning work to worker ranks.

The search ranges, pair-buffer target, and generated partition layout are
included in the progress-file fingerprint. Reusing a project name with
different search settings is rejected. The progress file belongs to the
overall project rather than a particular host, so an MPI job can resume with
rank 0 running on a different coordinator host while using the same shared
project directory.

A forced termination such as `SIGKILL`, a machine failure, or a power loss can
occur after bucket data is appended but before the completion record is
written. In that case, retrying the incomplete partition can append duplicate
collision records. Graceful `SIGINT` and `SIGTERM` shutdown avoids this window.

## Stage 2 signature scan

Stage 2 reads the single collision bucket set produced by the stage-1 manager
and finds records with the same complete `(u,v)` signature. It locates the
project's `<project>/buckets/` directory, sorts each bucket by signature, and
scans equal-signature runs. Only one of the 32 buckets is held in memory at a
time.

Build stage 2 with:

```sh
make stage2
```

Run it against a completed stage-1 project:

```sh
./magic-square-cubes-stage2 --project example-search
```

By default, stage 2 prints the first 10 repeated signatures in each bucket,
their group sizes, and the first three collision records in each group. It
also prints per-bucket and overall counts. This makes unexpected signature
collisions visible without flooding the output. To print every repeated
signature and all of its collision records, use:

```sh
./magic-square-cubes-stage2 \
    --project example-search \
    --show-matches
```

The initial stage-2 implementation sorts each bucket in memory. A future
external-sort path may be needed if a single bucket grows beyond available
RAM.
