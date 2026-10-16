CXX ?= g++
HIPCC ?= hipcc
MPICXX ?= mpicxx

CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
DEBUG_CXXFLAGS ?= -std=c++17 -O0 -g3 -Wall -Wextra -pthread
HIPFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
HIP_DEBUG_FLAGS ?= -std=c++17 -O0 -g3 -Wall -Wextra -pthread
MPI_COMPILE_FLAGS ?= $(shell $(MPICXX) --showme:compile)
MPI_LINK_FLAGS ?= $(shell $(MPICXX) --showme:link)
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

TARGET := magic-square-cubes-stage1
DEBUG_TARGET := magic-square-cubes-stage1-debug
HIP_TARGET := magic-square-cubes-stage1-hip
HIP_DEBUG_TARGET := magic-square-cubes-stage1-hip-debug
MPI_TARGET := magic-square-cubes-stage1-mpi
MPI_DEBUG_TARGET := magic-square-cubes-stage1-mpi-debug
HIP_MPI_TARGET := magic-square-cubes-stage1-hip-mpi
HIP_MPI_DEBUG_TARGET := magic-square-cubes-stage1-hip-mpi-debug
CPU_GPU_TEST_TARGET := magic-square-cubes-stage1-cpu-gpu-test
SOURCES := stage1.cpp search_cpu.cpp
HEADERS := search.h search_cpu.h
HIP_SOURCES := $(SOURCES) search_gpu.cpp
HIP_HEADERS := $(HEADERS) search_gpu.h

.PHONY: all debug mpi mpi-debug hip hip-debug gpu hip-mpi hip-mpi-debug cpu-gpu-test test-cpu-gpu clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(DEBUG_CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

mpi: $(MPI_TARGET)

$(MPI_TARGET): $(SOURCES) $(HEADERS)
	$(MPICXX) $(CPPFLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_MPI $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

mpi-debug: $(MPI_DEBUG_TARGET)

$(MPI_DEBUG_TARGET): $(SOURCES) $(HEADERS)
	$(MPICXX) $(CPPFLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_MPI $(DEBUG_CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

hip gpu: $(HIP_TARGET)

$(HIP_TARGET): $(HIP_SOURCES) $(HIP_HEADERS)
	$(HIPCC) $(CPPFLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_HIP $(HIPFLAGS) $(HIP_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

hip-debug: $(HIP_DEBUG_TARGET)

$(HIP_DEBUG_TARGET): $(HIP_SOURCES) $(HIP_HEADERS)
	$(HIPCC) $(CPPFLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_HIP $(HIP_DEBUG_FLAGS) $(HIP_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

hip-mpi: $(HIP_MPI_TARGET)

$(HIP_MPI_TARGET): $(HIP_SOURCES) $(HIP_HEADERS)
	$(HIPCC) $(CPPFLAGS) $(MPI_COMPILE_FLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_HIP -DMAGIC_SQUARE_CUBES_ENABLE_MPI $(HIPFLAGS) $(HIP_SOURCES) $(LDFLAGS) $(LDLIBS) $(MPI_LINK_FLAGS) -o $@

hip-mpi-debug: $(HIP_MPI_DEBUG_TARGET)

$(HIP_MPI_DEBUG_TARGET): $(HIP_SOURCES) $(HIP_HEADERS)
	$(HIPCC) $(CPPFLAGS) $(MPI_COMPILE_FLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_HIP -DMAGIC_SQUARE_CUBES_ENABLE_MPI $(HIP_DEBUG_FLAGS) $(HIP_SOURCES) $(LDFLAGS) $(LDLIBS) $(MPI_LINK_FLAGS) -o $@

cpu-gpu-test: $(CPU_GPU_TEST_TARGET)

$(CPU_GPU_TEST_TARGET): stage1_cpu_gpu_test.cpp search.h
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) stage1_cpu_gpu_test.cpp $(LDFLAGS) $(LDLIBS) -o $@

test-cpu-gpu: $(HIP_TARGET) $(CPU_GPU_TEST_TARGET)
	./$(CPU_GPU_TEST_TARGET) --stage1 ./$(HIP_TARGET)

clean:
	$(RM) $(TARGET) $(DEBUG_TARGET) $(MPI_TARGET) $(MPI_DEBUG_TARGET) $(HIP_TARGET) $(HIP_DEBUG_TARGET) $(HIP_MPI_TARGET) $(HIP_MPI_DEBUG_TARGET) $(CPU_GPU_TEST_TARGET)
