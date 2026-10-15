CXX ?= g++
HIPCC ?= hipcc

CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
DEBUG_CXXFLAGS ?= -std=c++17 -O0 -g3 -Wall -Wextra -pthread
HIPFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
HIP_DEBUG_FLAGS ?= -std=c++17 -O0 -g3 -Wall -Wextra -pthread
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

TARGET := magic-square-cubes-stage1
DEBUG_TARGET := magic-square-cubes-stage1-debug
HIP_TARGET := magic-square-cubes-stage1-hip
HIP_DEBUG_TARGET := magic-square-cubes-stage1-hip-debug
SOURCES := stage1.cpp search_cpu.cpp
HEADERS := search.h search_cpu.h
HIP_SOURCES := $(SOURCES) search_gpu.cpp
HIP_HEADERS := $(HEADERS) search_gpu.h

.PHONY: all debug hip hip-debug gpu clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(DEBUG_CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

hip gpu: $(HIP_TARGET)

$(HIP_TARGET): $(HIP_SOURCES) $(HIP_HEADERS)
	$(HIPCC) $(CPPFLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_HIP $(HIPFLAGS) $(HIP_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

hip-debug: $(HIP_DEBUG_TARGET)

$(HIP_DEBUG_TARGET): $(HIP_SOURCES) $(HIP_HEADERS)
	$(HIPCC) $(CPPFLAGS) -DMAGIC_SQUARE_CUBES_ENABLE_HIP $(HIP_DEBUG_FLAGS) $(HIP_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	$(RM) $(TARGET) $(DEBUG_TARGET) $(HIP_TARGET) $(HIP_DEBUG_TARGET)
