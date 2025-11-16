##############################################################
# OpenACC–Enabled KLT Makefile (nvc / nvc++)
# Configured for NVIDIA GeForce RTX 3060 (Ampere architecture)
##############################################################

# Use NVIDIA HPC SDK compilers (NOT nvcc)
CC  = nvc
CXX = nvc++

# GPU Architecture for RTX 3060 (Compute Capability 8.6 - Ampere)
# cc86 = RTX 3050, 3050 Ti, 3060, 3060 Ti (GA106/GA104)
GPU_ARCH = cc86

# OpenACC GPU Flags for RTX 3060
# -acc=gpu: Target GPU instead of multicore CPU
# -gpu=$(GPU_ARCH): Specify compute capability
# -gpu=managed: Use CUDA Unified Memory (easier debugging)
# -Minfo=accel: Show accelerator kernel information
# -fast: Enable aggressive optimizations
ACCFLAGS_GPU = -acc=gpu -gpu=$(GPU_ARCH),managed -Minfo=accel -fast

# Alternative: Multicore CPU fallback (if GPU not available)
ACCFLAGS_MULTICORE = -acc=multicore -Minfo=accel -fast

# Default to GPU acceleration
ACCFLAGS = $(ACCFLAGS_GPU)

FLAG1 = -DNDEBUG

# Compile flags: enable OpenACC + your flags
CFLAGS   = $(FLAG1) $(FLAG2) $(ACCFLAGS)
CXXFLAGS = $(FLAG1) $(FLAG2) $(ACCFLAGS)

# Libraries
LIB  = -L/usr/local/lib -L/usr/lib -L.
LIBS = -lm

# Source files
EXAMPLES = example1.c example2.c example3.c example4.c example5.c

ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

##############################################################
# Suffix rules
##############################################################

.SUFFIXES:  .c .o

##############################################################
# Build everything (OpenACC accelerated)
##############################################################

all: lib $(EXAMPLES:.c=)

##############################################################
# Compile object files with OpenACC
##############################################################

.c.o:
	$(CC) -c $(CFLAGS) $<

##############################################################
# Build static library
##############################################################

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f $(ARCH:.c=.o)

##############################################################
# Build example executables (GPU accelerated)
##############################################################

example1: libklt.a
	$(CC) $(CFLAGS) -O3 -o $@ $@.c $(LIB) -lklt $(LIBS)

example2: libklt.a
	$(CC) $(CFLAGS) -O3 -o $@ $@.c $(LIB) -lklt $(LIBS)

example3: libklt.a
	$(CC) $(CFLAGS) -O3 -o $@ $@.c $(LIB) -lklt $(LIBS)

example4: libklt.a
	$(CC) $(CFLAGS) -O3 -o $@ $@.c $(LIB) -lklt $(LIBS)

example5: libklt.a
	$(CC) $(CFLAGS) -O3 -o $@ $@.c $(LIB) -lklt $(LIBS)


##############################################################
# Profiling (same as before)
##############################################################

gprof: CFLAGS=-O1 -g -pg -fno-inline -fno-omit-frame-pointer -Wall -Wfatal-errors $(FLAG1) $(FLAG2)
gprof: LDFLAGS=-pg
gprof: clean lib $(EXAMPLES:.c=)
	@echo "Gprof-enabled executables built. Run one of the examples to generate gmon.out"

gprof-example1: gprof
	./example1
	gprof -b example1 > profile-example1.txt
	./gprof2pdf.sh profile-example1.txt

gprof-example2: gprof
	./example2
	gprof -b example2 > profile-example2.txt
	./gprof2pdf.sh profile-example2.txt

gprof-example3: gprof
	./example3
	gprof -b example3 > profile-example3.txt
	./gprof2pdf.sh profile-example3.txt

gprof-example4: gprof
	./example4
	gprof -b example4 > profile-example4.txt
	./gprof2pdf.sh profile-example4.txt

gprof-example5: gprof
	./example5
	gprof -b example5 > profile-example5.txt
	./gprof2pdf.sh profile-example5.txt

##############################################################
# Utility targets
##############################################################

# Build with multicore CPU (fallback if GPU issues)
multicore:
	$(MAKE) clean
	$(MAKE) ACCFLAGS="$(ACCFLAGS_MULTICORE)" all

# Build with GPU profiling enabled
gpu-profile:
	$(MAKE) clean
	$(MAKE) ACCFLAGS="$(ACCFLAGS_GPU) -Minfo=accel,inline" all
	@echo ""
	@echo "GPU profiling enabled. Run with:"
	@echo "  export PGI_ACC_TIME=1"
	@echo "  ./example3"

# Check GPU availability
check-gpu:
	@echo "=== Checking NVIDIA GPU ==="
	nvidia-smi || echo "ERROR: nvidia-smi not found. Is CUDA installed?"
	@echo ""
	@echo "=== Checking OpenACC Device Info ==="
	pgaccelinfo || nvc -acc -gpu=$(GPU_ARCH) -Minfo=accel test.c 2>&1 | grep -i "accelerator" || echo "Run 'pgaccelinfo' for detailed GPU info"

# Quick test build (example3 only, with verbose output)
test: libklt.a
	@echo "=== Building example3 with GPU acceleration ==="
	$(CC) $(CFLAGS) -Minfo=accel,inline -O3 -o example3 example3.c $(LIB) -lklt $(LIBS)
	@echo ""
	@echo "=== GPU Build Complete ==="
	@echo "Run with: export PGI_ACC_TIME=1 && ./example3"

# Run example3 with GPU profiling
run-test: test
	@echo "=== Running example3 with GPU profiling ==="
	@export PGI_ACC_TIME=1 && ./example3

# Detailed profiling information
info:
	@echo "=== Build Configuration ==="
	@echo "Compiler:      $(CC)"
	@echo "GPU Arch:      $(GPU_ARCH) (RTX 3060)"
	@echo "OpenACC Flags: $(ACCFLAGS)"
	@echo ""
	@echo "=== Available Targets ==="
	@echo "  make all          - Build all examples (GPU accelerated)"
	@echo "  make multicore    - Build with CPU multicore fallback"
	@echo "  make gpu-profile  - Build with detailed profiling"
	@echo "  make check-gpu    - Check GPU availability"
	@echo "  make test         - Quick build & test (example3 only)"
	@echo "  make run-test     - Build & run with profiling"
	@echo "  make clean        - Remove build artifacts"

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile-*.txt profile-*.pdf

.PHONY: clean depend all lib gprof gprof-example1 gprof-example2 gprof-example3 gprof-example4 gprof-example5 \
        multicore gpu-profile check-gpu test run-test info
