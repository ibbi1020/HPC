##############################################################
# OpenACC–Enabled KLT Makefile (nvc / nvc++)
# Configured for NVIDIA GeForce RTX 3060 (Ampere architecture)
# WSL2 Compatible
##############################################################

# Use NVIDIA HPC SDK compilers (NOT nvcc)
CC  = nvc
CXX = nvc++

# GPU Architecture for RTX 3060 (Compute Capability 8.6 - Ampere)
GPU_ARCH = cc86

# OpenACC GPU Flags for RTX 3060
# -acc=gpu: Target GPU instead of multicore CPU
# -gpu=$(GPU_ARCH): Specify compute capability
# -gpu=managed: Use CUDA Unified Memory (easier debugging)
# -Minfo=accel: Show accelerator kernel information
# -fast: Enable aggressive optimizations
ACCFLAGS = -acc=gpu -gpu=$(GPU_ARCH),managed -Minfo=accel -fast

# Compiler flags
FLAG1 = -DNDEBUG
CFLAGS   = $(FLAG1) $(FLAG2) $(ACCFLAGS)
CXXFLAGS = $(FLAG1) $(FLAG2) $(ACCFLAGS)

# WSL2 CUDA Library Path Fix (for libcuda.so)
# This prevents "libcuda.so not found" errors
WSL_CUDA_LIB = /usr/lib/wsl/lib

# Library paths and libraries
LIB  = -L$(WSL_CUDA_LIB) -L/usr/local/lib -L/usr/lib -L.
LIBS = -lm

# Add WSL CUDA library to runtime linker path
export LD_LIBRARY_PATH := $(WSL_CUDA_LIB):$(LD_LIBRARY_PATH)

# Object files needed for example3
OBJS = klt.o convolve.o klt_util.o pnmio.o error.o pyramid.o \
       storeFeatures.o selectGoodFeatures.o trackFeatures.o

##############################################################
# Build Targets
##############################################################

# Default: build example3 with OpenACC GPU acceleration
all: example3

# Build example3 with all object files directly
example3: $(OBJS) example3.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIB) $(LIBS)
	@echo ""
	@echo "✅ Built example3 successfully!"
	@echo "Run with: make run"

# Compile C source files
.c.o:
	$(CC) -c $(CFLAGS) $<

# Convenient run target with GPU profiling
run: example3
	@echo "🚀 Running example3 with GPU profiling..."
	@PGI_ACC_TIME=1 ./example3

# Clean build artifacts
clean:
	rm -f *.o example3 core
	@echo "🧹 Cleaned build artifacts"

# Test: verify build and show GPU code generation info
test: clean example3
	@echo ""
	@echo "✅ Build successful!"
	@echo "Look for 'Generating NVIDIA GPU code' messages above"
	@echo ""
	@echo "Run with: make run"

# Help target
help:
	@echo "📖 KLT OpenACC Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build example3 (default)"
	@echo "  make run      - Run example3 with GPU profiling"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Clean build and verify"
	@echo "  make help     - Show this message"
	@echo ""
	@echo "GPU: NVIDIA GeForce RTX 3060 ($(GPU_ARCH))"
	@echo "Compiler: $(CC)"

.PHONY: all clean test run help

