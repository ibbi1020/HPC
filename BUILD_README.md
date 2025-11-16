# Quick Build Guide - Windows + RTX 3060

## Prerequisites

1. **NVIDIA HPC SDK** (for OpenACC support)
   - Download: https://developer.nvidia.com/hpc-sdk-downloads
   - Install to default location
   - Add to PATH: `C:\Program Files\NVIDIA\HPC_SDK\Windows_x86_64\24.7\compilers\bin`

2. **NVIDIA GPU Driver**
   - Already installed with CUDA
   - Verify: `nvidia-smi`

3. **Make Utility** (optional, recommended)
   - MinGW: https://www.mingw-w64.org/
   - Or use WSL2
   - Or use batch scripts provided

## Quick Start (3 Steps)

### Using Batch Scripts (Easiest)

```cmd
1. build.bat        # Builds everything
2. run_example3.bat # Runs with profiling
```

### Using Make

```bash
make clean
make all            # Build with GPU acceleration
make run-test       # Build & run with profiling
```

### Manual Commands

```powershell
# Build library
nvc -c -acc=gpu -gpu=cc86,managed -Minfo=accel -fast -DNDEBUG *.c
ar ruv libklt.a *.o

# Build example3
nvc -acc=gpu -gpu=cc86,managed -Minfo=accel -fast -O3 -o example3.exe example3.c -L. -lklt -lm

# Run with profiling
$env:PGI_ACC_TIME = "1"
./example3.exe
```

## Makefile Configuration

The Makefile is now configured for your **RTX 3060**:

- **GPU Architecture**: cc86 (Compute Capability 8.6)
- **OpenACC Target**: GPU (not multicore CPU)
- **CUDA Unified Memory**: Enabled (managed)
- **Optimizations**: Aggressive (-fast -O3)

### Available Make Targets

| Command | Purpose |
|---------|---------|
| `make all` | Build all examples (GPU) |
| `make test` | Quick test (example3 only) |
| `make run-test` | Build & run with profiling |
| `make check-gpu` | Verify GPU detected |
| `make multicore` | CPU fallback (no GPU) |
| `make clean` | Remove build files |
| `make info` | Show configuration |

## Verifying GPU Acceleration

### 1. Check Compiler Output

Look for:
```
convolve.c:
    170, Generating NVIDIA GPU code
        171, #pragma acc loop gang, vector(128) collapse(2)
```

### 2. Check Runtime Output

```powershell
$env:PGI_ACC_TIME = "1"
./example3.exe
```

Should show:
```
Accelerator Kernel Timing data
_convolveImageHoriz NVIDIA devicenum=0
    time(us): 1,234
    170: kernel launched 10 times
        grid: [40x30]  block: [16x16]
```

### 3. Monitor GPU Usage

```powershell
# In another terminal
nvidia-smi dmon -i 0 -s pucvmet -d 1
```

Should show GPU utilization >0% during execution.

## Troubleshooting

### "nvc: command not found"

**Fix**:
```powershell
$env:Path += ";C:\Program Files\NVIDIA\HPC_SDK\Windows_x86_64\24.7\compilers\bin"
```

### "make: command not found"

**Fix**: Use `build.bat` instead of make

### No GPU acceleration

**Check**:
1. GPU detected: `nvidia-smi`
2. Compiler flags: Should include `-acc=gpu -gpu=cc86`
3. Runtime: `$env:PGI_ACC_TIME = "1"` should show kernel info

## Expected Performance

With **RTX 3060** and Phase 1&2 optimizations:

| Workload | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| 150 features × 150 frames | ~5.0s | ~1.5-2.0s | 3-5× |

## Files Created

- ✅ `Makefile` - Updated for RTX 3060 GPU
- ✅ `build.bat` - Windows build script
- ✅ `run_example3.bat` - Test script with profiling
- ✅ `BUILD_INSTRUCTIONS_RTX3060.md` - Detailed guide

## Next Steps

1. Run `build.bat` to compile
2. Run `run_example3.bat` to test
3. Check profiling output
4. Verify ~3-5× speedup over CPU
5. Proceed to Phase 3 optimization!

---

**Need Help?** Check `BUILD_INSTRUCTIONS_RTX3060.md` for detailed troubleshooting.
