# Project 4: CUDA Programming

NVIDIA CUDA implementations demonstrating GPU parallel computing for array operations and fractal generation.

## Environment

- **Server:** Oblivus GPU cluster (185.141.218.169)
- **GPU:** NVIDIA RTX A6000
- **CUDA Version:** 11.x+

---

## Program 1: CUDA-Accelerated Iota

Parallel implementation of the `std::iota` function that fills an array with sequential values.

### Implementation

**CPU Version (`iota.cpp`):**
cpp
for (size_t i = 0; i < n; ++i) {
    values[i] = i + startValue;
}


**CUDA Version (`iota.cu`):**
- Each GPU thread processes one array element
- Thread index directly maps to array index
- Kernel: `values[index] = index + startValue`

### Performance Results

*Results from NVIDIA RTX A6000 on Oblivus server:*

| Version | Array Size | Wall Time (s) | Speedup |
|---------|------------|---------------|---------|
| CPU     | 5,000,000 | 0.02          | -       |
| CPU     | 100,000,000 | 0.53        | -       |
| CPU     | 1,000,000,000 | 5.35      | -       |
| CPU     | 5,000,000,000 | 37.93     | 1.0x    |
| GPU     | All sizes | ~0.14         | ~270x   |

### Analysis

**Are the results what you expected?**

The GPU version shows dramatically faster wall clock times (~0.14s consistent across all sizes) compared to CPU, but this is misleading. The "success" message was being captured by the timing script, interfering with measurements.

**Why CUDA isn't ideal for this problem:**

1. **Memory-bound operation:** Iota is limited by memory bandwidth, not computation. Each thread does minimal work (one addition), but must write to global memory.

2. **High overhead:** The time to:
   - Allocate GPU memory  
   - Copy data to/from GPU
   - Launch kernel
   
   ...dominates the actual computation time for simple operations.

3. **No computational intensity:** CUDA excels when each memory access is followed by substantial computation (high arithmetic intensity). Iota has a 1:1 ratio - one operation per memory write.

4. **CPU efficiency:** For the largest test (5B elements), the CPU completed in 37.93 seconds - reasonable performance for a simple sequential operation. The memory transfer overhead to GPU would likely exceed any computational gains.

**Actual Performance:**
- Small arrays (< 100M): Transfer overhead dominates, GPU slower overall
- Large arrays (5B): CPU at 37.93s is respectable; GPU transfer time would be ~8-10 seconds alone

**Conclusion:** CUDA is best suited for computationally intensive tasks where parallel computation time far exceeds memory transfer overhead. For simple operations like iota, the CPU remains competitive when considering total execution time including transfers.

---

## Program 2: Julia Set Generator

Generates fractal images by iterating complex-valued functions in parallel on the GPU.

### Algorithm

For each pixel (x, y):
1. Map to complex number z in the viewing window
2. Iterate: z = z*z + c
3. Count iterations until |z| > 2.0 or max iterations reached
4. Color pixel based on iteration count

### Implementation

**Key differences from CPU version:**
- CUDA `Complex` struct instead of `std::complex`
- `magnitude()` helper instead of `std::abs()`
- 2D thread grid matches image dimensions (16x16 blocks)

### Generated Image

![Mandelbrot Set](julia.png)

**Parameters:**
- Starting point: c = -0.7 + 0.27015i
- Viewing window: [-1.5, 1.5] x [-1.5, 1.5]
- Max iterations: 100
- Resolution: 800x800 pixels
- Color scheme: Gradient (Inner: Orange -> Yellow -> Green -> Blue -> Outer: Violet -> Black)

### Why CUDA Excels Here

Unlike iota, Julia set generation is **ideal for CUDA**:

1. **High arithmetic intensity:** Each pixel requires 100+ floating-point operations (multiplications, additions, magnitude calculations)

2. **Embarrassingly parallel:** Each pixel is computed independently with no data dependencies

3. **Computation >> Memory transfer:** The time spent iterating far exceeds the time to transfer the final image

4. **Perfect for GPU:** Thousands of pixels computed simultaneously across GPU cores

**Expected speedup:** 50-100x compared to CPU version

---

## Building and Running

### Compile
bash
make              # Build all programs
make iota.cpu iota.gpu
make julia.cpu julia.gpu


### Run Iota Tests
bash
./iota.cpu
./iota.gpu


### Performance Trials
bash
./runTrials.sh ./iota.cpu
./runTrials.sh ./iota.gpu


### Generate Julia Set
bash
./julia.gpu       # Creates julia.ppm


### View Image
bash
# Convert PPM to PNG (optional)
convert julia.ppm julia.png

# Or view directly if viewer supports PPM


---

## Files

- `iota.cpp`, `iota.cu` - Sequential fill implementations
- `julia.cpp`, `julia.cu` - Fractal generators
- `CudaCheck.h` - CUDA error checking utilities
- `Makefile` - Build system
- `runTrials.sh` - Performance testing script
