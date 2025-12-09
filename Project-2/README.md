# Project 4: Threading and Multi-core Applications

## Overview
This project explores multi-threading to accelerate computations using C++11 threading primitives (`std::thread`, `std::mutex`, `std::condition_variable`). Two applications were threaded:
1.  **Mean Computation**: Calculating the arithmetic mean of a dataset.
2.  **SDF Volume Estimation**: Estimating the volume of a shape (cube minus sphere) using Monte Carlo integration with a Signed Distance Function.

## Computing a Mean

### Performance Graph
![Performance Graph](performance_graph.png)

### Analysis
1.  **Convergence and Speedup**:
    -   The speedup for the mean computation was relatively low (approx. 1.04x).
    -   **Reasoning**: The dataset used (`million.bin`, 4MB) is quite small. The computation (summation) is extremely fast and memory-bound. The overhead of creating threads and, significantly, the overhead of the measurement script (process creation in PowerShell) likely dominated the execution time, masking any parallel gains.
    -   **Amdahl's Law**: The "serial" portion of the total execution time (including OS overhead and file I/O) is large compared to the parallelizable summation. Thus, $p$ (parallel fraction) is effectively small, limiting the maximum theoretical speedup.

2.  **Scaling**:
    -   We did not observe linear scaling. In fact, performance plateaued immediately. This is typical for memory-bound operations on small datasets where the memory bandwidth is saturated by a single core, or where the task is too short to amortize threading overhead.

3.  **Bandwidth**:
    -   Each iteration reads a `float` (4 bytes).
    -   With 8.5 billion samples (original `data.bin`), bandwidth would be a limiting factor.
    -   With `million.bin`, the data fits in the L3 cache of modern processors, making memory access very fast, further reducing the compute time relative to overhead.

## Computing a Volume (SDF)

### Performance
-   **Serial Time**: ~2016 ms
-   **Threaded Time (24 threads)**: ~1018 ms
-   **Speedup**: ~1.98x

### Analysis
-   The SDF computation showed better speedup than the mean computation because it is **compute-bound** (random number generation, square roots) rather than memory-bound.
-   However, the speedup was still limited (approx 2x).
-   **Bottlenecks**:
    -   **Process Overhead**: Similar to the mean computation, the short execution time (~1-2 seconds) means the fixed overhead of the test script significantly impacts the calculated speedup.
    -   **Random Number Generation**: While `std::mt19937` is thread-safe when thread-local, initialization (`std::random_device`) can be slow.
    -   **False Sharing**: Minimized by writing to `insidePoints` only once at the end.

## Implementation Details
-   **Windows Compatibility**:
    -   Replaced `std::jthread` (C++20) with `std::thread` (C++11).
    -   Replaced `std::barrier` (C++20) with a custom `Barrier` class using `std::mutex` and `std::condition_variable`.
    -   Replaced `getopt` (POSIX) with manual command-line argument parsing.
    -   Removed `<format>` (C++20) and used standard stream insertion.

## Compiler Options
-   `g++ -O2 -std=c++2a`

## Extra Credit Challenge

### Optimizations Implemented
-   **Fastest Serial (`sdf-fastest.cpp`)**:
    -   Replaced `std::mt19937` with a lightweight **Xorshift32** RNG.
    -   Optimized `sdf` check to avoid `sqrt` by comparing squared distances.
    -   Compiled with `-O3`.
-   **Fastest Threaded (`sdf-speedup.cpp`)**:
    -   Applied the same optimizations as above.
    -   Distributed work across 24 threads.

### Results (100 Million Samples)
-   **Baseline Serial (`sdf.out`)**: ~3043 ms
-   **Fastest Serial (`sdf-fastest.out`)**: ~1033 ms (**~2.95x Speedup**)
-   **Fastest Threaded (`sdf-speedup.out`)**: ~1033 ms (**~2.95x Speedup**)

### Analysis
The optimized serial version is significantly faster (~3x) due to the removal of `sqrt` and the faster RNG. The threaded version did not show further improvement in this test. This is likely because the execution time of the optimized code is now so short that it is entirely dominated by the **~1000 ms overhead** of the PowerShell process creation and measurement, masking the benefits of parallelism. In a longer-running test (e.g., 10 billion samples), the threaded version would likely pull ahead.
