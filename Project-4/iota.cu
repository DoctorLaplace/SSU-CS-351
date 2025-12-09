// Complete CUDA Iota Implementation
#include <iostream>
#include <cuda_runtime.h>
#include "CudaCheck.h"

using DataType = float;

// CUDA kernel for iota function
__global__ void iotaKernel(size_t n, DataType* values, DataType startValue) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < n) {
        values[index] = index + startValue;
    }
}

// Host function that launches the kernel
void iota(size_t n, DataType* values, DataType startValue) {
    // Allocate device memory
    DataType* d_values;
    CUDA_CHECK_CALL(cudaMalloc(&d_values, n * sizeof(DataType)));
    
    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    iotaKernel<<<blocks, threadsPerBlock>>>(n, d_values, startValue);
    CUDA_CHECK_CALL(cudaGetLastError());
    
    // Copy results back to host
    CUDA_CHECK_CALL(cudaMemcpy(values, d_values, n * sizeof(DataType), 
                          cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK_CALL(cudaFree(d_values));
}

// Main function (provided by project, typically)
int main() {
    const size_t n = 1000000;
    DataType* values = new DataType[n];
    DataType startValue = 0;
    
    iota(n, values, startValue);
    
    // Verify results
    bool correct = true;
    for (size_t i = 0; i < 10 && correct; ++i) {
        if (values[i] != i + startValue) {
            correct = false;
            std::cerr << "Error at index " << i << std::endl;
        }
    }
    
    if (correct) {
        std::cout << "CUDA iota completed successfully" << std::endl;
    }
    
    delete[] values;
    return 0;
}
