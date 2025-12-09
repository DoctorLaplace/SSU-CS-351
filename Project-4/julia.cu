// Complete CUDA Julia Set Implementation
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "CudaCheck.h"

// Complex number class for CUDA
struct Complex {
    float r, i;
    
    __host__ __device__ Complex(float real = 0.0f, float imag = 0.0f) 
        : r(real), i(imag) {}
    
    __device__ float magnitude() const {
        return sqrtf(r * r + i * i);
    }
    
    __device__ Complex operator+(const Complex& other) const {
        return Complex(r + other.r, i + other.i);
    }
    
    __device__ Complex operator*(const Complex& other) const {
        return Complex(r * other.r - i * other.i,
                      r * other.i + i * other.r);
    }
};

// Helper function for magnitude
inline __device__ float magnitude(const Complex& z) { 
    return z.magnitude(); 
}

// Helper to mix colors linearly
__device__ void mixColors(float t, float r1, float g1, float b1, float r2, float g2, float b2, unsigned char* pixel) {
    pixel[0] = static_cast<unsigned char>((r1 + t * (r2 - r1)) * 255.0f);
    pixel[1] = static_cast<unsigned char>((g1 + t * (g2 - g1)) * 255.0f);
    pixel[2] = static_cast<unsigned char>((b1 + t * (b2 - b1)) * 255.0f);
}

// Set pixel color based on iteration count
__device__ void setColor(unsigned char* pixel, int iterations, int maxIterations) {
    if (iterations == maxIterations) {
        // Point is in the set - White
        pixel[0] = 255;
        pixel[1] = 255;
        pixel[2] = 255;
    } else {
        // Normalized iteration count (0.0 to 1.0)
        // t=1.0 is near the set (high iterations)
        // t=0.0 is far from the set (low iterations)
        float t = static_cast<float>(iterations) / maxIterations;
        
        // Desired Gradient:
        // Inner (High t) -> Outer (Low t)
        // Cyan -> Violet -> Black
        
        if (t > 0.5f) {
            // Violet to Cyan
            // t goes from 0.5 to 1.0
            // map to 0.0 to 1.0
            float localT = (t - 0.5f) / 0.5f;
            
            // Violet (0.5, 0, 1.0) -> Cyan (0, 1.0, 1.0)
            mixColors(localT, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, pixel);
        } else {
            // Black to Violet
            // t goes from 0.0 to 0.5
            // map to 0.0 to 1.0
            float localT = t / 0.5f;
            
            // Black (0,0,0) -> Violet (0.5, 0, 1.0)
            mixColors(localT, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 1.0f, pixel);
        }
    }
}

// Julia set kernel
__global__ void juliaKernel(int width, int height, unsigned char* image,
                           Complex c, Complex ll, Complex ur, int maxIterations) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (x >= width || y >= height) return;
    
    // Map pixel to complex plane
    float real = ll.r + (x / static_cast<float>(width)) * (ur.r - ll.r);
    float imag = ll.i + (y / static_cast<float>(height)) * (ur.i - ll.i);
    
    // Initialize z (starting point for Julia iteration)
    Complex z(real, imag);
    
    // Iterate Julia function: z = z² + c
    int iterations = 0;
    while (magnitude(z) < 2.0f && iterations < maxIterations) {
        z = z * z + c;
        iterations++;
    }
    
    // Set pixel color
    int pixelIndex = (y * width + x) * 3;  // RGB = 3 bytes per pixel
    setColor(&image[pixelIndex], iterations, maxIterations);
}

// Host function to generate Julia set
void generateJuliaSet(int width, int height, unsigned char* image,
                     Complex c, Complex ll, Complex ur, int maxIterations) {
    // Allocate device memory
    unsigned char* d_image;
    size_t imageSize = width * height * 3;
    CUDA_CHECK_CALL(cudaMalloc(&d_image, imageSize));
    CUDA_CHECK_CALL(cudaMemset(d_image, 0, imageSize));  // Initialize to zero
    
    // Configure 2D grid
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    
    // Launch kernel
    juliaKernel<<<blocks, threadsPerBlock>>>(width, height, d_image, 
                                            c, ll, ur, maxIterations);
    CUDA_CHECK_CALL(cudaGetLastError());
    CUDA_CHECK_CALL(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK_CALL(cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK_CALL(cudaFree(d_image));
}

// Write PPM image file
void writePPM(const char* filename, int width, int height, unsigned char* image) {
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; i += 3) {
        file << static_cast<int>(image[i]) << " " 
             << static_cast<int>(image[i+1]) << " " 
             << static_cast<int>(image[i+2]) << "\n";
    }
    file.close();
}

// Main function
int main() {
    const int width = 800;
    const int height = 800;
    const int maxIterations = 100;
    
    // Julia set with interesting structure
    Complex c(-0.7f, 0.27015f);
    
    // Viewing window in complex plane
    Complex ll(-1.5f, -1.5f);  // lower left
    Complex ur(1.5f, 1.5f);    // upper right
    
    // Allocate image buffer
    unsigned char* image = new unsigned char[width * height * 3];
    
    // Generate Julia/Mandelbrot set
    std::cout << "Generating Julia set..." << std::endl;
    generateJuliaSet(width, height, image, c, ll, ur, maxIterations);
    
    // Save image
    writePPM("julia.ppm", width, height, image);
    std::cout << "Image saved to julia.ppm" << std::endl;
    
    delete[] image;
    return 0;
}
