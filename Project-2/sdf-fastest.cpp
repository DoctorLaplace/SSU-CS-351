#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

// Simple Xorshift RNG for speed
struct Xorshift32 {
    uint32_t state;
    Xorshift32(uint32_t seed) : state(seed) {
        if (state == 0) state = 0xDEADBEEF; // Handle 0 seed
    }
    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return state = x;
    }
    // Return float in [0, 1)
    float next_float() {
        return (next() & 0xFFFFFF) / 16777216.0f;
    }
};

struct vec3 {
    float x, y, z;
};

// Optimized SDF check avoiding sqrt and function call overhead
// Sphere center (0.5, 0.5, 0.5), radius 0.5
// Point p is in [0, 1]
// Check if distance > radius
// dist^2 > radius^2
// (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2 > 0.25
inline bool is_outside_sphere(float x, float y, float z) {
    float dx = x - 0.5f;
    float dy = y - 0.5f;
    float dz = z - 0.5f;
    return (dx*dx + dy*dy + dz*dz) > 0.25f;
}

int main(int argc, char* argv[]) {
    size_t numSamples = 2000000;
    
    // Manual arg parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            numSamples = std::stol(argv[++i]);
        }
    }

    Xorshift32 rng(12345);
    size_t count = 0;

    // Main loop
    for (size_t i = 0; i < numSamples; ++i) {
        float x = rng.next_float();
        float y = rng.next_float();
        float z = rng.next_float();
        
        if (is_outside_sphere(x, y, z)) {
            count++;
        }
    }

    std::cout << static_cast<double>(count) / numSamples << "\n";
    return 0;
}
