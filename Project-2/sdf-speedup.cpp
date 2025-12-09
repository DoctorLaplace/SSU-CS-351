#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <string>

// Simple Xorshift RNG for speed
struct Xorshift32 {
    uint32_t state;
    Xorshift32(uint32_t seed) : state(seed) {
        if (state == 0) state = 0xDEADBEEF;
    }
    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return state = x;
    }
    float next_float() {
        return (next() & 0xFFFFFF) / 16777216.0f;
    }
};

inline bool is_outside_sphere(float x, float y, float z) {
    float dx = x - 0.5f;
    float dy = y - 0.5f;
    float dz = z - 0.5f;
    return (dx*dx + dy*dy + dz*dz) > 0.25f;
}

int main(int argc, char* argv[]) {
    size_t numSamples = 2000000;
    size_t numThreads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            numSamples = std::stol(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            numThreads = std::stol(argv[++i]);
        }
    }

    std::vector<std::thread> threads(numThreads);
    std::vector<size_t> counts(numThreads);
    size_t chunkSize = numSamples / numThreads;

    for (size_t id = 0; id < numThreads; ++id) {
        threads[id] = std::thread([&, id]() {
            Xorshift32 rng(12345 + id * 997); // Different seed per thread
            size_t local_count = 0;
            // Handle remainder for last thread? 
            // Simplified: just do chunkSize. Total samples might be slightly less.
            // Or better:
            size_t my_chunk = (id == numThreads - 1) ? (numSamples - id * chunkSize) : chunkSize;
            
            for (size_t i = 0; i < my_chunk; ++i) {
                float x = rng.next_float();
                float y = rng.next_float();
                float z = rng.next_float();
                if (is_outside_sphere(x, y, z)) {
                    local_count++;
                }
            }
            counts[id] = local_count;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    size_t total_count = 0;
    for (size_t c : counts) {
        total_count += c;
    }

    std::cout << static_cast<double>(total_count) / numSamples << "\n";
    return 0;
}
