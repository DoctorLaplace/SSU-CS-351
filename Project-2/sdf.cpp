
//#include <barrier>
#include <cmath>
//#include <format>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <algorithm>

#include "Shapes.h"

/////////////////////////////////////////////////////////////////////////////
//
// --- sdf() ---
//
// A signed-distance function, which when provided with a 3D point (in this
//   case), it will return a boolean value indicating if the value is inside
//   the region of interest.  In this case, passed parameter (p) is expected
//   to be a point within the unit cube (i.e., (x, y, z) are all values 
//   in [0.0, 1.0]).
//
// p is transformed into the sphere's coordinate system (by subtracting off
//   the sphere's center), and then p's distance from the origin is computed
//   and compared to the sphere's radius.  As we're looking to compute the
//   volume of the cube with the sphere removed, we're interested to know if
//   the point is inside or outside of the sphere.
//

bool sdf(vec3 p) {
    // Create a sphere centered inside of our cube
    static const Sphere sphere(vec3(0.5), 0.5);
    
    // Determine if the point is outside the sphere.  It's guaranteed
    //   to be inside the unit cube because of how the points are
    //   generated.
    p -= sphere.center;
    return p.length() > sphere.radius;
}

class Barrier {
public:
    explicit Barrier(std::ptrdiff_t count) : threshold(count), count(count), generation(0) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        auto gen = generation;
        if (--count == 0) {
            generation++;
            count = threshold;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
private:
    std::mutex mutex;
    std::condition_variable cv;
    std::ptrdiff_t threshold;
    std::ptrdiff_t count;
    std::ptrdiff_t generation;
};

/////////////////////////////////////////////////////////////////////////////
//
// --- main ---
//

int main(int argc, char* argv[]) {
    //
    // Specify the program's default options:
    //   * numSamples - the default number of points to test with
    //   * partitions - how many pieces the generator’s output interval
    //       is split into
    //   * numThreads - the default number of threads spawnedd
    //
    size_t numSamples = 2'000'000;
    size_t partitions = 1'000'000;
    size_t numThreads = 4;

    //-----------------------------------------------------------------------
    //
    // Process command-line options.
    //
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h") {
            const char* help =
                "Usage: %s -[hpnt]\n"
                "    -h           show help message\n"
                "    -p <value>   paritions for uniform number generator (default :%u)\n"
                "    -n <value>   total number of sample points (default :%u)\n"
                "    -t <value>   use <values> number of threads (default: %u)\n";

            fprintf(stderr, help, argv[0], partitions, numSamples, numThreads);
            exit(EXIT_SUCCESS);
        } else if (arg == "-p" && i + 1 < argc) {
            partitions = std::stol(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            numSamples = std::stol(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            numThreads = std::stol(argv[++i]);
        }
    }

    //-----------------------------------------------------------------------
    //
    // A collection of variables to make threading the application simpler.
    //
    //   * threads - is merely an array of std::jthreads to store the created
    //       threads, similar to what was demonstrated in class
    //   * insidePoints - provides a per-thread variable allowing you to 
    //       accumulate the values computed in a thread independent of other
    //       threads
    //   * barrier - provides a synchronization barrier to prevent threads
    //       exiting before their peers
    //
    std::vector<std::thread>   threads(numThreads);
    std::vector<size_t>        insidePoints(numThreads);
    Barrier                    barrier(numThreads);

    // Computation of how much work a thread should do. (yes, this doesn't
    //   divide evenly in most cases, but we'll lose at most numThreads
    //   samples, which isn't a big deal if the numSamples is sufficiently
    //   large)
    size_t chunkSize = numSamples / numThreads;

    //-----------------------------------------------------------------------
    //
    // The computational kernel, where you should enter your thread
    //   implementation.  Much of the thread framework is provided below,
    //   and you mostly need to:
    //     * set up the lambda's closure arguments
    //     * implement the kernel's core iteration loop
    //     * collect the results of testing point p's position using the
    //         sdf() function.  (Hint: you can simply tally up the results
    //         returned by sdf() to count how many samples are in the
    //         region of interest).  Something like 
    //
    //           tallyVariable += sdf(p);
    //
    //         will do nicely.
    //
    //   (really, you'll need to add four lines to this, one of which is a
    //      closing brace :-)   
    //
    for (size_t id = 0; id < threads.size(); ++id) {
        threads[id] = std::thread(
            [&, id]() {

            // C++ 11's random number generation system.  These functions
            //   will generate uniformly distributed unsigned integers in
            //   the range [0, partitions].  The functions are used in the
            //   helper function rand() (implemented as a lambda)
            std::random_device device;
            std::mt19937 generator(device());
            std::uniform_int_distribution<unsigned int> uniform(0, partitions);


                // Define a helper function to generate random floating-point
                //   values in the range [0.0, 1.0]
                auto rand = [&]() {
                    return static_cast<double>(uniform(generator)) / partitions;
                };
            
                // Generate points inside the volume cube.  First, create uniformly
                //   distributed points in the range [0.0, 1.0] for each dimension.
                size_t count = 0;
                for (size_t i = 0; i < chunkSize; ++i) {
                    vec3 p(rand(), rand(), rand());
                    if (sdf(p)) {
                        count++;
                    }
                }
                insidePoints[id] = count;

                barrier.arrive_and_wait();
            }
        );
    }

    // Add in the last necessary parts for our threaded programs.  These
    //   may include summing up the individual threads' computations, or
    //   having the main thread wait on a thread to keep it from exiting
    //
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    size_t totalInside = 0;
    for (size_t c : insidePoints) {
        totalInside += c;
    }

    std::cout << "Volume = " << static_cast<double>(totalInside) / numSamples << "\n";
}

