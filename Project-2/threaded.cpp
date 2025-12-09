//#include <barrier>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>

// Header file for the Data template class
#include "Data.h"

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
    //   * the default filename
    //   * the default number of threads
    //
    std::string filename = "data.bin";
    size_t numThreads = 4;

    //-----------------------------------------------------------------------
    //
    // Process command-line options.
    //
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h") {
            const char* help =
                "Usage: %s -[fht]\n"
                "    -h           show help message\n"
                "    -f <name>    read data from <name>\n"
                "    -t <value>   use <values> number of threads (default: %u)\n";

            fprintf(stderr, help, argv[0], numThreads);
            exit(EXIT_SUCCESS);
        } else if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            numThreads = std::stol(argv[++i]);
        }
    }

    //-----------------------------------------------------------------------
    //
    // Access our the data file through our Data C++ class.  Under the hood,
    //   this class uses an advanced file-access technique called memory
    //   mapping, which makes the file looked like an array (although our
    //   Data class makes it look more like a std::vector), allowing indexed
    //   random-access to the data.
    //
    Data<float>  data(filename.c_str());

    //-----------------------------------------------------------------------
    //
    // A collection of variables to make threading the application simpler.
    //
    //   * threads - is merely an array of std::jthreads to store the created
    //       threads, similar to what was demonstrated in class
    //   * sums - provides a per-thread sum allowing you to accumulate the
    //       values computed in a thread independent of other threads
    //   * barrier - provides a synchronization barrier to prevent threads
    //       exiting before their peers
    //
    std::vector<std::thread>  threads(numThreads);
    std::vector<double>        sums(numThreads);
    Barrier                    barrier(numThreads);

    // Computation of how much work a thread should do.
    size_t chunkSize = (data.size() / numThreads) + 1;

    //-----------------------------------------------------------------------
    //
    // The computational kernel, where you should enter your thread
    //   implementation.  Much of the thread framework is provided below,
    //   and you mostly needed to compute array bounds to be processed in
    //   the threads, the computational loop (see mean.cpp), and lambda's
    //   closure configuration.
    //
    for (size_t id = 0; id < threads.size(); ++id) {
        threads[id] = std::thread(
            [&, id]() {
                size_t start = id * chunkSize;
                size_t end = std::min(start + chunkSize, data.size());

                double local_sum = 0.0;
                for (size_t i = start; i < end; ++i) {
                    local_sum += data[i];
                }
                sums[id] = local_sum;

                barrier.arrive_and_wait();
            }
        );
    }

    //-----------------------------------------------------------------------
    //
    // The main thread's final work.  As discussed in class, we wait on
    //   the last thread explicitly (because we're using std::jthreads
    //   recall that we've already joined them at their creation), but this
    //   line pauses the main thread until the last thread terminates
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    //-----------------------------------------------------------------------
    //
    // Compute the final sum by tallying the values from each thread, and
    //   report the results
    //
    double sum = std::accumulate(std::begin(sums), std::end(sums), 0.0);

    std::cout << "Samples = " << data.size() << "\n";
    std::cout << "Mean = " << sum / data.size() << "\n";
}
