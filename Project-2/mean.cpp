
#include <iostream>

// Header file for the Data template class
#include "Data.h"

int main(int argc, char* argv[]) {
    try {
        std::cout << "Starting mean.cpp..." << std::endl;
        const char* filename = argc < 2 ? "data.bin" : argv[1];

        std::cout << "Opening file: " << filename << std::endl;
        Data<float>  data(filename);
        std::cout << "File opened. Size: " << data.size() << std::endl;

        //-----------------------------------------------------------------------
        //
        // The computational kernel that computes the mean by summing the
        //   values in the data array. 
        double sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        std::cout << "Sum computed: " << sum << std::endl;

        //-----------------------------------------------------------------------
        //
        // Report the results.
        //
        std::cout << "Samples = " << data.size() << "\n";
        std::cout << "Mean = " << sum / data.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught." << std::endl;
        return 1;
    }
    return 0;
}

