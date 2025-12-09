#include <iostream>
#include <fstream>
#include <vector>

int main() {
    std::vector<float> data;
    for (int i = 1; i <= 50; ++i) {
        data.push_back(static_cast<float>(i));
    }

    std::ofstream out("tiny.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close();

    std::cout << "Generated tiny.bin with " << data.size() << " values." << std::endl;
    return 0;
}
