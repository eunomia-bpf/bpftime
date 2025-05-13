#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm> // For std::reverse

// MNIST dataset constants
const int MNIST_IMAGE_MAGIC = 2051;
const int MNIST_LABEL_MAGIC = 2049;
const int MNIST_IMAGE_WIDTH = 28;
const int MNIST_IMAGE_HEIGHT = 28;
const int MNIST_IMAGE_SIZE = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT; // 784
const int MNIST_NUM_CLASSES = 10;

// MLP Layer dimensions (必须与PyTorch模型匹配)
const int MLP_INPUT_SIZE = 784;
const int MLP_HIDDEN_SIZE = 128;
const int MLP_OUTPUT_SIZE = 10;

// Helper to convert big-endian to little-endian
inline int32_t read_int_big_endian(std::ifstream& ifs) {
    int32_t val;
    ifs.read(reinterpret_cast<char*>(&val), sizeof(val));
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    char* bytes = reinterpret_cast<char*>(&val);
    std::reverse(bytes, bytes + sizeof(int32_t));
    #endif
    return val;
}

std::vector<std::vector<float>> load_mnist_images(const std::string& path, int& num_images, int& img_rows, int& img_cols);
std::vector<unsigned char> load_mnist_labels(const std::string& path, int& num_labels);
std::vector<float> load_weights_from_bin(const std::string& path, size_t expected_elements);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // UTILS_HPP
