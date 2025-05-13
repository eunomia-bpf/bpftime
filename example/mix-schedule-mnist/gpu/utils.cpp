#include "utils.hpp"

std::vector<std::vector<float>> load_mnist_images(const std::string& path, int& num_images, int& img_rows, int& img_cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open image file: " + path);
    }
    int32_t magic_number = read_int_big_endian(file);
    if (magic_number != MNIST_IMAGE_MAGIC) throw std::runtime_error("Invalid MNIST image file magic number.");
    num_images = read_int_big_endian(file);
    img_rows = read_int_big_endian(file);
    img_cols = read_int_big_endian(file);
    if (img_rows != MNIST_IMAGE_HEIGHT || img_cols != MNIST_IMAGE_WIDTH) throw std::runtime_error("MNIST image dimensions mismatch.");

    std::vector<std::vector<float>> images(num_images, std::vector<float>(img_rows * img_cols));
    const float mean = 0.1307f;
    const float std_dev = 0.3081f; // 与PyTorch的Normalize一致
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < img_rows * img_cols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = (static_cast<float>(pixel) / 255.0f - mean) / std_dev;
        }
    }
    file.close();
    return images;
}

std::vector<unsigned char> load_mnist_labels(const std::string& path, int& num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open label file: " + path);
    }
    int32_t magic_number = read_int_big_endian(file);
    if (magic_number != MNIST_LABEL_MAGIC) throw std::runtime_error("Invalid MNIST label file magic number.");
    num_labels = read_int_big_endian(file);
    std::vector<unsigned char> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    file.close();
    return labels;
}

std::vector<float> load_weights_from_bin(const std::string& path, size_t expected_elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size != expected_elements * sizeof(float)) {
        throw std::runtime_error("Weight file size mismatch for " + path + 
                                 ". Expected " + std::to_string(expected_elements * sizeof(float)) + 
                                 " bytes, got " + std::to_string(size) + " bytes.");
    }
    
    std::vector<float> weights(expected_elements);
    if (!file.read(reinterpret_cast<char*>(weights.data()), size)) {
        throw std::runtime_error("Error reading weights from file: " + path);
    }
    file.close();
    return weights;
}
