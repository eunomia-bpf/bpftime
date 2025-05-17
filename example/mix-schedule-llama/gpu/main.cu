// cuda_inference/src/main.cu

#include "utils.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

// --- CUDA Kernels ---
// MODIFIED FOR SINGLE THREAD SERIAL EXECUTION:
// When launched with grid(1,1,1) and block(1,1,1), this single thread will perform all calculations.
__global__ void matrix_multiply_add_bias_relu_kernel_SERIAL(
    const float* A_input,
    const float* W_weights,
    const float* B_bias,
    float* C_output,
    int batch_size_param, int in_features_param, int out_features_param,
    bool apply_relu)
{
    // This kernel is intended to be launched with <<<1, 1>>>
    // The single thread (0,0,0) will execute the loops.
    for (int r = 0; r < batch_size_param; ++r) {
        for (int c = 0; c < out_features_param; ++c) {
            float sum = 0.0f;
            for (int k_idx = 0; k_idx < in_features_param; ++k_idx) {
                sum += A_input[r * in_features_param + k_idx] * W_weights[c * in_features_param + k_idx];
            }
            sum += B_bias[c];

            if (apply_relu) {
                C_output[r * out_features_param + c] = fmaxf(0.0f, sum);
            } else {
                C_output[r * out_features_param + c] = sum;
            }
        }
    }
}


void forward_mlp_cuda_inference_SERIAL(
    float* d_input_images,
    float* d_W1, float* d_b1, float* d_W2, float* d_b2,
    float* d_hidden_layer_output,
    float* d_final_logits,
    int current_batch_size
) {
    // --- SERIAL EXECUTION CONFIGURATION: Grid(1,1,1), Block(1,1,1) ---
    dim3 serial_grid_dim(1, 1, 1);
    dim3 serial_block_dim(1, 1, 1);

    matrix_multiply_add_bias_relu_kernel_SERIAL<<<serial_grid_dim, serial_block_dim>>>(
        d_input_images, d_W1, d_b1, d_hidden_layer_output,
        current_batch_size, MLP_INPUT_SIZE, MLP_HIDDEN_SIZE,
        true
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    matrix_multiply_add_bias_relu_kernel_SERIAL<<<serial_grid_dim, serial_block_dim>>>(
        d_hidden_layer_output, d_W2, d_b2, d_final_logits,
        current_batch_size, MLP_HIDDEN_SIZE, MLP_OUTPUT_SIZE,
        false
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


int main() {
    std::cout << "--- CUDA C++ MNIST MLP Inference (SERIAL EXECUTION MODE) ---" << std::endl;
    std::cout << "--- WARNING: This mode is very slow as it does not use GPU parallelism. ---" << std::endl;

    // --- Load Test Data (Host) ---
    std::cout << "Loading MNIST test data..." << std::endl;
    int num_total_test_images, img_rows, img_cols; // Renamed to avoid confusion
    std::string test_images_path = "../train/data/MNIST/raw/t10k-images-idx3-ubyte";
    std::string test_labels_path = "../train/data/MNIST/raw/t10k-labels-idx1-ubyte";
    
    std::vector<std::vector<float>> h_test_images_vec;
    std::vector<unsigned char> h_test_labels_vec;
    try {
        h_test_images_vec = load_mnist_images(test_images_path, num_total_test_images, img_rows, img_cols);
        int num_labels_check;
        h_test_labels_vec = load_mnist_labels(test_labels_path, num_labels_check);
        if (num_total_test_images != num_labels_check) {
            throw std::runtime_error("Number of test images and labels mismatch.");
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading MNIST data: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Loaded " << num_total_test_images << " total test images." << std::endl;

    // Determine the number of images to actually test
    const int MAX_IMAGES_TO_TEST = 2000;
    int num_images_to_process = std::min(num_total_test_images, MAX_IMAGES_TO_TEST);
    std::cout << "INFO: Will process a maximum of " << MAX_IMAGES_TO_TEST << " images. Actual images to process: " << num_images_to_process << std::endl;


    std::vector<float> h_test_images_flat(num_total_test_images * MLP_INPUT_SIZE); // Still load all for simplicity, but only use subset
    for(int i=0; i < num_total_test_images; ++i) {
        std::copy(h_test_images_vec[i].begin(), h_test_images_vec[i].end(), 
                  h_test_images_flat.begin() + i * MLP_INPUT_SIZE);
    }

    // --- Load Weights (Host) ---
    std::cout << "Loading MLP weights..." << std::endl;
    std::vector<float> h_W1, h_b1, h_W2, h_b2;
    try {
        std::string weights_dir = "../train/mlp_weights_for_cpp/"; 
        h_W1 = load_weights_from_bin(weights_dir + "fc1_weights.bin", MLP_HIDDEN_SIZE * MLP_INPUT_SIZE);
        h_b1 = load_weights_from_bin(weights_dir + "fc1_bias.bin", MLP_HIDDEN_SIZE);
        h_W2 = load_weights_from_bin(weights_dir + "fc2_weights.bin", MLP_OUTPUT_SIZE * MLP_HIDDEN_SIZE);
        h_b2 = load_weights_from_bin(weights_dir + "fc2_bias.bin", MLP_OUTPUT_SIZE);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Weights loaded." << std::endl;
    
    // --- Allocate GPU Memory ---
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_batch_input_images, *d_hidden_output, *d_final_logits_batch;

    CUDA_CHECK(cudaMalloc(&d_W1, h_W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, h_b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, h_W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, h_b2.size() * sizeof(float)));

    const int SERIAL_PROCESSING_BATCH_SIZE = 1;
    std::cout << "INFO: SERIAL_PROCESSING_BATCH_SIZE set to " << SERIAL_PROCESSING_BATCH_SIZE << std::endl;

    CUDA_CHECK(cudaMalloc(&d_batch_input_images, SERIAL_PROCESSING_BATCH_SIZE * MLP_INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden_output, SERIAL_PROCESSING_BATCH_SIZE * MLP_HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_logits_batch, SERIAL_PROCESSING_BATCH_SIZE * MLP_OUTPUT_SIZE * sizeof(float)));

    // --- Copy Weights to GPU ---
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- Perform Inference Serially ---
    std::cout << "Performing serial inference on GPU (will be slow)..." << std::endl;
    int correct_predictions = 0;
    std::vector<float> h_batch_logits(SERIAL_PROCESSING_BATCH_SIZE * MLP_OUTPUT_SIZE);

    // Loop up to num_images_to_process
    for (int i = 0; i < num_images_to_process; i += SERIAL_PROCESSING_BATCH_SIZE) {
        int current_batch_size = std::min(SERIAL_PROCESSING_BATCH_SIZE, num_images_to_process - i);
        if (current_batch_size <= 0) break;

        if (i % (SERIAL_PROCESSING_BATCH_SIZE * 100) == 0 || i + current_batch_size >= num_images_to_process) {
             printf("INFO: Processing images %d to %d (out of %d to be processed)\n", i, i + current_batch_size -1, num_images_to_process);
        }

        CUDA_CHECK(cudaMemcpy(d_batch_input_images, 
                              h_test_images_flat.data() + i * MLP_INPUT_SIZE,
                              current_batch_size * MLP_INPUT_SIZE * sizeof(float), 
                              cudaMemcpyHostToDevice));

        forward_mlp_cuda_inference_SERIAL(d_batch_input_images,
                                          d_W1, d_b1, d_W2, d_b2,
                                          d_hidden_output, d_final_logits_batch,
                                          current_batch_size);

        CUDA_CHECK(cudaMemcpy(h_batch_logits.data(), d_final_logits_batch,
                              current_batch_size * MLP_OUTPUT_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        for (int j = 0; j < current_batch_size; ++j) {
            float* current_image_logits = h_batch_logits.data() + j * MLP_OUTPUT_SIZE;
            int predicted_class = 0;
            float max_logit = current_image_logits[0];
            for (int k = 1; k < MLP_OUTPUT_SIZE; ++k) {
                if (current_image_logits[k] > max_logit) {
                    max_logit = current_image_logits[k];
                    predicted_class = k;
                }
            }
            if (predicted_class == static_cast<int>(h_test_labels_vec[i + j])) {
                correct_predictions++;
            }
        }
    }

    if (num_images_to_process > 0) {
        float accuracy = static_cast<float>(correct_predictions) / num_images_to_process * 100.0f;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nCUDA C++ SERIAL Inference Accuracy on " << num_images_to_process << " test images: " << accuracy << "% ("
                  << correct_predictions << "/" << num_images_to_process << ")" << std::endl;
    } else {
        std::cout << "\nNo images were processed." << std::endl;
    }

    // --- Free GPU Memory ---
    CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_W2)); CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_batch_input_images));
    CUDA_CHECK(cudaFree(d_hidden_output));
    CUDA_CHECK(cudaFree(d_final_logits_batch));

    std::cout << "\n--- CUDA C++ serial inference script finished ---" << std::endl;
    return 0;
}
