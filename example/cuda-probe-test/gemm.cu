#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

// constant memory for dimensions
__constant__ int d_N;
__constant__ int d_numRows;

#define TILE_WIDTH 32

// --------------------------------------------------
// 1) Dense matrix multiplication (1024×1024) kernel
// --------------------------------------------------
__global__ void matMulTiled(const float *A,
                            const float *B,
                            float *C)
{
    // global row & col
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    // number of phases = N / TILE_WIDTH
    int phases = d_N / TILE_WIDTH;

    // shared-memory tiles
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    for (int ph = 0; ph < phases; ++ph) {
        // load one TILE_WIDTH×TILE_WIDTH sub-block of A and B
        int aIdx = row * d_N + ph * TILE_WIDTH + threadIdx.x;
        int bIdx = (ph * TILE_WIDTH + threadIdx.y) * d_N + col;
        sA[threadIdx.y][threadIdx.x] = A[aIdx];
        sB[threadIdx.y][threadIdx.x] = B[bIdx];
        __syncthreads();

        // multiply-accumulate within the tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // write back
    if (row < d_N && col < d_N) {
        C[row * d_N + col] = acc;
    }
}

// --------------------------------------------------
// 2) SpMV in CSR format on a 10 M-edge graph
// --------------------------------------------------
__global__ void spmvCSR(const int *rowPtr,
                        const int *colInd,
                        const float *vals,
                        const float *x,
                        float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_numRows) return;

    float sum = 0.0f;
    int start = rowPtr[row];
    int end   = rowPtr[row + 1];
    for (int idx = start; idx < end; ++idx) {
        sum += vals[idx] * x[colInd[idx]];
    }
    y[row] = sum;
}

int main()
{
    // ---- 1) Dense mat-mul setup ----
    const int h_N = 1024;
    cudaMemcpyToSymbol(d_N, &h_N, sizeof(h_N));

    size_t matBytes = size_t(h_N) * h_N * sizeof(float);
    std::vector<float> h_A(h_N*h_N), h_B(h_N*h_N), h_C(h_N*h_N);

    // init A and B
    for (int i = 0; i < h_N*h_N; ++i) {
        h_A[i] = 1.0f;  // e.g. all ones
        h_B[i] = 2.0f;  // e.g. all twos
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matBytes);
    cudaMalloc(&d_B, matBytes);
    cudaMalloc(&d_C, matBytes);

    cudaMemcpy(d_A, h_A.data(), matBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), matBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((h_N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (h_N + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, matBytes, cudaMemcpyDeviceToHost);
    std::cout << "C[0,0] = " << h_C[0] << "  (expected "
              << float(h_N) * 2.0f << ")\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}