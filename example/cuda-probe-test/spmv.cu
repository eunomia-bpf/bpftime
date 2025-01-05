#include <cstdint>
#include <cstdio>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

// matrix size
__constant__ int d_N;
// graph rows
__constant__ int d_numRows;

#define TILE_WIDTH 32

// --------------------------------------------------
// 1) Tile 化的 1024×1024 整数矩阵乘法 (uint64_t)
//    在加载和存储时使用 ld.global.u64 / st.global.u64
// --------------------------------------------------
__global__ void matMulTiled_u64(const uint64_t *A,
                                const uint64_t *B,
                                uint64_t *C)
{
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    uint64_t acc = 0;

    __shared__ uint64_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ uint64_t sB[TILE_WIDTH][TILE_WIDTH];

    int phases = d_N / TILE_WIDTH;
    for (int ph = 0; ph < phases; ++ph) {
        // 从全局内存加载一个元素到寄存器，再写入 shared memory
        uint64_t a_val, b_val;
        const uint64_t *addrA = A + row * d_N + ph * TILE_WIDTH + threadIdx.x;
        const uint64_t *addrB = B + (ph * TILE_WIDTH + threadIdx.y) * d_N + col;
        asm volatile(
            "ld.global.u64 %0, [%1];\n\t"
            "ld.global.u64 %2, [%3];\n\t"
            : "=l"(a_val), "=l"(b_val)
            : "l"(addrA), "l"(addrB)
        );
        sA[threadIdx.y][threadIdx.x] = a_val;
        sB[threadIdx.y][threadIdx.x] = b_val;
        __syncthreads();

        // 瓶内相乘累加
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 最后将结果通过 st.global.u64 存回全局内存
    uint64_t *addrC = C + row * d_N + col;
    asm volatile(
        "st.global.u64 [%0], %1;\n\t"
        :
        : "l"(addrC), "l"(acc)
    );
}

// --------------------------------------------------
// 2) CSR 格式的 SpMV，10M 边的图，uint64_t 数据
//    同样使用 ld.global.u64 / st.global.u64
// --------------------------------------------------
__global__ void spmvCSR_u64(const int      *rowPtr,
                            const int      *colInd,
                            const uint64_t *vals,
                            const uint64_t *x,
                            uint64_t       *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_numRows) return;

    uint64_t sum = 0;
    int start = rowPtr[row];
    int end   = rowPtr[row + 1];
    for (int idx = start; idx < end; ++idx) {
        // load vals[idx]
        uint64_t v;
        const uint64_t *addrV = vals + idx;
        asm volatile("ld.global.u64 %0, [%1];"
                     : "=l"(v)
                     : "l"(addrV));
        // load x[colInd[idx]]
        int col = colInd[idx];  // 索引用常规 int
        uint64_t xv;
        const uint64_t *addrX = x + col;
        asm volatile("ld.global.u64 %0, [%1];"
                     : "=l"(xv)
                     : "l"(addrX));
        sum += v * xv;
    }
    // store result y[row]
    uint64_t *addrY = y + row;
    asm volatile("st.global.u64 [%0], %1;"
                 :
                 : "l"(addrY), "l"(sum));
}

int main()
{
    // ----------------------------
    // 2) SpMV 准备
    // ----------------------------
    const int h_numRows  = 1000000;
    const int h_numEdges = 10000000;
    cudaMemcpyToSymbol(d_numRows, &h_numRows, sizeof(h_numRows));

    std::vector<int>       h_rowPtr(h_numRows+1);
    std::vector<int>       h_colInd(h_numEdges);
    std::vector<uint64_t>  h_vals(h_numEdges, 1);
    std::vector<uint64_t>  h_x(h_numRows, 1);
    std::vector<uint64_t>  h_y(h_numRows, 0);

    // TODO: 填充 h_rowPtr, h_colInd, h_vals，表示你的图结构

    int *d_rowPtr, *d_colInd;
    uint64_t *d_vals, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (h_numRows+1)*sizeof(int));
    cudaMalloc(&d_colInd,  h_numEdges * sizeof(int));
    cudaMalloc(&d_vals,    h_numEdges * sizeof(uint64_t));
    cudaMalloc(&d_x,       h_numRows  * sizeof(uint64_t));
    cudaMalloc(&d_y,       h_numRows  * sizeof(uint64_t));

    cudaMemcpy(d_rowPtr, h_rowPtr.data(), (h_numRows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, h_colInd.data(),  h_numEdges *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,   h_vals.data(),    h_numEdges *sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,      h_x.data(),       h_numRows *sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blk = 256;
    int grd = (h_numRows + blk - 1) / blk;
    spmvCSR_u64<<<grd, blk>>>(d_rowPtr, d_colInd, d_vals, d_x, d_y);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y.data(), d_y, h_numRows*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    std::cout << "y[0] = " << h_y[0] << "\n";

    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}