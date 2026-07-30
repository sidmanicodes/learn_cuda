#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 256
#define N 1024

__global__ void matMulKernel(float *A, float *B, float *C, int n) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockIdx.x) + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int j = 0; j < n; j++) {
            sum += A[(row * n) + j] * B[(j * n) + col];
        }
        C[(row * n) + col] = sum;
    }
}

void matMul(float *A, float *B, float *C, int n) {
    nvtxRangePush("Begin matrix multiplication");

    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);

    nvtxRangePush("Device memory allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel execution");
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Cuda memcpy D2H");
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePop();
}

void initSquareMatrix(float *A, int n) {
    // n = # rows * # columns
    for (int i = 0; i < n; i++) {
        A[i] = (float)(rand() / RAND_MAX);
    }
}

int main() {
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    srand(time(NULL));

    initSquareMatrix(A, N*N);
    initSquareMatrix(C, N*N);
    matMul(A, B, C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}