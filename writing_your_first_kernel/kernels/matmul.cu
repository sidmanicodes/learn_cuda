#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define N 512
#define K 256
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Naiive CPU implementation
void matmul_cpu(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += A[(i * n) + j] * B[(j * k) + p];
            }
            C[(i * k) + p] = sum;
        }
    }
}

// GPU implementation
__global__ void matmul_gpu(float* A, float* B, float* C, int m, int n, int k) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        
        for (int l = 0; l < n; l++) {
            sum += A[(row * n) + l] * B[(l * k) + col];
        }
        C[(row * k) + col] = sum;
    }
}

// Random (flattened) matrix initialization
void init_random_matrix(float* A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[(i * n) + j] = (float)rand() / RAND_MAX;
        }
    }
}

void print_matrix(float* A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f ", A[(i * m) + j]);
        }
        printf("\n");
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (ts.tv_nsec * 1e-9);
}

int main() {
    // Initialize host and device pointers
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t A_size = (M * N) * sizeof(float);
    size_t B_size = (N * K) * sizeof(float);
    size_t C_size = (M * K) * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(A_size);
    h_B = (float*)malloc(B_size);
    h_C_cpu = (float*)malloc(C_size);
    h_C_gpu = (float*)malloc(C_size);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B, B_size));
    CUDA_CHECK(cudaMalloc(&d_C, C_size));

    // Initialize A and B matrices
    srand(time(NULL));
    init_random_matrix(h_A, M, N);
    init_random_matrix(h_B, N, K);

    // Copy data to CPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice));

    // Define number of blocks to use
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((K + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Warm-up runs
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int num_trials = 20;
    
    // Benchmark CPU matmul
    double total_time_cpu = 0.0;
    for (int i = 0; i < num_trials; i++) {
        double start = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
        double end = get_time();
        total_time_cpu += end - start;
    }
    double avg_time_cpu = total_time_cpu / num_trials;
    
    // Benchmark GPU matmul
    double total_time_gpu = 0.0;
    for (int i = 0; i < num_trials; i++) {
        double start = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        total_time_gpu += end - start;
    }
    double avg_time_gpu = total_time_gpu / num_trials;

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, C_size, cudaMemcpyDeviceToHost));
    bool correct = true;

    const float abs_tol = 1e-5f;
    const float rel_tol = 1e-3f;
    for (int i = 0; i < M * K; i++) {
        float diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        float tol = abs_tol + rel_tol * fabs(h_C_cpu[i]);
        if (diff > tol) {
            correct = false;
            printf("ERROR: difference at index %d is: %.10f (tolerance %.10f)\n", i, diff, tol);
            break;
        }
    }

    // // Print matrices (optional)
    // printf("C on CPU:\n");
    // print_matrix(h_C_cpu, M, K);
    // printf("C on GPU:\n");
    // print_matrix(h_C_gpu, M, K);

    printf("Implementations match: %s\n", correct ? "true" : "false");

    printf("Average CPU time for matmul of (%d, %d) and (%d, %d) matrices: %.3f ms\n",
        M, N, N, K, avg_time_cpu * 1'000);
    
    printf("Average GPU time for matmul of (%d, %d) and (%d, %d) matrices: %.3f ms\n",
        M, N, N, K, avg_time_gpu * 1'000);

    printf("CPU -> GPU speedup: %.3fx\n", (float)(avg_time_cpu / avg_time_gpu));

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}