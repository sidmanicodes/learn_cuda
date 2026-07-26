#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10'000'000
#define BLOCK_SIZE 256

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + (ts.tv_nsec * 1e-9);
}

void init_vector(float* vec, size_t n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX; // normalize elements to be in [0, 1]
    }
}

void vector_add_cpu(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i ++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float* a, float* b, float* c, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Initialize pointers to arrays on both host and device
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL)); // seed random number generator
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warm-up runs to allow GPU to reach steady-state internal temp
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    
    int num_trials = 20;

    // Benchmark CPU runs
    printf("Benchmarking CPU performance...\n");
    double total_time_cpu = 0;
    for (int i = 0; i < num_trials; i++) {
        double time_start_cpu = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double time_end_cpu = get_time();
        total_time_cpu += time_end_cpu - time_start_cpu;
    }
    total_time_cpu = total_time_cpu / num_trials;
    
    // Benchmark GPU runs
    printf("Benchmarking GPU performance...\n");
    double total_time_gpu = 0;
    for (int i = 0; i < num_trials; i++) {
        double time_start_gpu = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double time_end_gpu = get_time();
        total_time_gpu +=  time_end_gpu - time_start_gpu;
    }
    total_time_gpu = total_time_gpu / num_trials;

    // Display results
    printf("Final results:\n");
    printf("Total CPU time: %.3f\n", total_time_cpu);
    printf("Total GPU time: %.3f\n", total_time_gpu);
    printf("CPU -> GPU speedup: %.3fx\n", total_time_gpu / total_time_cpu);
}