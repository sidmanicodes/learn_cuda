#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256
#define N 512
#define K 256

// Naiive CPU implementation
void matmul_cpu(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A[(i * n) + j] * B[(j * k) + p];
            }
            C[(i * n) + p] = sum;
        }
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
    // Allocate memory on host
    float *h_A, *h_B, *h_C;
    size_t A_size = (M * N) * sizeof(float);
    size_t B_size = (N * K) * sizeof(float);
    size_t C_size = (M * K) * sizeof(float);

    h_A = (float*)malloc(A_size);
    h_B = (float*)malloc(B_size);
    h_C = (float*)malloc(C_size);
    
    // Initialize A and B matrices
    srand(time(NULL));
    init_random_matrix(h_A, M, N);
    init_random_matrix(h_B, N, K);

    int num_trials = 20;
    
    // Benchmark CPU matmul
    double total_time_cpu = 0.0;
    for (int i = 0; i < num_trials; i++) {
        double start = get_time();
        matmul_cpu(h_A, h_B, h_C, M, N, K);
        double end = get_time();
        total_time_cpu += end - start;
    }
    double avg_time_cpu = total_time_cpu / num_trials;

    printf("Average CPU time for matmul of (%d, %d) and (%d, %d) matrices: %.3f ms\n",
        M, N, N, K, avg_time_cpu * 1'000);


    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
}