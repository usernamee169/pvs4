#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixOperations(float *A, float *B, float *add, float *sub, float *mul, float *div, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int idx = row * N + col;
        add[idx] = A[idx] + B[idx];
        sub[idx] = A[idx] - B[idx];
        mul[idx] = A[idx] * B[idx];
        div[idx] = A[idx] / B[idx];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Matrix size must be positive\n");
        return 1;
    }

    float *h_A = (float *)malloc(N * N * sizeof(float));
    float *h_B = (float *)malloc(N * N * sizeof(float));
    float *h_add = (float *)malloc(N * N * sizeof(float));
    float *h_sub = (float *)malloc(N * N * sizeof(float));
    float *h_mul = (float *)malloc(N * N * sizeof(float));
    float *h_div = (float *)malloc(N * N * sizeof(float));
    
    float *d_A, *d_B, *d_add, *d_sub, *d_mul, *d_div;

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (float)rand() / RAND_MAX;
            h_B[i * N + j] = (float)rand() / RAND_MAX + 0.1f;
        }
    }

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_add, N * N * sizeof(float));
    cudaMalloc(&d_sub, N * N * sizeof(float));
    cudaMalloc(&d_mul, N * N * sizeof(float));
    cudaMalloc(&d_div, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixOperations<<<numBlocks, blockSize>>>(d_A, d_B, d_add, d_sub, d_mul, d_div, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_add, d_add, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sub, d_sub, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mul, d_mul, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_div, d_div, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Parallel matrix operations completed\n");
    printf("Time: %f seconds\n", milliseconds / 1000);

    // Пример вывода первых 2x2 элементов
    printf("Sample results (first 2x2 elements):\n");
    for (int i = 0; i < 2 && i < N; i++) {
        for (int j = 0; j < 2 && j < N; j++) {
            int idx = i * N + j;
            printf("[%d][%d]: %f + %f = %f\n", i, j, h_A[idx], h_B[idx], h_add[idx]);
            printf("[%d][%d]: %f - %f = %f\n", i, j, h_A[idx], h_B[idx], h_sub[idx]);
            printf("[%d][%d]: %f * %f = %f\n", i, j, h_A[idx], h_B[idx], h_mul[idx]);
            printf("[%d][%d]: %f / %f = %f\n", i, j, h_A[idx], h_B[idx], h_div[idx]);
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(h_A); free(h_B); free(h_add); free(h_sub); free(h_mul); free(h_div);

    return 0;
}
