#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void arrayOperations(float *A, float *B, float *add, float *sub, float *mul, float *div, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        add[idx] = A[idx] + B[idx];
        sub[idx] = A[idx] - B[idx];
        mul[idx] = A[idx] * B[idx];
        div[idx] = A[idx] / B[idx];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Array size must be positive\n");
        return 1;
    }

    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_add = (float *)malloc(N * sizeof(float));
    float *h_sub = (float *)malloc(N * sizeof(float));
    float *h_mul = (float *)malloc(N * sizeof(float));
    float *h_div = (float *)malloc(N * sizeof(float));
    
    float *d_A, *d_B, *d_add, *d_sub, *d_mul, *d_div;

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX + 0.1f;
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_add, N * sizeof(float));
    cudaMalloc(&d_sub, N * sizeof(float));
    cudaMalloc(&d_mul, N * sizeof(float));
    cudaMalloc(&d_div, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    arrayOperations<<<numBlocks, blockSize>>>(d_A, d_B, d_add, d_sub, d_mul, d_div, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_add, d_add, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sub, d_sub, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mul, d_mul, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_div, d_div, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Parallel array operations completed\n");
    printf("Time: %f seconds\n", milliseconds / 1000);

    // Пример вывода первых 5 элементов
    printf("Sample results (first 5 elements):\n");
    for (int i = 0; i < 5 && i < N; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_add[i]);
        printf("%f - %f = %f\n", h_A[i], h_B[i], h_sub[i]);
        printf("%f * %f = %f\n", h_A[i], h_B[i], h_mul[i]);
        printf("%f / %f = %f\n", h_A[i], h_B[i], h_div[i]);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(h_A); free(h_B); free(h_add); free(h_sub); free(h_mul); free(h_div);

    return 0;
}
