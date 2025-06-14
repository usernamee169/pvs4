#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ROWS 1000
#define COLS 1000
#define N (ROWS * COLS)
#define BLOCK_SIZE 16

__global__ void matrixOpsKernel(float *a, float *b, float *add, float *sub, float *mul, float *div, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        div[idx] = a[idx] / b[idx];
    }
}

int main() {
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_add = (float*)malloc(N * sizeof(float));
    float *h_sub = (float*)malloc(N * sizeof(float));
    float *h_mul = (float*)malloc(N * sizeof(float));
    float *h_div = (float*)malloc(N * sizeof(float));
    
    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;
    
    // Инициализация матриц
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX + 0.1f;
        h_b[i] = (float)rand() / RAND_MAX + 0.1f;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_add, N * sizeof(float));
    cudaMalloc(&d_sub, N * sizeof(float));
    cudaMalloc(&d_mul, N * sizeof(float));
    cudaMalloc(&d_div, N * sizeof(float));
    
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((COLS + block.x - 1) / block.x, (ROWS + block.y - 1) / block.y);
    
    cudaEventRecord(start);
    
    matrixOpsKernel<<<grid, block>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, ROWS, COLS);
    
    cudaMemcpy(h_add, d_add, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sub, d_sub, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mul, d_mul, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_div, d_div, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("First add result: %f\n", h_add[0]);
    printf("First sub result: %f\n", h_sub[0]);
    printf("First mul result: %f\n", h_mul[0]);
    printf("First div result: %f\n", h_div[0]);
    printf("Time: %f seconds\n", milliseconds / 1000.0f);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(h_a); free(h_b); free(h_add); free(h_sub); free(h_mul); free(h_div);
    
    return 0;
}
