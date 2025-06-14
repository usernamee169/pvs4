#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ROWS 1000
#define COLS 1000
#define N (ROWS * COLS)

__global__ void matrixOpsKernel(float *a, float *b, float *add, float *sub, 
                               float *mul, float *div, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        div[idx] = a[idx] / (b[idx] + 0.0001f);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        return 1;
    }
    
    int blockDimX = atoi(argv[1]);
    int blockDimY = atoi(argv[2]);
    
    if (blockDimX <= 0 || blockDimY <= 0 || blockDimX * blockDimY > 1024) {
        return 1;
    }

    
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_add = (float*)malloc(N * sizeof(float));
    float *h_sub = (float*)malloc(N * sizeof(float));
    float *h_mul = (float*)malloc(N * sizeof(float));
    float *h_div = (float*)malloc(N * sizeof(float));
    
    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;
    
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
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
    
    dim3 block(blockDimX, blockDimY);
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
    
    printf("Размер матрицы: %dx%d\n", ROWS, COLS);
    printf("Время: %f seconds\n\n", milliseconds / 1000.0f);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_add); 
    cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(h_a); free(h_b); free(h_add); free(h_sub); free(h_mul); free(h_div);
    
    return 0;
}
