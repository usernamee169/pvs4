#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumKernel(float *array, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, array[idx]);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    float *h_array = (float*)malloc(n * sizeof(float));
    float *d_array, *d_result;
    float h_result = 0.0;
    
    for(int i = 0; i < n; i++) {
        h_array[i] = 1.0;
    }
    
    cudaMalloc((void**)&d_array, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    sumKernel<<<gridSize, blockSize>>>(d_array, d_result, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Array size: %d\n", n);
    printf("Sum: %f\n", h_result);
    printf("Time: %f ms\n", milliseconds);
    
    cudaFree(d_array);
    cudaFree(d_result);
    free(h_array);
    
    return 0;
}
