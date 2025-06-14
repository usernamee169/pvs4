#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1048576  // Должно быть степенью 2 для bitonic sort
#define THREADS 256

__device__ void swap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

__global__ void bitonicSortStep(float *devValues, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    if (ixj > i) {
        if ((i & k) == 0) {
            if (devValues[i] > devValues[ixj]) {
                swap(devValues[i], devValues[ixj]);
            }
        }
        if ((i & k) != 0) {
            if (devValues[i] < devValues[ixj]) {
                swap(devValues[i], devValues[ixj]);
            }
        }
    }
}

int main() {
    float *h_values = (float*)malloc(N * sizeof(float));
    float *d_values;
    
    // Инициализация массива
    for (int i = 0; i < N; i++) {
        h_values[i] = (float)rand() / RAND_MAX;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc(&d_values, N * sizeof(float));
    cudaMemcpy(d_values, h_values, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (N + THREADS - 1) / THREADS;
    
    cudaEventRecord(start);
    
    int j, k;
    for (k = 2; k <= N; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocks, THREADS>>>(d_values, j, k);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(h_values, d_values, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("First element: %f\n", h_values[0]);
    printf("Last element: %f\n", h_values[N-1]);
    printf("Time: %f seconds\n", milliseconds / 1000.0f);
    
    cudaFree(d_values);
    free(h_values);
    
    return 0;
}
