#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N1 131072   // 2^17
#define N2 524288   // 2^19
#define N3 1048576  // 2^20

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

void testSort(int size) {
    float *h_values = (float*)malloc(size * sizeof(float));
    float *d_values;
    
    for (int i = 0; i < size; i++) {
        h_values[i] = (float)rand() / RAND_MAX;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc(&d_values, size * sizeof(float));
    cudaMemcpy(d_values, h_values, size * sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (size + THREADS - 1) / THREADS;
    
    cudaEventRecord(start);
    
    int j, k;
    for (k = 2; k <= size; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocks, THREADS>>>(d_values, j, k);
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(h_values, d_values, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    int sorted = 1;
    for (int i = 0; i < size - 1; i++) {
        if (h_values[i] > h_values[i+1]) {
            sorted = 0;
            break;
        }
    }
    
    printf("Размер массива: %d\n", size);
    printf("Время: %f seconds\n\n", milliseconds / 1000.0f);
    
    cudaFree(d_values);
    free(h_values);
}

int main() {
    srand(time(NULL));
    
    printf("Параллельная сортировка:\n");
    testSort(N1);
    testSort(N2);
    testSort(N3);
    
    return 0;
}
