#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ void swap(float *a, float *b) {
    float t = *a;
    *a = *b;
    *b = t;
}

__global__ void bitonicSortStep(float *devValues, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    if (ixj > i) {
        if ((i & k) == 0) {
            if (devValues[i] > devValues[ixj]) {
                swap(&devValues[i], &devValues[ixj]);
            }
        }
        if ((i & k) != 0) {
            if (devValues[i] < devValues[ixj]) {
                swap(&devValues[i], &devValues[ixj]);
            }
        }
    }
}

void bitonicSort(float *values, int N) {
    float *devValues;
    cudaMalloc(&devValues, N * sizeof(float));
    cudaMemcpy(devValues, values, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocks, threads>>>(devValues, j, k);
        }
    }

    cudaMemcpy(values, devValues, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devValues);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0 || (N & (N - 1)) != 0) {
        printf("Array size must be positive and power of 2\n");
        return 1;
    }

    float *array = (float *)malloc(N * sizeof(float));
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bitonicSort(array, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Parallel bitonic sort completed\n");
    printf("Time: %f seconds\n", milliseconds / 1000);

    // Проверка сортировки
    for (int i = 0; i < N - 1; i++) {
        if (array[i] > array[i + 1]) {
            printf("Sorting error at index %d\n", i);
            break;
        }
    }

    free(array);
    return 0;
}
