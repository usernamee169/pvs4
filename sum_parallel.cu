#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumArray(float *array, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, array[idx]);
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

    float *h_array = (float *)malloc(N * sizeof(float));
    float *d_array, *d_result;
    float h_result = 0.0f;
    
    // Инициализация массива
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)rand() / RAND_MAX;
    }

    // Выделение памяти на устройстве
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Запуск ядра
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sumArray<<<numBlocks, blockSize>>>(d_array, d_result, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копирование результата обратно
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Parallel sum: %f\n", h_result);
    printf("Time: %f seconds\n", milliseconds / 1000);

    // Освобождение памяти
    cudaFree(d_array);
    cudaFree(d_result);
    free(h_array);

    return 0;
}
