#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

__global__ void sumKernel(float *input, float *output, int size, int threadsPerBlock) {
    extern __shared__ float sharedData[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedData[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    for (int s = threadsPerBlock/2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }
    
    int threadsPerBlock = atoi(argv[1]);
    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        return 1;
    }

    
    float *h_array = (float*)malloc(N * sizeof(float));
    float *d_array, *d_sum;
    float *h_sum = (float*)malloc(((N + threadsPerBlock - 1) / threadsPerBlock) * sizeof(float));
    float finalSum = 0.0f;
    
    // Initialize array with random values
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)rand() / RAND_MAX;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMalloc(&d_sum, ((N + threadsPerBlock - 1) / threadsPerBlock) * sizeof(float));
    
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(threadsPerBlock);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEventRecord(start);
    
    sumKernel<<<grid, block, threadsPerBlock * sizeof(float)>>>(d_array, d_sum, N, threadsPerBlock);
    
    cudaMemcpy(h_sum, d_sum, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < grid.x; i++) {
        finalSum += h_sum[i];
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Размер массива: %d\n", N);
    printf("Время: %f seconds\n\n", milliseconds / 1000.0f);
    
    cudaFree(d_array);
    cudaFree(d_sum);
    free(h_array);
    free(h_sum);
    
    return 0;
}
