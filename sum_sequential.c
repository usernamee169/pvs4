#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000

int main() {
    float *array = (float*)malloc(N * sizeof(float));
    float sum = 0.0f;
    
    // Инициализация массива
    for (int i = 0; i < N; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }
    
    clock_t start = clock();
    
    // Последовательное суммирование
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Sum: %f\n", sum);
    printf("Time: %f seconds\n", time);
    
    free(array);
    return 0;
}
