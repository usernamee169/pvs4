#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 1000
#define COLS 1000
#define N (ROWS * COLS)

int main() {
    float *a = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *add = (float*)malloc(N * sizeof(float));
    float *sub = (float*)malloc(N * sizeof(float));
    float *mul = (float*)malloc(N * sizeof(float));
    float *div = (float*)malloc(N * sizeof(float));
    
    // Инициализация матриц
    for (int i = 0; i < N; i++) {
        a[i] = (float)rand() / RAND_MAX + 0.1f;
        b[i] = (float)rand() / RAND_MAX + 0.1f;
    }
    
    clock_t start = clock();
    
    // Операции с матрицами
    for (int i = 0; i < N; i++) {
        add[i] = a[i] + b[i];
        sub[i] = a[i] - b[i];
        mul[i] = a[i] * b[i];
        div[i] = a[i] / b[i];
    }
    
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("First add result: %f\n", add[0]);
    printf("First sub result: %f\n", sub[0]);
    printf("First mul result: %f\n", mul[0]);
    printf("First div result: %f\n", div[0]);
    printf("Time: %f seconds\n", time);
    
    free(a); free(b); free(add); free(sub); free(mul); free(div);
    return 0;
}
