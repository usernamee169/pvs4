#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    float *A = (float *)malloc(N * sizeof(float));
    float *B = (float *)malloc(N * sizeof(float));
    float *add = (float *)malloc(N * sizeof(float));
    float *sub = (float *)malloc(N * sizeof(float));
    float *mul = (float *)malloc(N * sizeof(float));
    float *div = (float *)malloc(N * sizeof(float));
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX + 0.1f; // Чтобы избежать деления на 0
    }

    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        add[i] = A[i] + B[i];
        sub[i] = A[i] - B[i];
        mul[i] = A[i] * B[i];
        div[i] = A[i] / B[i];
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequential array operations completed\n");
    printf("Time: %f seconds\n", time_spent);

    // Пример вывода первых 5 элементов
    printf("Sample results (first 5 elements):\n");
    for (int i = 0; i < 5 && i < N; i++) {
        printf("%f + %f = %f\n", A[i], B[i], add[i]);
        printf("%f - %f = %f\n", A[i], B[i], sub[i]);
        printf("%f * %f = %f\n", A[i], B[i], mul[i]);
        printf("%f / %f = %f\n", A[i], B[i], div[i]);
    }

    free(A); free(B); free(add); free(sub); free(mul); free(div);
    return 0;
}
