#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Matrix size must be positive\n");
        return 1;
    }

    float **A = (float **)malloc(N * sizeof(float *));
    float **B = (float **)malloc(N * sizeof(float *));
    float **add = (float **)malloc(N * sizeof(float *));
    float **sub = (float **)malloc(N * sizeof(float *));
    float **mul = (float **)malloc(N * sizeof(float *));
    float **div = (float **)malloc(N * sizeof(float *));
    
    for (int i = 0; i < N; i++) {
        A[i] = (float *)malloc(N * sizeof(float));
        B[i] = (float *)malloc(N * sizeof(float));
        add[i] = (float *)malloc(N * sizeof(float));
        sub[i] = (float *)malloc(N * sizeof(float));
        mul[i] = (float *)malloc(N * sizeof(float));
        div[i] = (float *)malloc(N * sizeof(float));
    }

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX + 0.1f;
        }
    }

    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            add[i][j] = A[i][j] + B[i][j];
            sub[i][j] = A[i][j] - B[i][j];
            mul[i][j] = A[i][j] * B[i][j];
            div[i][j] = A[i][j] / B[i][j];
        }
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequential matrix operations completed\n");
    printf("Time: %f seconds\n", time_spent);

    // Пример вывода первых 2x2 элементов
    printf("Sample results (first 2x2 elements):\n");
    for (int i = 0; i < 2 && i < N; i++) {
        for (int j = 0; j < 2 && j < N; j++) {
            printf("[%d][%d]: %f + %f = %f\n", i, j, A[i][j], B[i][j], add[i][j]);
            printf("[%d][%d]: %f - %f = %f\n", i, j, A[i][j], B[i][j], sub[i][j]);
            printf("[%d][%d]: %f * %f = %f\n", i, j, A[i][j], B[i][j], mul[i][j]);
            printf("[%d][%d]: %f / %f = %f\n", i, j, A[i][j], B[i][j], div[i][j]);
        }
    }

    for (int i = 0; i < N; i++) {
        free(A[i]); free(B[i]); free(add[i]); free(sub[i]); free(mul[i]); free(div[i]);
    }
    free(A); free(B); free(add); free(sub); free(mul); free(div);

    return 0;
}
