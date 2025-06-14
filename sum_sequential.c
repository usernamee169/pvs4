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

    float *array = (float *)malloc(N * sizeof(float));
    
    // Инициализация массива случайными значениями
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }

    // Вычисление суммы
    clock_t start = clock();
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequential sum: %f\n", sum);
    printf("Time: %f seconds\n", time_spent);

    free(array);
    return 0;
}
