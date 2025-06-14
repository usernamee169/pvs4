#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge(float *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    float *L = (float *)malloc(n1 * sizeof(float));
    float *R = (float *)malloc(n2 * sizeof(float));

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void mergeSort(float *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
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

    float *array = (float *)malloc(N * sizeof(float));
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }

    clock_t start = clock();
    mergeSort(array, 0, N - 1);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequential merge sort completed\n");
    printf("Time: %f seconds\n", time_spent);

    // Проверка сортировки (опционально)
    for (int i = 0; i < N - 1; i++) {
        if (array[i] > array[i + 1]) {
            printf("Sorting error at index %d\n", i);
            break;
        }
    }

    free(array);
    return 0;
}
