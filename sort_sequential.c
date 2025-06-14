#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N1 100000
#define N2 500000
#define N3 1000000

void merge(float *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    float *L = (float*)malloc(n1 * sizeof(float));
    float *R = (float*)malloc(n2 * sizeof(float));

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

void testSort(int size) {
    float *array = (float*)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }
    
    clock_t start = clock();
    
    mergeSort(array, 0, size - 1);
    
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    
    int sorted = 1;
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i+1]) {
            sorted = 0;
            break;
        }
    }
    
    printf("Размер массива: %d\n", size);
    printf("Время: %f seconds\n\n", time);
    
    free(array);
}

int main() {
    srand(time(NULL));
    
    printf("Последовательная ортировка:\n");
    testSort(N1);
    testSort(N2);
    testSort(N3);
    
    return 0;
}
