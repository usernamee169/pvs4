#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    float *array = (float*)malloc(n * sizeof(float));
    
    srand(time(NULL));
    for(int i = 0; i < n; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }
    
    clock_t start = clock();
    float sum = 0.0;
    for(int i = 0; i < n; i++) {
        sum += array[i];
    }
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Array size: %d\n", n);
    printf("Sum: %f\n", sum);
    printf("Time: %f seconds\n", time_spent);
    
    free(array);
    return 0;
}
