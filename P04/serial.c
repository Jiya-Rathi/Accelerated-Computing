#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define N 1000
#define MAX_ITER 1000000
#define MAX_RESIDUAL 1.0e-8

#define T(i, j) (T[(i) * (N + 2) + (j)])
#define T_new(i, j) (T_new[(i) * (N + 2) + (j)])

int main() {
    double *T = (double *)malloc((N + 2) * (N + 2) * sizeof(double));
    double *T_new = (double *)malloc((N + 2) * (N + 2) * sizeof(double));
    if (!T || !T_new) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize grid and boundary conditions
    for (unsigned i = 0; i <= N + 1; i++) {
        for (unsigned j = 0; j <= N + 1; j++) {
            if (j == 0 || j == (N + 1)) {
                T(i, j) = 1.0;
            } else {
                T(i, j) = 0.0;
            }
        }
    }

    int iteration = 0;
    double residual = 1.0;
    clock_t start, end;
    
    FILE *log_fp = fopen("jacobi_log.txt", "w");
    if (!log_fp) {
        printf("Failed to open log file\n");
        return 1;
    }
    fprintf(log_fp, "Iteration Time(s) Residual\n");
    
    start = clock();
    
    while (residual > MAX_RESIDUAL && iteration < MAX_ITER) {
        residual = 0.0;

        // Jacobi iteration
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        // Compute residual and update T
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
                T(i, j) = T_new(i, j);
            }
        }

        iteration++;
        
        if (iteration % 1000 == 0) {
            end = clock();
            double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
            printf("Iteration %d, Residual = %.9e, Time = %.6f sec\n", iteration, residual, elapsed_time);
            fprintf(log_fp, "%d %.6f %.9e\n", iteration, elapsed_time, residual);
        }
    }
    
    end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Final Residual = %.9e after %d iterations, Total Time = %.6f sec\n", residual, iteration, total_time);
    fprintf(log_fp, "%d %.6f %.9e\n", iteration, total_time, residual);
    fclose(log_fp);

    // Write final result to a file
    FILE *fp = fopen("jacobi_output.txt", "w");
    if (fp) {
        for (unsigned i = 0; i <= N + 1; i++) {
            for (unsigned j = 0; j <= N + 1; j++) {
                fprintf(fp, "%lf ", T(i, j));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    } else {
        printf("Failed to open output file\n");
    }

    free(T);
    free(T_new);
    return 0;
}
