#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

    // Generate unique log filenames
    char log_filename[50], output_filename[50];
    sprintf(log_filename, "mpi_parallel_log.txt");
    sprintf(output_filename, "mpi_parallel_output.txt");

    FILE *log_file = fopen(log_filename, "w");
    FILE *output_file = fopen(output_filename, "w");
    if (!log_file || !output_file) {
        printf("Error opening log files\n");
        free(T);
        free(T_new);
        return 1;
    }

    double residual = 1.0;
    int iteration = 0;

    clock_t start = clock(); // Start timing

    while (residual > MAX_RESIDUAL && iteration <= MAX_ITER) {
        if (iteration % 1000 == 0) {  // Print every 1000 iterations
            printf("Iteration: %d, Residual: %.9e\n", iteration, residual);
            fflush(stdout);
        }

        // Main computational kernel: average over neighbors in the grid
        #pragma omp target teams distribute parallel for simd collapse(2) map(tofrom: T[:(N+2)*(N+2)], T_new[:(N+2)*(N+2)])
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }

        // Reset residual
        double local_residual = 0.0;

        // Compute the largest change and copy T_new to T
        #pragma omp target teams distribute parallel for simd reduction(max : local_residual) collapse(2) map(tofrom: T[:(N+2)*(N+2)], T_new[:(N+2)*(N+2)])
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                local_residual = MAX(fabs(T_new(i, j) - T(i, j)), local_residual);
                T(i, j) = T_new(i, j);
            }
        }

        residual = local_residual;
        iteration++;
    }

    clock_t end = clock(); // End timing
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(log_file, "Final Iteration: %d, Final Residual: %.10e\n", iteration, residual);
    fprintf(log_file, "Execution Time: %.4f seconds\n", time_spent);
    fclose(log_file);

    // Save final temperature distribution
    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            fprintf(output_file, "%.6f ", T(i, j));
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);

    free(T);
    free(T_new);

    printf("Jacobi iteration completed in %.6f seconds.\n", time_spent);
    return 0;
}
