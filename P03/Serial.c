#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double f(double x) {
    return acos(cos(x) / (1.0 + 2.0 * cos(x)));
}

int main(int argc, char *argv[]) {
    if (argc < 1) {
        fprintf(stderr, "Usage: %s\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    double a = 0.0;
    double b = M_PI / 2.0;
    
    double exact = (5.0 * M_PI * M_PI) / 24.0;
    
    for (long long n = 10; n <= 100000000; n *= 10) {
        if (n % 2 != 0) {
            n++;
        }
        
        double h = (b - a) / n;
        
 
        double sum = f(a) + f(b);
        
        double start_time = omp_get_wtime();
        
         #pragma omp parallel for reduction(+:sum) schedule(static)
        for (long long i = 1; i < n; i++) {
            double x = a + i * h;
            double weight = (i % 2 == 0) ? 2.0 : 4.0;
            sum += weight * f(x);
        }
        
  
        double integral = (h / 3.0) * sum;
        
           double end_time = omp_get_wtime();
        double runtime = end_time - start_time;
        
        double error = fabs(integral - exact);
        
   
        printf("n = %lld\n", n);
        printf("Approximate integral: %.15f\n", integral);
        printf("Exact integral:       %.15f\n", exact);
        printf("Absolute error:       %.15e\n", error);
        printf("Runtime:              %.6f seconds\n", runtime);
        printf("----------------------------------------\n");
    }
    
    return 0;

