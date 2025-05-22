#define ORDER 1000
#define AVAL 3.0
#define BVAL 5.0
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

int main (int argc, char *argv[]){
  int Ndim, Pdim, Mdim;
  int i,j,k ;

  double *A, *B , *C;
  double tmp;
  Ndim = ORDER;
  Pdim = ORDER;
  Mdim = ORDER;

  A= (double *)malloc(Ndim*Pdim*sizeof(double));
  B= (double *)malloc(Pdim*Mdim*sizeof(double));
  C= (double *)malloc(Ndim*Mdim*sizeof(double));

  for (i=0; i<Ndim; i++)
    for (j=0; j<Pdim ; j++)
      *(A+(i*Ndim+j)) = AVAL ;

  for (i=0; i<Pdim; i++)
    for (j=0; j<Mdim; j++)
      *(B+(i*Pdim+j)) = BVAL;

  for (i=0; i<Ndim; i++)
    for (j=0; j<Mdim; j++)
      *(C+(i*Ndim+j)) = 0.0;


  int z;
 for (z=1; z<129;){
   double  start_time=omp_get_wtime();
   omp_set_num_threads(z);
#pragma omp parallel for private(j,k,tmp) 
	for (i=0; i<Ndim; i++){ 
	  for (j=0; j<Mdim; j++){
	  double tmp = 0.0;
	    for (k=0; k<Pdim; k++){
	      tmp+= *(A+(i*Ndim+k)) * *(B+(k*Pdim+j));
	    }
	    *(C+(i*Ndim+j))=tmp;

	  }}

double	run_time = omp_get_wtime() - start_time ;
	double dN = (double)ORDER;
	double mflops=2.0*dN*dN*dN/(1000000.0*run_time) ;
	printf("%d number of threads used, %f mflops, %f Wall Clock\n",z, mflops, run_time);
	z = z * 2;
     
 }
	return 0;}
	    
