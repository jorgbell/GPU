#include <string>
#include <iostream>
#include <ostream>

// macro de manejo de errores
#include <stdio.h>
#include <assert.h>
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

//ejemplo de suma de vectores
//#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h> //CUDA

//------------------

// void CPUFunction()
// {
//   printf("This function is defined to run on the CPU.\n");
// }

// __global__ void GPUFunction()
// {
//   printf("This function is defined to run on the GPU.\n");
// }

//SUMA DE VECTORES

double wtime(void)
{
        static struct timeval   tv0;
        double time_;

        gettimeofday(&tv0,(struct timezone*)0);
        time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
        return( time_/1000000);
}


void vecAdd(float* A, float* B, float* C,
   int n)
{
	int i;
	for (i = 0; i < n; i++)
		C[i] = A[i] + B[i];
}


__global__ 
void vecAdd_GPU(float* A, float* B, float* C,
   int n)
{
	int i;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<n) 
		C[i] = A[i] + B[i];
}

int main(int argc, char *argv[])
{

//   CPUFunction();
//   GPUFunction<<<1, 1>>>();
//   /*
//  * The macro can be wrapped around any function returning
//  * a value of type `cudaError_t`.
//  */
//   checkCuda( cudaDeviceSynchronize() );
//--------------------------------------------------------------------------
//EJEMPLO DE SUMA DE VECTORES

	float *a, *b, *c, *c_host;
	float *a_GPU, *b_GPU, *c_GPU;

	int i, N;

	double t0, t1;


	if(argc>1) {
		N = atoi(argv[1]); printf("N=%i\n", N);
	} else {
		printf("Error!!!! \n ./exec number\n");
	return (0);
	}

	// Mallocs CPU
	a  = (float *)malloc(sizeof(float)*N);
	b  = (float *)malloc(sizeof(float)*N);
	c  = (float *)malloc(sizeof(float)*N);
	c_host  = (float *)malloc(sizeof(float)*N);
	for (i=0; i<N; i++){ a[i] = i-1; b[i] = i;}

	/*****************/
	/* Add Matrix CPU*/
	/*****************/
	t0 = wtime();
	vecAdd(a, b, c, N);
	t1 = wtime(); printf("Time CPU=%f\n", t1-t0);

	// Get device memory for A, B, C
	// copy A and B to device memory
	cudaMalloc((void **) &a_GPU, N*sizeof(float));
	cudaMemcpy(a_GPU, a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &b_GPU, N*sizeof(float));
	cudaMemcpy(b_GPU, b, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &c_GPU, N*sizeof(float));

	// Kernel execution in device
	// (vector add in device)
	dim3 DimBlock(256); // 256 thread per block
	dim3 DimGrid(ceil(N/256.0)+1);
	t0 = wtime();
	vecAdd_GPU<<<DimGrid,DimBlock>>>(a_GPU, b_GPU, c_GPU, N);
	checkCuda(cudaThreadSynchronize());
	t1 = wtime(); printf("Time GPU=%f\n", t1-t0);

	// copy C to host memory
	cudaMemcpy(c_host, c_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	/************/
	/* Results  */
	/************/
	for (i=0; i<N; i++)
		if(fabs(c[i]-c_host[i])>1e-5){
			printf("c!=c_host in (%i): ", i);
			printf("C[%i] = %f C_GPU[%i]=%f\n", i, c[i], i, c_host[i] );
		}

	/* Free CPU */
	free(a);
	free(b);
	free(c);
	free(c_host);

	cudaFree(a_GPU); cudaFree(b_GPU); cudaFree(c_GPU);

	return(1);
}
