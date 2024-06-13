
#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>


//Link HIP
//#include "hip/hip_runtime.h"
//#include "roctx.h"
//#include "roctracer_ext.h"


//Links for dev
#include <thread>
#include <vector>
#include <array>
#include <typeinfo>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>
#include <algorithm> //for Each_fors
#include <string>
#include <utility>
#include <functional>
#include <future>
#include <cassert>
#include <chrono>
#include <type_traits>
#include <list>
#include <ranges>
#include <atomic> 
#include <cmath>


//Links Specx
#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"

//Links Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


#include "na.hpp"
#include "Tools.hpp"
#include "Taskflow_HPC.hpp"

#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20

//Links mpi
//#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

//Links omp
#define UseOpenMP
#ifdef UseOpenMP
	#include <omp.h>
#endif

//#define UseCUDA
#define UseHIP


#define HIP_CHECK(command) {               \
  hipError_t status = command;             \
  if (status!=hipSuccess) {                \
    std::cerr <<"Error: HIP reports "<< hipGetErrorString(status)<< std::endl; \
    std::abort(); } }


#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif





#define blockD 32

float elapsedTime1D,elapsedTime2D;
int   dimMatrix;


__global__ void Kernel(float *Md, float *Nd, float *Pd, int Width) {

  // Calculate the column index of the Pd element, denote by x
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  // Calculate the row index of the Pd element, denote by y
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  float Pvalue = 0;
  // each thread computes one element of the output matrix Pd.      
  for (int k = 0; k < Width; ++k) {
    Pvalue += Md[y*Width + k] * Nd[k*Width + x];
  }

  // write back to the global memory
  Pd[y*Width + x] = Pvalue;
}





__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
  // declare cache in the shared memory
  __shared__ float Mds[blockD][blockD];
  __shared__ float Nds[blockD][blockD];
 
  // keep track of column index of the Pd element using thread index
  int x = threadIdx.x + blockIdx.x * blockDim.x; // x is column
  // keep track of row index of the Pd element using thread index
  int y = threadIdx.y + blockIdx.y * blockDim.y; // y is row

  float Pvalue = 0;
  // Loop over the Md and Nd block dimension required to compute the Pd element
  for (int m = 0; m < Width/blockD; m++){
	
    // collaboratively loading of Md and Nd blocks into shared memory	 
    Mds[threadIdx.y][threadIdx.x] = Md[y * Width + (m * blockD + threadIdx.x)];
    Nds[threadIdx.y][threadIdx.x] = Md[(m * blockD + threadIdx.y) * Width + x];
    __syncthreads();
    
    // keep track of the running sum    
    for (int k = 0; k < blockD; k++)
      Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
    __syncthreads();
  }
  
  // write back to the global memory
  Pd[y * Width + x] = Pvalue;
}

void MatrixMultiplication2D(float *M, float *N, float *P, int Width) {

    int size = Width * Width * sizeof(float);
    float *Md, *Nd, *Pd; 

    // capture start time
    hipEvent_t     start, stop;
    hipEventCreate( &start ) ;
    hipEventCreate( &stop ) ;
    hipEventRecord( start, 0 ) ;

    // allocate memory on the GPU
    hipMalloc((void**)&Md, size) ;
    hipMalloc((void**)&Nd, size) ;
    hipMalloc((void**)&Pd, size) ;

    // transfer M and N to device memory
    hipMemcpy(Md, M, size, hipMemcpyHostToDevice) ;
    hipMemcpy(Nd, N, size, hipMemcpyHostToDevice) ;

    // kernel invocation code
    dim3 dimBlock(blockD, blockD);
    dim3 dimGrid(Width/blockD, Width/blockD);
    //MatrixMulKernel<<<dimGrid, dimBlock>>>( Md, Nd, Pd, Width);
    hipLaunchKernelGGL(MatrixMulKernel,dimGrid, dimBlock,0,0, Md, Nd, Pd, Width);

    // transfer P from device    
    hipMemcpy(P, Pd, size, hipMemcpyDeviceToHost) ;

    // get stop time, and display the timing results
    hipEventRecord( stop, 0 ) ;
    hipEventSynchronize( stop ) ;
    float   elapsedTime;
    hipEventElapsedTime( &elapsedTime,start, stop ) ;
    printf( "Time to generate 2D:  %3.1f ms\n", elapsedTime );

    elapsedTime2D=elapsedTime;

    // free the memory allocated on the GPU
    hipFree(Md);
    hipFree(Nd);
    hipFree(Pd);

    // destroy events to free memory
    hipEventDestroy( start );
    hipEventDestroy( stop );
}



void MatrixMultiplication1D(float *M, float *N, float *P, int Width) {

    int size = Width * Width * sizeof(float);
    float *Md, *Nd, *Pd;
    
    // capture start time
    hipEvent_t     start, stop;
    hipEventCreate( &start );
    hipEventCreate( &stop );
    hipEventRecord( start, 0 );

    // allocate memory on the GPU
    hipMalloc((void**)&Md, size);
    hipMalloc((void**)&Nd, size);
    hipMalloc((void**)&Pd, size);

    // transfer M and N to device memory
    hipMemcpy(Md, M, size, hipMemcpyHostToDevice);
    hipMemcpy(Nd, N, size, hipMemcpyHostToDevice);

    // kernel invocation code
    dim3 dimBlock(32, 32);
    dim3 dimGrid(Width/32, Width/32);
    //Kernel<<<dimGrid, dimBlock>>>( Md, Nd, Pd, Width);
    hipLaunchKernelGGL(Kernel,dimGrid, dimBlock,0,0, Md, Nd, Pd, Width);

    // transfer P from device     
    hipMemcpy(P,Pd,size,hipMemcpyDeviceToHost);

    // get stop time, and display the timing results
    hipEventRecord( stop, 0 );
    hipEventSynchronize( stop );
    float   elapsedTime;
    hipEventElapsedTime( &elapsedTime,start, stop );
    printf( "Time to generate 1D:  %3.1f ms\n", elapsedTime );

    elapsedTime1D=elapsedTime;

    // free the memory allocated on the GPU
    hipFree(Md);
    hipFree(Nd);
    hipFree(Pd);

    // destroy events to free memory
    hipEventDestroy( start );
    hipEventDestroy( stop );
}

void Test001 (void)
{    
    const int Width = dimMatrix;
    int size = Width * Width * sizeof(float);
    float *M, *N, *P;

    // allocate memory on the CPU
    M = (float*)malloc(size);
    N = (float*)malloc(size);
    P = (float*)malloc(size);

    // initialize the matrices
    for (int y=0; y<Width; y++) {
      for (int x=0; x<Width; x++){
          M[y*Width + x] = x + y*Width;
          N[y*Width + x] = x + y*Width; 
      }
    }

    MatrixMultiplication1D(M, N, P, Width);

    // free the memory allocated on the CPU
    free( M );
    free( N );
    free( P );
}

void Test002 (void) 
{ 
    const int Width = dimMatrix;

    int size = Width * Width * sizeof(float);
    float *M, *N, *P;   
    
    // allocate memory on the CPU
    M = (float*)malloc(size);
    N = (float*)malloc(size);
    P = (float*)malloc(size);

    // initialize the matrices
    for (int y=0; y<Width; y++) {
	    for (int x=0; x<Width; x++){
	   		  M[y*Width + x] = x + y*Width;
       		N[y*Width + x] = x + y*Width; 
	   }
    }

    MatrixMultiplication2D(M, N, P, Width);

    // free the memory allocated on the CPU
    free( M );
    free( N );
    free( P );
}   

int main()
{
  dimMatrix=1024*10;
  Test001();
  Test002();
  printf( " Speed Ratio:  %3.1f \n", elapsedTime1D/elapsedTime2D);
  return 0;
}







