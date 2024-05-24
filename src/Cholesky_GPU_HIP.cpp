
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

//#define BLOCK_SIZE 128
//#define BLOCK_SIZE 1024

 typedef struct {
 	  unsigned int num_columns;
   	  unsigned int num_rows;
	  unsigned int dimension; 
      unsigned int dimensionSizeof; 
 	  unsigned int pitch; 
 	  double* data;
  } Matrix;







/*********************************************************************************************************************************************************/



/*********************************************************************************************************************************************************/
// BEGIN::INTRODUCTION
int check_if_symmetric                 (const Matrix M); 
int check_if_diagonal_dominant         (const Matrix M);
Matrix build_init_matrix               (unsigned int, unsigned int);
Matrix allocate_matrix                 (int num_rows, int num_columns, int init);

void writeMatrix                       (const Matrix);
void copy_matrix_to_device             (Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device           (Matrix Mhost,   const Matrix Mdevice);
Matrix allocate_matrix_on_gpu          (const Matrix M);
// END::INTRODUCTION
/*********************************************************************************************************************************************************/



/*********************************************************************************************************************************************************/
// BEGIN::HIP AMD GPU

__global__ void chol_kernel(double * U, int ops_per_thread, int m) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i, j, k;
    unsigned int num_rows = m;
    for (k = 0; k < num_rows; k++) {
        if (tx == 0) {
            U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
            for (j = (k + 1); j < num_rows; j++) {
                U[k * num_rows + j] /= U[k * num_rows + k];
            }
        }
        __syncthreads();
        int istart = ( k + 1 )  +  tx * ops_per_thread;
        int iend = istart + ops_per_thread;      
        for (i = istart; i < iend; i++) {
            for (j = i; j < num_rows; j++) {
                U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    int istart = tx * ops_per_thread;
    int iend   = istart + ops_per_thread;
    for (i = istart; i < iend; i++) {
        for (j = 0; j < i; j++) {
            U[i * num_rows + j] = 0.0;
        }
    }
}


__global__ void chol_kernel_optimized_div(double * U, int k, int stride, int m) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j;
    unsigned int num_rows = m;
    if (tx == 0) { U[k * num_rows + k] = sqrt(U[k * num_rows + k]); }
    int offset  = (k + 1); 
    int jstart  = threadIdx.x + offset;
    int jstep   = stride;
    int jtop    = num_rows - 1;
    int jbottom = (k + 1);
    if (blockIdx.x == 0) {
        for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
            U[k * num_rows + j] /= U[k * num_rows + k]; 
        }
    }
}

__global__ void chol_kernel_optimized(double * U, int k, int stride, int m) {
    unsigned int j;
    unsigned int num_rows = m; 
    int i       = blockIdx.x + (k + 1);
    int offset  = i;
    int jstart  = threadIdx.x + offset;
    int jstep   = stride;
    int jtop    = num_rows - 1;
    int jbottom = i;
    for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
        U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
    }
}

__global__ void matrix_mult(double* C, double* A, double* B, int m, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int i = idx / n;
	int k = idx - n * i;
	if (n * m > idx) {
		for (int j = 0; j < n; j++) {
			C[idx] += A[n * i + j] * B[n * j + k];
		}
	}
}

__global__ void matrix_equal(volatile bool *Q, double* A, double* B, int nb, double deltaError) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < nb)
		//if (abs(A[idx]-B[idx])>deltaError) { Q[0]=false; printf("F"); } else  { printf("T"); }
		if (abs(A[idx]-B[idx])>deltaError) { Q[0]=false;  } 
}

__global__ void matrix_copy(double *R, double *A, int r, int c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int i,j;
	if (idx < r*c) { i=idx/r; j=idx-i*r;  R[j * c + i] = A[j * c + i]; }
}


__global__ void parallelComputation(int* gpuData, int startIndex, int endIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (startIndex+tid < endIndex) {
        gpuData[startIndex+tid] *= 2; 
    }
}

__global__ void matrix_copy(double *R, double *A, int nb) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < nb)
		R[idx]=A[idx];
  
}

__global__ void matrix_transpose(double *R, double *A, int r, int c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int i,j;
	if (idx < r*c) { i=idx/r; j=idx-i*r;  R[i * r + j] = A[j * c + i]; }
}

__global__ void matrix_lower_triangular(double *R, int r, int c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int i,j;
	if (idx < r*c) { i=idx/r; j=idx-i*r;  if (j<i) { R[i * r + j]=0.0; } }
}

__global__ void calculat_mandelbrot(int *pos,int w,int h,int max_iteration)
{
	int N        = (h*w);
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
    int idx = row * w + col;

    if(col > w || row > h || idx > N) return;

    float x0 = (float)row/w*(float)3.5-(float)2.5;
    float y0 = (float)col/h*(float)2.0-(float)1.0; 

    int x = 0, y = 0, iter = 0, xtemp = 0;
    while((x*x-y*y <= 4) && (iter < max_iteration))
    { 
        xtemp = x*x-y*y+x0;
        y = 2*x*y+y0;
        x = xtemp;
        iter++;
    }
    int color = 255 - (int)(iter/(float)max_iteration*(float)255);
    __syncthreads();
    pos[idx] = color;
}




 __global__ void convolution1DKernel(double *R,double *A, double *K,int nx, int ny, int ks)
{
	int tid    = threadIdx.x;
	int iy     = blockIdx.x  + (ks - 1)/2;
	int ix     = threadIdx.x + (ks - 1)/2;
	int idx    = iy*nx +ix;
	int K2     = ks*ks;
	int center = (ks -1)/2;
	int ii, jj;
	double sum = 0.0;
	extern __shared__ float sdata[];
	if (tid<K2)
		sdata[tid] = K[tid];
	__syncthreads();      
	if (idx<nx*ny){
		for (int ki = 0; ki<ks; ki++)
			for (int kj = 0; kj<ks; kj++){
					ii = kj + ix - center;
					jj = ki + iy - center;
					sum+=A[jj*nx+ii]*sdata[ki*ks + kj];
			}	
		R[idx] = sum;
	}
}


__global__ void convolution2DKernel(const double *input, const double *kernel, double *output,
                          int inputWidth, int inputHeight,
                          int kernelWidth, int kernelHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < inputWidth && row < inputHeight) {
        int halfKernelWidth = kernelWidth / 2;
        int halfKernelHeight = kernelHeight / 2;

        double result = 0.0f;

        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                int inputRow = row - halfKernelHeight + i;
                int inputCol = col - halfKernelWidth + j;

                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                    result += input[inputRow * inputWidth + inputCol] * kernel[i * kernelWidth + j];
                }
            }
        }

        output[row * inputWidth + col] = result;
    }
}


// END::HIP AMD GPU
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::TOOLS MEMORY TRANSFER HIP AMD GPU

Matrix allocate_matrix(int num_rows, int num_columns, int init) {
   	Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    M.dimension=M.num_rows * M.num_columns;
    unsigned int size = M.num_rows * M.num_columns;
    M.dimension=size;
    M.dimensionSizeof=size*sizeof(double);
	M.data = (double *)malloc(M.dimensionSizeof);
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.data[i] = 0;
        else
            M.data[i] = (double) rand() / (double) RAND_MAX;
    }
    return M;
}

Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    hipMalloc((void**)&Mdevice.data, M.dimensionSizeof);
    return Mdevice;
}

void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    hipMemcpy(Mdevice.data, Mhost.data,Mhost.dimensionSizeof,hipMemcpyHostToDevice);
}

void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    hipMemcpy(Mhost.data, Mdevice.data,Mdevice.dimensionSizeof,hipMemcpyDeviceToHost);
}

//END::TOOLS MEMORY TRANSFER HIP AMD GPU
/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
//BEGIN:: BUILD INIT MATRIX


Matrix build_init_matrix(unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns     = M.pitch = num_columns;
	M.num_rows        = num_rows; 
    unsigned int size = M.num_rows * M.num_columns;
    M.dimension       = size;
    M.dimensionSizeof = size*sizeof(double);
	M.data            = (double *)malloc(M.dimensionSizeof);

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
    std::cout<<"[INFO]: Create Matrix definite positiv"<<"\n";
    std::cout<<"[INFO]: Creating a "<<num_rows<<"x"<<num_columns<<" matrix with random numbers between [-.5, .5]... ";
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.data[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
        std::cout<<"done"<<"\n";

	// Step 2: Make the matrix symmetric by adding its transpose to itself
    std::cout<<"[INFO]: Generating the symmetric matrix...";
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_columns;
	transpose.num_rows = num_rows; 
	size = transpose.num_rows * transpose.num_columns;
	transpose.data = (double *)malloc(size * sizeof(double));

	for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			transpose.data[i * M.num_rows + j] = M.data[j * M.num_columns + i];
	// writeMatrix(transpose);

	for(i = 0; i < size; i++)
		M.data[i] += transpose.data[i];
	if (check_if_symmetric(M))
		std::cout<<"done"<<"\n";
	else{ 
        std::cout<<"error !!!"<<"\n";
		free(M.data);
		M.data = NULL;
	}
	// Step 3: Make the diagonal entries large with respect to the row and column entries
    std::cout<<"[INFO]: Generating the positive definite matrix...";
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++){
			if(i == j) 
				M.data[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if(check_if_diagonal_dominant(M))
		std::cout<<"done"<<"\n";
	else{
		std::cout<<"error !!!"<<"\n";
		free(M.data);
		M.data = NULL;
	}
	free(transpose.data);
	return M;
}

Matrix build_index_matrix(const int num_rows,const int num_columns) 
{
	unsigned int index=0;
    Matrix R= allocate_matrix(num_rows,num_columns,0);
	for(unsigned int i = 0; i < num_rows; i++){
		for(unsigned int j = 0; j < num_columns; j++) { index++;R.data[i*R.num_columns + j]=index; }
	} 
	//printf("\n");
	return R;
}


void writeMatrix(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
		{
			printf("%f ", M.data[i*M.num_columns + j]);
		}
		printf("\n");
	} 
	printf("\n");
}

void writeMatrix2(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
		{
			printf("%1.2f ", M.data[i*M.num_columns + j]);
		}
		printf("\n");
	} 
	printf("\n");
}

void saveMatrixView(const Matrix M, char *filename) 
{
    FILE* FICH = fopen(filename,"w");
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
            fprintf(FICH,"%f ", M.data[i*M.num_columns + j]);
        fprintf(FICH,"\n");
    }
    fprintf(FICH,"\n");
    fclose(FICH);
}


void saveMatrix(const Matrix M, char *filename) 
{
    std::ofstream myfile;
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
			myfile<<M.data[i*M.num_columns + j];
    }
    myfile<<"\n";
    myfile.close();
}

Matrix readMatrix(char *filename,const int num_rows,const int num_columns) 
{
	Matrix M= allocate_matrix(num_rows,num_columns,0);
	std::ifstream myfile;
	myfile.open (filename);
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
			myfile>>M.data[i*M.num_columns + j];
    }
    myfile.close();
	return M;
}

void readFileViewInformation(char *filename) 
{
	FILE* FICH = NULL;
    int c = 0;
	FICH = fopen(filename, "r");
    if (FICH != NULL) { do { c = fgetc(FICH); printf("%c",c); } while (c != EOF); fclose(FICH); }
}


int check_if_symmetric(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
		for(unsigned int j = 0; j < M.num_columns; j++)
			if(M.data[i * M.num_rows + j] != M.data[j * M.num_columns + i]) return 0;
	return 1;
}

int check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.data[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j) sum += abs(M.data[i * M.num_rows + j]);
		}
		if(diag_element <= sum) return 0;
	}
	return 1;
}

Matrix matrix_product(const Matrix A, const Matrix B) 
{
    Matrix C;
    C.num_columns = C.pitch = A.num_columns;
    C.num_rows = A.num_rows;
    unsigned int size = C.num_rows * C.num_columns;
    C.data = (double *) malloc(size * sizeof (double));

    for (unsigned int i = 0; i < A.num_columns; i++)
        for (unsigned int j = 0; j < B.num_rows; j++) {
            double sum = 0.0f;
            for (unsigned int k = 0; k < A.num_columns; k++) {
                double a = A.data[i * A.num_columns + k];
                double b = B.data[k * B.num_rows + j];
                sum += a * b;
            }
            C.data[i * B.num_rows + j] = (double) sum;
        }
    return C;
}

Matrix matrix_tanspose(const Matrix M) 
{
  Matrix R= allocate_matrix(M.num_rows,M.num_columns,0);
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.data[i * M.num_rows + j] = M.data[j * M.num_columns + i];
  return R;
}

Matrix matrix_copy(const Matrix M) 
{
  Matrix R= allocate_matrix(M.num_rows,M.num_columns,0);
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.data[i * M.num_rows + j] = M.data[i * M.num_rows + j];
  return R;
}

void matrix_copy_data(Matrix R,const Matrix M) 
{
   	int i,j;
    for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.data[i * M.num_rows + j] = M.data[i * M.num_rows + j];
}


void matrix_lower_triangular(Matrix M) 
{
    unsigned int i, j;
    for (i = 0; i < M.num_rows; i++)
        for (j = 0; j < i; j++)
            M.data[i * M.num_rows + j] = 0.0;
}


Matrix matrix_scalar(const Matrix M,double coeff) 
{
  	Matrix R= allocate_matrix(M.num_rows,M.num_columns,0);
  	unsigned int k;
  	for(k = 0; k < R.dimension; k++) { R.data[k]=coeff*M.data[k]; }
  	return R;
}


void get_matrix_min_max(double &m,double &M,Matrix R) 
{
	unsigned int k;
	m=R.data[0]; M=m;
	for(k = 0; k < R.dimension; k++) { m=std::min(R.data[k],m); M=std::max(R.data[k],M); }
}


Matrix getConvolutionCPU(Matrix M, Matrix K)
{
	Matrix R= allocate_matrix(M.num_rows,M.num_columns,0);
	int kernel_size=K.num_rows;
	float sum = 0;
  	int center = (kernel_size -1)/2;;
  	int ii, jj;
  
	/*
	for (int i = center; i<(M.num_columns-center); i++)
		for (int j = center; j<(M.num_rows-center); j++){
		sum = 0;
		for (int ki = 0; ki<kernel_size; ki++)
			for (int kj = 0; kj<kernel_size; kj++){
				ii = kj + j - center;
				jj = ki + i - center;
				sum+=M.data[jj*M.num_rows+ii]*K.data[ki*kernel_size + kj];
			}
			R.data[i*M.num_rows +j] = sum;
		}
	*/

	for (int i = 0; i<(M.num_columns); i++)
		for (int j = 0; j<(M.num_rows); j++){
		sum = 0;
		for (int ki = 0; ki<kernel_size; ki++)
			for (int kj = 0; kj<kernel_size; kj++){
				ii = kj + j - center;
				jj = ki + i - center;
				if ((ii>=0.0) && (jj>=0.0) && (ii<M.num_columns) && (jj<M.num_rows))
				{
					sum+=M.data[jj*M.num_rows+ii]*K.data[ki*kernel_size + kj];
				}
			}
			R.data[i*M.num_rows +j] = sum;
		}

	return(R);
}


Matrix kernel_Gaussian_nD(int ks, double sigma)
{
	const double pi=3.14159265359;
	Matrix K= allocate_matrix(ks,ks,0);
	double x,y, center;
	double sg2=sigma*sigma;
	center = (ks-1)/2.0;
	for (int i = 0; i<(ks*ks); i++){
		x = (double)(i%ks)-center; y = (double)(i/ks)-center;
		K.data[i] = (1.0/(2.0*pi*sg2))*exp(-0.5*(x*x+y*y)/(sg2));
	}
	return(K);
}

Matrix kernel_LaplacianOfGaussian(int ks, double sigma)
{
	const double pi=3.14159265359;
	Matrix K= allocate_matrix(ks,ks,0);
	double x,y, center;
	center = (ks-1)/2.0;
	for (int i = 0; i<(ks*ks); i++){
		x = (double)(i%ks)-center; y = (double)(i/ks)-center;
		K.data[i] = -(1.0/pi*pow(sigma,4))*(1.0 - 0.5*(x*x+y*y)/(sigma*sigma))*exp(-0.5*(x*x+y*y)/(sigma*sigma));
	}
	return(K);
}

Matrix kernel_identity(int ks)
{
	Matrix K= allocate_matrix(ks,ks,0);
	int center = (ks-1)/2.0;
	K.data[center*ks+center] = 1.0;
	return(K);
}

Matrix kernel_laplacian1()
{
	Matrix K= allocate_matrix(3,3,0);
	K.data[1] = 1;
	K.data[3] = 1; K.data[4] = -4; K.data[5] = 1;
	K.data[7] = 1;
	return(K);
}

Matrix kernel_laplacian2()
{
	Matrix K= allocate_matrix(3,3,-1);
	K.data[4] = 8; 
	return(K);
}

Matrix kernel_blur()
{
	Matrix K= allocate_matrix(3,3,1.0/9.0);
	return(K);
}

Matrix kernel_sharping()
{
	Matrix K= allocate_matrix(3,3,0);
	K.data[1] = 0.5;
	K.data[3] = 0.5; K.data[4] = 3.0; K.data[5] = 0.5;
	K.data[7] = 0.5;
	return(K);
}

Matrix kernel_sharping(double alpha)
{
	Matrix T= allocate_matrix(3,3,-alpha); T.data[4] = 9.0-alpha; 
	Matrix K=matrix_scalar(K,1.0/(9.0*(1.0-alpha))); free(T.data);
	return(K);
}


Matrix kernel_edge_kx()
{
	Matrix K= allocate_matrix(3,3,0);
	K.data[0] = -1;   K.data[2] = 1;
	K.data[3] = -2;   K.data[5] = 2;
	K.data[6] = -1;   K.data[8] = 1;
	return(K);
}

Matrix kernel_edge_ky()
{
	Matrix K= allocate_matrix(3,3,0);
	K.data[0] = -1;   K.data[1] = -2;  K.data[2] = -1;   
	K.data[6] =  1;   K.data[7] =  2;  K.data[8] =  1;
	return(K);
}

Matrix kernel_gaussian()
{
	Matrix T= allocate_matrix(3,3,0);
	T.data[0] = 1;	 T.data[1] = 2;   T.data[2] = 1;
	T.data[3] = 2;   T.data[4] = 4;   T.data[5] = 2;
	T.data[6] = 1;   T.data[8] = 2;   T.data[4] = 1;
	Matrix K=matrix_scalar(K,1.0/16.0); free(T.data);
	return(K);
}


//END:: BUILD INIT MATRIX
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
unsigned compareArrays(double *reference,double * device, int size)
{    
    for(int i=0; i<size*size; i++) {        
        float epsilon = 0.15;        
        int x = i / size;
        int y = i % size;
        if(x==y){ epsilon = 1; }        
        if (fabs(reference[i] - device[i]) > epsilon) {
            printf("\ni=%d : reference=%f  !=  device=%f   | x=%d y=%d   \n" , i, reference[i], device[i], x, y);
            return 0;
        }
    }
    return 1;
}

void checkSolution(Matrix MatRef,Matrix MatRes)
{
    unsigned res = compareArrays(MatRef.data, MatRes.data,MatRef.num_rows);
    printf("[INFO]:	%s\n", (1 == res) ? "WELL DONE PASSED :-)" : "FAILED");
}


bool isFileExist(std::string ch)
{
    std::ifstream myfile;
    myfile.open(ch); bool qOK=false;
    if (myfile) { qOK=true; }
    myfile.close();
    return (qOK);
}


void distributeVectorData(std::vector<int>& dataset, int numGPUs) {
    int dataSize = dataset.size();
    int chunkSize = dataSize / numGPUs; 
    for (int deviceId = 0; deviceId < numGPUs; ++deviceId) {
        hipSetDevice(deviceId);
        // Calculate the start and end indices for the chunk assigned to this GPU
        int startIndex = deviceId * chunkSize;
        int endIndex = (deviceId == numGPUs - 1) ? dataSize : (deviceId + 1) * chunkSize;
        // Copy the corresponding chunk of data to the GPU
        int* gpuData; hipMalloc((void**)&gpuData, sizeof(int) * (endIndex - startIndex));
        hipMemcpy(gpuData, &dataset[startIndex], sizeof(int) * (endIndex - startIndex), hipMemcpyHostToDevice);
        std::cout << "Data for GPU " << deviceId << " transferred successfully." << "\n";
        // ...
        hipFree(gpuData);
    }
}

void launchParallelComputations(int* gpuData, int startIndex, int endIndex) {
    int blockSize = 256;
    int numBlocks = (endIndex - startIndex + blockSize - 1) / blockSize;
	hipLaunchKernelGGL(parallelComputation,numBlocks, blockSize,0,0,gpuData, startIndex, endIndex); 
    hipDeviceSynchronize(); 
}

void synchronizeAndAggregate(int* gpuData, int dataSize, int numGPUs) {
    for (int deviceId = 0; deviceId < numGPUs; ++deviceId) {
        hipSetDevice(deviceId);
        hipDeviceSynchronize(); 
        std::cout << "[INFO]: GPU " << deviceId << " synchronization completed." << "\n";
    }
    int sum = 0; for (int i = 0; i < dataSize; ++i) { sum += gpuData[i]; }
    std::cout << "[INFO]: Aggregate result: " << sum << "\n";
}


/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
//BEGIN::Product Matrix and ...

Matrix matrix_product_GPU(const Matrix A, const Matrix B) 
{
	int block_size = 512;
	int matrixSize=A.num_columns;
    Matrix C= allocate_matrix(matrixSize,matrixSize,0);

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_B = allocate_matrix_on_gpu(B);
	Matrix gpu_C = allocate_matrix_on_gpu(C);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_B, B );
	copy_matrix_to_device(gpu_C, C );
	
	int num_blocks = (matrixSize*matrixSize + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_mult,grid, thread_block,0,0,gpu_C.data,gpu_A.data,gpu_B.data,matrixSize,matrixSize); 

	copy_matrix_from_device(C,gpu_C);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_B.data);
	hipFree(gpu_C.data);

	return C;
}


Matrix matrix_copy_GPU(const Matrix A) 
{
	int block_size = 512;
    //Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);
	Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_R = allocate_matrix_on_gpu(R);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_R, R );
	
	int num_blocks = (A.num_columns*A.num_rows + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_copy,grid, thread_block,0,0,gpu_R.data,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(R,gpu_R);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_R.data);
	return R;
}


Matrix matrix_transpose_GPU(const Matrix A) 
{
	int block_size = 512;
    //Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);
	Matrix R= allocate_matrix(A.num_columns,A.num_rows,0);

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_R = allocate_matrix_on_gpu(R);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_R, R );
	
	int num_blocks = (A.num_columns*A.num_rows + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_transpose,grid, thread_block,0,0,gpu_R.data,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(R,gpu_R);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_R.data);
	return R;
}

Matrix matrix_lower_triangular_GPU(const Matrix A) 
{
	int block_size = 512;
	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	
	int num_blocks = (A.dimension + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_lower_triangular,grid, thread_block,0,0,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(A,gpu_A);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	return A;
}



bool is_matrix_equal_GPU(const Matrix A, const Matrix B,const double deltaError) 
{
	int block_size = 512;
	int matrixSize=A.num_columns;
	int sizeQ = sizeof(bool) * 1;
    bool *h_Q = (bool *)malloc(sizeQ);
	h_Q[0]=true;

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_B = allocate_matrix_on_gpu(B);
	bool  *d_Q;    hipMalloc((void **)&d_Q,sizeQ);

	hipEventRecord(start, 0);   
	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_B, B );
	hipMemcpy(d_Q,h_Q,sizeQ, hipMemcpyHostToDevice);
	int num_blocks = (matrixSize*matrixSize + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);
	hipLaunchKernelGGL(matrix_equal,grid, thread_block,0,0,d_Q,gpu_A.data,gpu_B.data,matrixSize*matrixSize,deltaError); 
	hipMemcpy(h_Q,d_Q,sizeof(bool), hipMemcpyDeviceToHost);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_B.data);
	hipFree(d_Q);

	return (h_Q[0]);
}

bool is_matrix_equal_GPU(const Matrix A, const Matrix B) 
{
	double deltaError=0.000001;
	return(is_matrix_equal_GPU(A,B,deltaError));
}


void checkSolution_GPU(Matrix A,Matrix B)
{
	bool res=is_matrix_equal_GPU(A,B);
	printf("[INFO]:	%s\n", (true == res) ? "WELL DONE PASSED :-)" : "FAILED");
}

//END::Product Matrix
/*********************************************************************************************************************************************************/



/*********************************************************************************************************************************************************/
//BEGIN::Cholesky Factoristation

Matrix getCholeskySerial(Matrix A)
{
  int i,j,k;
  int n = A.num_rows;
  Matrix L = allocate_matrix(n,n,0);
  for (i = 0; i < n; i++)
      for (j = 0; j < (i+1); j++) {
            double s = 0;
            for (k = 0; k < j; k++)
                s += L.data[i * n + k] * L.data[j * n + k];
            L.data[i * n + j] = (i == j) ? sqrt(A.data[i * n + i] - s) : (1.0 / L.data[j * n + j] * (A.data[i * n + j] - s));
  }
  Matrix U=matrix_tanspose(L);
  return U;
}

#ifdef UseOpenMP
Matrix getCholeskyOpenMPVers1(Matrix A,int num_threads)
{
	omp_set_num_threads(num_threads);
	double t1, t2;
	int k, i, j, l;
	int n = A.num_rows;
	Matrix U=matrix_copy(A);
	double c;
	t1 = omp_get_wtime();
	for(k=0; k<n; k++)
	{
		c=sqrt(U.data[k*n+k]);  U.data[k*n+k]=c;
		#pragma omp parallel for
		for(i=k+1; i<=n; i++) {
			U.data[i+k*n]=U.data[i+k*n]/c;
		}
		#pragma omp parallel for private(l,j)
		for (l=k+1; l<n; l++) {
			for (j=k+1; j<=l-1; j++) {
				U.data[l+j*n]=U.data[l+j*n]-U.data[l+k*n]*U.data[j+k*n];
			}
			U.data[l*n+l]=U.data[l*n+l]-U.data[l+k*n]*U.data[l+k*n];
		}
	}
	t2 = omp_get_wtime();
	//std::cout << "[INFO]: Elapsed microseconds inside: "<<t2-t1<< " us\n";
	matrix_lower_triangular(U);
	return U;
}

Matrix getCholeskyOpenMPVers2(Matrix A,int num_threads)
{
	omp_set_num_threads(num_threads);
	double t1, t2;
	int k, i, j, l;
	int n = A.num_rows;
	Matrix U=matrix_copy(A);
	double c;
	t1 = omp_get_wtime();
	for(k=0; k<n; k++)
	{
		c=sqrt(U.data[k*n+k]); U.data[k*n+k]=c;
		for(i=k+1; i<=n; i++) {
			#pragma omp task
			U.data[i+k*n]=U.data[i+k*n]/c;
		}

		#pragma omp taskwait
		for (l=k+1; l<=n; l++) {
			for (j=k+1; j<=l-1; j++) {
				#pragma omp task
				U.data[l+j*n]=U.data[l+j*n]-U.data[l+k*n]*U.data[j+k*n];
			}
			#pragma omp task
			U.data[l*n+l]=U.data[l*n+l]-U.data[l+k*n]*U.data[l+k*n];
		}
	}
	matrix_lower_triangular(U);
	t2 = omp_get_wtime();
	//std::cout << "[INFO]: Elapsed microseconds inside: "<<t2-t1<< " us\n";
	return U;
}
#endif


#ifdef USE_MPI
void getCholeskyMPIVers2(int argc, char *argv[])
{
	int n=atoi(argv[1]);
    Matrix MatA=allocate_matrix(n,n,0); 
	Matrix MatM=allocate_matrix(n,n,0); 

	//Matrix MatA;
	//Matrix MatM;

    bool qView=true;

	std::chrono::steady_clock::time_point t_begin,t_end;
	long int t_laps;

	int world_rank_mpi,world_size_mpi;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_mpi);
	
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    if (world_rank_mpi == 0) { 
      int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      std::cout << "[INFO]: Name worlds processor: "<<processor_name<<"\n";
      std::cout << "[INFO]: Nb CPU available: "<<numCPU<< "\n";
      std::cout << "\n";
      std::cout << "[INFO]: Scan..."<<"\n";
    }
    std::cout << "[INFO]: rank: "<<world_rank_mpi<<" out of "<<world_size_mpi<<"\n";
   
    MPI_Barrier(MPI_COMM_WORLD); 

    if (world_rank_mpi == 0) { 
      std::cout << "\n";
      MatA = build_init_matrix(n,n); 
	  matrix_copy_data(MatM,MatA);	
      if (qView) { writeMatrix(MatA);	}
      t_begin = std::chrono::steady_clock::now();
    }

	MPI_Barrier(MPI_COMM_WORLD); 


    for (int k=0; k<n; k++)
    {		  
      if (world_rank_mpi == 0){
        for (int j=0; j<k; j++)
        {
          MatM.data[k * n + k]-=(MatM.data[k * n + j] * MatM.data[k * n + j]);	
        }
        MatM.data[k * n + k]=sqrt(MatM.data[k * n + k]);
        
        for (int p=1; p<n-k; p++) 
        {	
          for (int c=0; c<=k; c++)
          {
            for (int r=c; r<n; r++)
            {	
              MPI_Send(&MatM.data[r * n + c], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
          }
          MPI_Recv(&MatM.data[( p + k )* n + k], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }	
        if (k==n-1)
        {	
          for(int i=0; i<n-1; i++)
          {
            for(int j=i+1; j<n; j++)
            {
              MatM.data[i * n + j]=0;	
            }	
          }
        }	
      }

      else if (world_rank_mpi<n-k){
        for (int c=0; c<=k; c++) 
        {
          for (int r=c; r<n; r++)
          {
            MPI_Recv(&MatM.data[r * n + c], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
        
        for (int g=0; g<k; g++) //calculating non-diagonal data concurrently
        {
          MatM.data[(k+world_rank_mpi) * n + k]-=MatM.data[k * n + g] * MatM.data[(k+world_rank_mpi) * n + g];		
        }	
        MatM.data[(k+world_rank_mpi) * n + k]/=MatM.data[k * n + k];
        MPI_Send(&MatM.data[(k+world_rank_mpi) * n + k], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }   
    }	

	if (world_rank_mpi == 0) {
		t_end = std::chrono::steady_clock::now();
    	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
        std::cout << "\n";
	    std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
        std::cout << "\n";

		std::string chName="DataMPI.csv";
        std::ofstream myfile;
        if (!isFileExist(chName)) {  myfile.open (chName); } else { myfile.open(chName,std::ios::app); }
        myfile<<world_size_mpi<<","<<n<<","<<t_laps<<"\n";
        myfile.close();
		
		if (qView) { 
            printf("[INFO]: Cholesky decomposition of matrix\n");
            if (qView) { writeMatrix(MatM); }
            if (1==1) {
                  std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
                  Matrix MatMt=matrix_tanspose(MatM);
                  //Matrix MatT=matrix_product(MatM,MatLMt);
				  Matrix MatT=matrix_product_GPU(MatM,MatMt); 
                  if (qView) { writeMatrix(MatT); }
                  //checkSolution(MatA,MatT);
				  checkSolution_GPU(MatA,MatT);
                  free(MatMt.data);
                  free(MatT.data);
            }
            
          }
		  
		free(MatM.data);
	}
    MPI_Finalize();
}
#endif


Matrix getCholeskyGPUVers1(Matrix A)
{
	int matrixSize=A.num_rows;
	Matrix U= allocate_matrix(matrixSize,matrixSize,0);
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	int num_blocks = 1;
	int threads_per_block = 512;
	int ops_per_thread = matrixSize/(threads_per_block * num_blocks);
	hipEventRecord(start, 0);
	Matrix gpu_u = allocate_matrix_on_gpu(U);

	copy_matrix_to_device(gpu_u, A );
	dim3 thread_block(threads_per_block, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(chol_kernel,grid, thread_block,0,0,gpu_u.data,ops_per_thread,matrixSize); 

	const unsigned int block_size_2 = 256;
    const unsigned int num_blocks_2 = (A.dimension+ block_size_2 - 1) / block_size_2;
    dim3 thread_block_2(block_size_2, 1, 1);
    dim3 grid_2(num_blocks_2, 1);
	hipLaunchKernelGGL(matrix_lower_triangular,grid_2, thread_block_2,0,0,gpu_u.data,matrixSize,matrixSize);

	hipDeviceSynchronize();
	copy_matrix_from_device(U,gpu_u);
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipFree(gpu_u.data);
  
	return U;
}

Matrix getCholeskyGPUVers2(Matrix A)
{
	int matrixSize=A.num_rows;
    Matrix U= allocate_matrix(matrixSize,matrixSize,0);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    int threads_per_block = 256; 
    int stride = threads_per_block;    
    hipEventRecord(start, 0);    
    Matrix gpu_u = allocate_matrix_on_gpu(U);
    copy_matrix_to_device(gpu_u, A);
    int k;
    for (k = 0; k < matrixSize; k++) {
        int isize = (matrixSize - 1) - (k + 1) + 1;
        int num_blocks = isize;
        if (num_blocks <= 0) { num_blocks = 1; }
        dim3 thread_block(threads_per_block, 1, 1);
        dim3 grid(num_blocks, 1);
        hipLaunchKernelGGL(chol_kernel_optimized_div,grid, thread_block,0,0,gpu_u.data,k,stride,matrixSize); 
        hipLaunchKernelGGL(chol_kernel_optimized,grid, thread_block,0,0,gpu_u.data,k,stride,matrixSize); 
    }


	const unsigned int block_size_2 = 256;
    const unsigned int num_blocks_2 = (A.dimension+ block_size_2 - 1) / block_size_2;
    dim3 thread_block_2(block_size_2, 1, 1);
    dim3 grid_2(num_blocks_2, 1);
	hipLaunchKernelGGL(matrix_lower_triangular,grid_2, thread_block_2,0,0,gpu_u.data,matrixSize,matrixSize);


    copy_matrix_from_device(U, gpu_u);  				 
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipFree(gpu_u.data);

	//matrix_lower_triangular(U);
		
    return U;
}


Matrix getCholeskyGPUVers3(Matrix A,int threads_per_block)
{
	int matrixSize=A.num_rows;
    Matrix U= allocate_matrix(matrixSize,matrixSize,0);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    //int threads_per_block = 256; 
    int stride = threads_per_block;    
    hipEventRecord(start, 0);    
    Matrix gpu_u = allocate_matrix_on_gpu(U);
    copy_matrix_to_device(gpu_u, A);
    int k;
    for (k = 0; k < matrixSize; k++) {
        int isize = (matrixSize - 1) - (k + 1) + 1;
        int num_blocks = isize;
        if (num_blocks <= 0) { num_blocks = 1; }
        dim3 thread_block(threads_per_block, 1, 1);
        dim3 grid(num_blocks, 1);
        hipLaunchKernelGGL(chol_kernel_optimized_div,grid, thread_block,0,0,gpu_u.data,k,stride,matrixSize); 
        hipLaunchKernelGGL(chol_kernel_optimized,grid, thread_block,0,0,gpu_u.data,k,stride,matrixSize); 
    }

	const unsigned int block_size_2 = threads_per_block;
    const unsigned int num_blocks_2 = (A.dimension+ block_size_2 - 1) / block_size_2;
    dim3 thread_block_2(block_size_2, 1, 1);
    dim3 grid_2(num_blocks_2, 1);
	hipLaunchKernelGGL(matrix_lower_triangular,grid_2, thread_block_2,0,0,gpu_u.data,matrixSize,matrixSize);

    copy_matrix_from_device(U, gpu_u);  				 
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipFree(gpu_u.data);

	//matrix_lower_triangular(U);

    return U;
}

//END::Cholesky Factoristation
/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
//BEGIN::Convolution Matrix

Matrix getConvolutionGPUVers1(Matrix A,Matrix K)
{
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	float milliseconds = 0;
	int block_size = 512;
    //Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);
	Matrix R= allocate_matrix(A.num_columns,A.num_rows,0);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_K = allocate_matrix_on_gpu(K);
	Matrix gpu_R = allocate_matrix_on_gpu(R);
	 
	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_K, K );
	copy_matrix_to_device(gpu_R, R );

	int nx=A.num_columns;
	int ny=A.num_rows;

	int kernel_size = K.num_columns;
	int Nblocks  = ny - (kernel_size-1);
    int Nthreads = nx - (kernel_size-1);	

	hipEventRecord(start);
	hipLaunchKernelGGL(convolution1DKernel,Nblocks, Nthreads, kernel_size*kernel_size*sizeof(double),0,
		gpu_R.data,
		gpu_A.data,
		gpu_K.data,
		nx,ny,kernel_size);
	hipDeviceSynchronize();
	hipEventRecord(stop);
	hipEventElapsedTime(&milliseconds, start, stop);
	copy_matrix_from_device(R,gpu_R);
	hipFree(gpu_A.data);
	hipFree(gpu_K.data);
	hipFree(gpu_R.data);

  	//std::cout<<"[INFO]: Convolution Completed !!! \n";
  	//std::cout<<"[INFO]: Ellapsed Time (GPU): "<<milliseconds<<" ms\n";
  	//std::cout<<"\n";

	return R;
}

Matrix getConvolutionGPUVers2(Matrix A,Matrix K)
{
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	float milliseconds = 0;
	int block_size = 512;
    //Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);
	Matrix R= allocate_matrix(A.num_columns,A.num_rows,0);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_K = allocate_matrix_on_gpu(K);
	Matrix gpu_R = allocate_matrix_on_gpu(R);


	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_K, K );
	copy_matrix_to_device(gpu_R, R );

	dim3 blockSize(16, 16);
    dim3 gridSize((A.num_columns + blockSize.x - 1) / blockSize.x, (A.num_rows + blockSize.y - 1) / blockSize.y);

	hipEventRecord(start);
	hipLaunchKernelGGL(convolution2DKernel,gridSize, blockSize, 0,0,
		gpu_A.data,gpu_K.data,gpu_R.data,
		A.num_columns,A.num_rows,
		K.num_columns,K.num_rows);
	hipDeviceSynchronize();
	hipEventRecord(stop);
	hipEventElapsedTime(&milliseconds, start, stop);
	copy_matrix_from_device(R,gpu_R);
	hipFree(gpu_A.data);
	hipFree(gpu_K.data);
	hipFree(gpu_R.data);

  	//std::cout<<"[INFO]: Convolution Completed !!! \n";
  	//std::cout<<"[INFO]: Ellapsed Time (GPU): "<<milliseconds<<" ms\n";
  	//std::cout<<"\n";

	return R;
}
//END::Convolution Matrix
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::pthread Cholesky Factoristation

typedef struct chol_pthread_args
{
	Matrix A,U;
	int copyi_start,copyi_end;
	int zeroi_start,zeroi_end;
	pthread_barrier_t * barrier;
	int id;
    int nbTh;
} chol_pthread_args;


//Range splitter helper function
void range_splitter(int size, int num_threads, int * items_per_thread, int * items_last_thread)
{
	//Divide up total size by number of threads
	//How many are left over?
	int elems_left_over = size%num_threads;
	int data_per_thread = size/num_threads;
	int last_thread_data = data_per_thread;

	if(elems_left_over !=0)
	{
		last_thread_data = data_per_thread+elems_left_over;
	}

	//Double check because math is hard
	if( (((num_threads-1)*data_per_thread) + last_thread_data) != size || (last_thread_data<0))
	{
		printf("AH! MATH! threads:%d dataperthread:%d lastthreadelm:%d size:%d leftover:%d\n", num_threads,data_per_thread,last_thread_data,size,elems_left_over);
		exit(-1);
	}
	*items_per_thread = data_per_thread;
	*items_last_thread = last_thread_data;
}


void range_maker(int items_per_thread,int items_last_thread, int num_threads, int index, int offset, int is_last_thread, int * start, int * end)
{
    *start=items_per_thread*index + offset;
	if (is_last_thread==1) { *end = *start + items_last_thread;   }
	else                   { *end = *start + items_per_thread -1; }
}

void populate_thread_args(chol_pthread_args * arg_list,Matrix A, Matrix U, pthread_barrier_t * barrier,int NbThread)
{
	//Matrix size
	unsigned int size = A.num_rows * A.num_columns;

	//Copy
	int copyisize = size;
	int copyi_items_per_thread, copyi_items_last_thread;
	range_splitter(copyisize,NbThread, &copyi_items_per_thread, &copyi_items_last_thread);

	//Zero out
	int zeroisize = U.num_rows;
	int zeroi_items_per_thread, zeroi_items_last_thread;
	range_splitter(zeroisize,NbThread, &zeroi_items_per_thread, &zeroi_items_last_thread);

	//Zero offset for both sets of work
	int offset = 0;

	//Loop through threads
	int i;
	for(i=0;i<NbThread; i++)
	{
		//Easy ones for all threads
		arg_list[i].A=A;
		arg_list[i].U=U;
		arg_list[i].barrier = barrier;
		arg_list[i].id = i;
        arg_list[i].nbTh = NbThread;

		if(i == (NbThread-1))
		{
			//Last thread
			range_maker(copyi_items_per_thread,copyi_items_last_thread,NbThread, i, offset,1, &(arg_list[i].copyi_start), &(arg_list[i].copyi_end));
			range_maker(zeroi_items_per_thread,zeroi_items_last_thread,NbThread, i, offset,1, &(arg_list[i].zeroi_start), &(arg_list[i].zeroi_end));
		}
		else
		{
			//Regular threads
			range_maker(copyi_items_per_thread,copyi_items_last_thread,NbThread, i, offset,0, &(arg_list[i].copyi_start), &(arg_list[i].copyi_end));
			range_maker(zeroi_items_per_thread,zeroi_items_last_thread,NbThread, i, offset,0, &(arg_list[i].zeroi_start), &(arg_list[i].zeroi_end));
		}
	}
}

void sync_pthreads(pthread_barrier_t * barrier, int thread_id)
{
	// Synchronization point
    int rc = pthread_barrier_wait(barrier);
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier.\n");
        exit(-1);
    }
    //std::cout<<"sync_pthreads\n";
}


void * chol_pthread(void * arg)
{
    //std::cout<<"BEGIN::chol_pthread\n";
	//Get arg as struct
	chol_pthread_args * args = (chol_pthread_args *)arg;
	//Matrices
	Matrix A = args->A;
	Matrix U = args->U;
    int NbThread = args->nbTh;

	//Copy work
	int copyi_start = args->copyi_start;
	int copyi_end = args->copyi_end;
	//Zero out work
	int zeroi_start = args->zeroi_start;
	int zeroi_end = args->zeroi_end;
	//Barrier to sync
	pthread_barrier_t * barrier = args->barrier;
	//Id
	int id = args->id;

    if (id == NbThread - 1) {
        zeroi_end--;
        copyi_end--;
    }


	//Iterators
	unsigned int i, j, k;
	unsigned int size = A.num_rows * A.num_columns;

	//Copy the contents of the A matrix into the working matrix U
	for (i = copyi_start; i <= copyi_end; i ++)
	{
		U.data[i] = A.data[i];
	}

	//Sync threads!!!
	sync_pthreads(barrier, id);

	// Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++)
	{
		//Only one thread does squre root and division
		if(id==0)
		{
			// Take the square root of the diagonal element
			U.data[k * U.num_rows + k] = sqrt(U.data[k * U.num_rows + k]);
			if(U.data[k * U.num_rows + k] <= 0){
					 printf("Cholesky decomposition failed. \n");
					 return 0;
			}

			// Division step
			for(j = (k + 1); j < U.num_rows; j++)
			{
				U.data[k * U.num_rows + j] /= U.data[k * U.num_rows + k]; // Division step
			}
		}

		//Sync threads!!!!!
		sync_pthreads(barrier, id);

		//For this k iteration, split up i
		//Size of i range originally
		int isize = U.num_rows - (k + 1);
		int items_per_thread, items_last_thread;
		range_splitter(isize,NbThread, &items_per_thread, &items_last_thread);
		//Divy up work
		//Elim work
		int elimi_start, elimi_end;
		int offset = (k + 1); //To account for not starting at i=0 each time

		elimi_start=items_per_thread*id + offset;
		if(id == (NbThread-1))
		{
			//Last thread
			elimi_end = elimi_start + items_last_thread;
		}
		else
		{
			//Regular threads
			elimi_end = elimi_start + items_per_thread -1 ;
		}

		// Elimination step
		//printf("K Loop. I range %d to %d\n",(k + 1),U.num_rows-1);
		for(i = elimi_start; i <=elimi_end; i++)
		{
			for(j = i; j < U.num_rows; j++)
			{
				U.data[i * U.num_rows + j] -= U.data[k * U.num_rows + i] * U.data[k * U.num_rows + j];
			}
		}

		//Sync threads!!!!!
		sync_pthreads(barrier, id);
	}

	//Sync threads!!!!!
	sync_pthreads(barrier, id);

	// As the final step, zero out the lower triangular portion of U
	for(i = zeroi_start; i <=zeroi_end; i++)
	{
		for(j = 0; j < i; j++)
		{
			U.data[i * U.num_rows + j] = 0.0;
		}
	}
	//Don't sync, will join outside here
    //writeMatrix(U);
    //std::cout<<"END::chol_pthread\n";
    return NULL;
}


Matrix getCholesky_pthreads(const Matrix A,int NbThread)
{
	int matrixSize=A.num_rows;
    int i;
    Matrix U= allocate_matrix(matrixSize,matrixSize,0);
	pthread_t threads[NbThread];
	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, NULL,NbThread);
	chol_pthread_args args[NbThread];
	populate_thread_args(&args[0],A,U,&barrier,NbThread);
    for(i=0; i < NbThread; i++) { 
        int status=pthread_create(&threads[i],NULL,chol_pthread,&(args[i])); 
        if(status != 0) { fprintf(stderr, "pthread_create failed with i = %d. errno = %d, %s\n",i, errno, strerror(errno)); }
    }
    for(i=0; i < NbThread; i++) { pthread_join(threads[i],NULL); }
    return U;
}

//END::pthread Cholesky Factoristation
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::Tools save data to image 0-255

namespace intarray2bmp
{
	struct lwrite
	{
		unsigned long value;
		unsigned      size;
		lwrite( unsigned long value, unsigned size ):
		value( value ), size( size )
		{ }
	};

	//--------------------------------------------------------------------------
	inline std::ostream& operator << ( std::ostream& outs, const lwrite& v )
	{
		unsigned long value = v.value;
		for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
		outs.put( static_cast <char> (value & 0xFF) );
		return outs;
	}


	template <typename IntType>
	bool intarray2bmp(
		const std::string& filename,
		IntType**          intarray,
		unsigned           rows,
		unsigned           columns,
		IntType            min_value,
		IntType            max_value
		) {
		// This is the difference between each color based upon
		// the number of distinct values in the input array.
		double granularity = 360.0 / ((double)( max_value - min_value ) + 1);

		// Open the output BMP file
		std::ofstream f( filename.c_str(),
						std::ios::out | std::ios::trunc | std::ios::binary );
		if (!f) return false;

		// Some basic
		unsigned long headers_size    = 14  // sizeof( BITMAPFILEHEADER )
									+ 40; // sizeof( BITMAPINFOHEADER )
		unsigned long padding_size    = (4 - ((columns * 3) % 4)) % 4;
		unsigned long pixel_data_size = rows * ((columns * 3) + padding_size);

		// Write the BITMAPFILEHEADER
		f.put( 'B' ).put( 'M' );                           // bfType
		f << lwrite( headers_size + pixel_data_size, 4 );  // bfSize
		f << lwrite( 0,                              2 );  // bfReserved1
		f << lwrite( 0,                              2 );  // bfReserved2
		f << lwrite( headers_size,                   4 );  // bfOffBits

		// Write the BITMAPINFOHEADER
		f << lwrite( 40,                             4 );  // biSize
		f << lwrite( columns,                        4 );  // biWidth
		f << lwrite( rows,                           4 );  // biHeight
		f << lwrite( 1,                              2 );  // biPlanes
		f << lwrite( 24,                             2 );  // biBitCount
		f << lwrite( 0,                              4 );  // biCompression=BI_RGB
		f << lwrite( pixel_data_size,                4 );  // biSizeImage
		f << lwrite( 0,                              4 );  // biXPelsPerMeter
		f << lwrite( 0,                              4 );  // biYPelsPerMeter
		f << lwrite( 0,                              4 );  // biClrUsed
		f << lwrite( 0,                              4 );  // biClrImportant

		// Write the pixel data
		for (unsigned row = rows; row; row--)           // bottom-to-top
		{
		for (unsigned col = 0; col < columns; col++)  // left-to-right
			{
			unsigned char red, green, blue;
			//
			// This is how we convert an integer value to a color:
			// by mapping it evenly along the CIECAM02 hue color domain.
			//
			// http://en.wikipedia.org/wiki/Hue
			// http://en.wikipedia.org/wiki/hsl_and_hsv#conversion_from_hsv_to_rgb
			//
			// The following algorithm takes a few shortcuts since
			// both 'value' and 'saturation' are always 1.0.
			//
			double hue = (intarray[ row - 1 ][ col ] - min_value) * granularity;
			int    H = (int)( hue / 60 ) % 6;
			double F = (hue / 60) - H;
			double Q = 1.0 - F;

			#define c( x ) (255 * x)
			switch (H)
			{
			case 0:  red = c(1);  green = c(F);  blue = c(0);  break;
			case 1:  red = c(Q);  green = c(1);  blue = c(0);  break;
			case 2:  red = c(0);  green = c(1);  blue = c(F);  break;
			case 3:  red = c(0);  green = c(Q);  blue = c(1);  break;
			case 4:  red = c(F);  green = c(0);  blue = c(1);  break;
			default: red = c(1);  green = c(0);  blue = c(Q);
			}
			#undef c

			f.put( static_cast <char> (blue)  )
			.put( static_cast <char> (green) )
			.put( static_cast <char> (red)   );
			}

		if (padding_size) f << lwrite( 0, padding_size );
		}

		// All done!
		return f.good();
		}

	template <typename IntType>
	bool intarray2bmp(
		const std::string& filename,
		IntType*           intarray,
		unsigned           rows,
		unsigned           columns,
		IntType            min_value,
		IntType            max_value
		)
	{
		IntType** ia = new( std::nothrow ) IntType* [ rows ];
		for (unsigned row = 0; row < rows; row++)
		{
		ia[ row ] = intarray + (row * columns);
		}
		bool result = intarray2bmp(
						filename, ia, rows, columns, min_value, max_value
						);
		delete [] ia;
		return result;
	}

} 


typedef unsigned int uint;
int* read_bmp(const std::string filename, uint& width, uint& height) {
    std::ifstream file(filename, std::ios::in|std::ios::binary);
    uint w=0, h=0;
    char header[54];
    file.read(header, 54);
    for(uint i=0; i<4; i++) {
        w |= (header[18+i]&255)<<(8*i);
        h |= (header[22+i]&255)<<(8*i);
    }
    const int pad=(4-(3*w)%4)%4, imgsize=(3*w+pad)*h;
    char* img = new char[imgsize];
    file.read(img, imgsize);
    file.close();
    int* data = new int[w*h];
    for(uint y=0; y<h; y++) {
        for(uint x=0; x<w; x++) {
            const int i = 3*x+y*(3*w+pad);
            data[x+(h-1-y)*w] = (img[i]&255)|(img[i+1]&255)<<8|(img[i+2]&255)<<16;
        }
    }
    delete[] img;
    width = w;
    height = h;
    return data;
}


void write_bmp(const std::string filename, const uint width, const uint height, const int* const data)
 {
    const int pad=(4-(3*width)%4)%4, filesize=54+(3*width+pad)*height; // horizontal line must be a multiple of 4 bytes long, header is 54 bytes
    char header[54] = { 'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0, 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,24,0 };
    for(uint i=0; i<4; i++) {
        header[ 2+i] = (char)((filesize>>(8*i))&255);
        header[18+i] = (char)((width   >>(8*i))&255);
        header[22+i] = (char)((height  >>(8*i))&255);
    }
    char* img = new char[filesize];
    for(uint i=0; i<54; i++) img[i] = header[i];
    for(uint y=0; y<height; y++) {
        for(uint x=0; x<width; x++) {
            const int color = data[x+(height-1-y)*width];
            const int i = 54+3*x+y*(3*width+pad);
            img[i  ] = (char)( color     &255);
			img[i+1] = (char)( color     &255);
			img[i+2] = (char)( color     &255);
            //img[i+1] = (char)((color>> 8)&255);
            //img[i+2] = (char)((color>>16)&255);
        }
        for(uint p=0; p<pad; p++) img[54+(3*width+p)+y*(3*width+pad)] = 0;
    }
    std::ofstream file(filename, std::ios::out|std::ios::binary);
    file.write(img, filesize);
    file.close();
    delete[] img;
}



void saveMatrixToImageBMP2(const std::string name,Matrix A)
{
	bool qView=true;
	//qView=false;
	int* img; img = (int *)malloc(A.dimensionSizeof);
	double min,max; get_matrix_min_max(min,max,A);

	//std::cout << "min ="<<min<<"\n";
	//std::cout << "max ="<<max<<"\n";

	int i, j;
    for (i = 0; i < A.num_rows; i++)
	{
        for(j = 0; j < A.num_columns; j++)
		{
		   double v=A.data[i*A.num_columns + j];
           img[i*A.num_columns + j] = int(std::round(((v-min)/(max-min))*255.0));
		   if (v==0.0) {  img[i*A.num_columns + j] = 0; }
		   if (qView) { printf("%i ",img[i*A.num_columns + j]); }
		}
		if (qView) { printf("\n"); }
	}
	if (qView) { printf("\n"); }

	write_bmp(name, A.num_rows,A.num_columns,img);

	std::cout << "[INFO]: Save IMG Matrix"<<"\n";
}




void saveMatrixToImageBMP(const std::string name,Matrix A)
{
	bool qView=true;
	//qView=false;
	int* img; img = (int *)malloc(A.dimensionSizeof);
	double min,max; get_matrix_min_max(min,max,A);

	//std::cout << "min ="<<min<<"\n";
	//std::cout << "max ="<<max<<"\n";

	int i, j;
    for (i = 0; i < A.num_rows; i++)
	{
        for(j = 0; j < A.num_columns; j++)
		{
		   double v=A.data[i*A.num_columns + j];
           img[i*A.num_columns + j] = int(std::round(((v-min)/(max-min))*255.0));
		   if (v==0.0) {  img[i*A.num_columns + j] = min; }
		   if (qView) { printf("%i ",img[i*A.num_columns + j]); }
		}
		if (qView) { printf("\n"); }
	}
	if (qView) { printf("\n"); }

	bool result = intarray2bmp::intarray2bmp(name,img,A.num_rows,A.num_columns,0,9);
	std::cout << "[INFO]: Save IMG Matrix ="<<result<<"\n";
}


void saveMatrixToImagePGM(const std::string name,Matrix A)
{
	double min,max; get_matrix_min_max(min,max,A);
	typedef unsigned char pixval_t;
	auto float_to_pixval = [](float img_val) -> pixval_t {
		int tmpval = static_cast<int>(::std::floor(256 * img_val));
		if (tmpval < 0) {
			return 0u;
		} else if (tmpval > 255) {
			return 255u;
		} else {
			return tmpval & 0xffu;
		}
	};

	std::ofstream out(name, std::ios::binary | std::ios::out | std::ios::trunc);
	//out << "P5\n512 512\n255\n";
	out << "P5\n"<<A.num_rows<<" "<<A.num_columns<<"\n255\n";
	int i, j,v;
    for (i = 0; i < A.num_rows; i++)
        for(j = 0; j < A.num_columns; j++)
		{
    		v = int(std::round(((A.data[i*A.num_columns + j]-min)/(max-min))*255.0));
			const pixval_t pixval = float_to_pixval(v);
			const char outpv = static_cast<const char>(pixval);
			out.write(&outpv, 1);
		}
}

//END::Tools save data to image 0-255
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
//BEGIN::mandelbro demo

void mandelbrot(unsigned int w,unsigned h)
{
	int N=h*w;
    int *pos = (int *)malloc(h*w*sizeof(int));
    int *d_pgmData;
    hipMalloc((void **)&d_pgmData, sizeof(int)*w*h);
    hipMemcpy(d_pgmData, pos ,h*w*sizeof(int) ,hipMemcpyHostToDevice);
    dim3 block_size(16, 16);
    dim3 grid_size((N)/block_size.x, (int) N / block_size.y);
    hipLaunchKernelGGL(calculat_mandelbrot,grid_size,block_size,0,0,d_pgmData,512,512,1000);
    hipMemcpy(pos,d_pgmData,h*w*sizeof(int) ,hipMemcpyDeviceToHost);
    hipFree(d_pgmData);

	for(unsigned int i = 0; i < h; i++){
		for(unsigned int j = 0; j < w; j++)
		{
			printf("%i ",pos[i*w + j]);
		}
		printf("\n");
	} 
	printf("\n");

	bool result = intarray2bmp::intarray2bmp("mandelbrot.bmp",pos,h,w,0,9);
	std::cout << "[INFO]: Save IMG Mandelbro ="<<result<<"\n";
}

//END::mandelbro demo
/*********************************************************************************************************************************************************/




void getShortInformationGPU()
{
	int deviceCount=0;
	std::cout <<"\n";
	std::cout << "[INFO]: Information GPU"<<"\n";

	#ifdef  UseHIP
		hipGetDeviceCount(&deviceCount);
		if (deviceCount>0) {
			std::cout << "[INFO]: Number of available GPUs AMD: " << deviceCount << "\n";
			for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
				hipSetDevice(deviceId);
				std::cout << "[INFO]: GPU " << deviceId << " initialized and resources allocated." << "\n";
			}
		}
	#endif

	#ifdef UseCUDA
		cudaGetDeviceCount(&deviceCount);
		if (deviceCount>0) {
			std::cout << "[INFO]: Number of available GPUs NVIDIA: " << deviceCount << "\n";
			for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
				cudaSetDevice(deviceId);
				std::cout << "[INFO]: GPU " << deviceId << " initialized and resources allocated." << "\n";
			}
		}
	#endif
	std::cout <<"\n";
	if (deviceCount == 0) { std::cerr << "[INFO]: No GPUs found. Exiting." << "\n"; }
}


void getHipInformation()
{
  //BEGIN::INFO HIP AMD
    std::cout<<std::endl;
    int numDevices=0;
    HIP_CHECK(hipGetDeviceCount(&numDevices));
    std::cout<<"[INFO]: Get numDevice                = "<<numDevices<<"\n";
    int deviceID=0;
    HIP_CHECK(hipGetDevice(&deviceID));
    std::cout<<"[INFO]: Get deviceID activated       = "<<deviceID<<"\n";
    deviceID=0;
    hipSetDevice(deviceID);

    hipDeviceProp_t devProp;
    for (int i = 0; i < numDevices; i++)
    {
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipGetDeviceProperties(&devProp,i));
                std::cout<<"[INFO]:"<<std::endl;
                std::cout<<"[INFO]: DeviceID                     = "<<i<<std::endl;
                std::cout<<"[INFO]: Agent prop name              = "<< devProp.name<<std::endl;
                std::cout<<"[INFO]: System minor                 = "<< devProp.minor<<std::endl;
                std::cout<<"[INFO]: System major                 = "<< devProp.major<<std::endl;
                std::cout<<"[INFO]: Memory Clock Rate (KHz)      = "<< devProp.memoryClockRate<<std::endl;
                std::cout<<"[INFO]: Memory Bus Width (bits)      = "<< devProp.memoryBusWidth<<std::endl;
                std::cout<<"[INFO]: Peak Memory Bandwidth (GB/s) = "<< 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6<<std::endl;
                std::cout<<"[INFO]: max ThreadsPerBlock          = "<< devProp.maxThreadsPerBlock<<std::endl;
                std::cout<<"[INFO]: max ThreadsPerMultiProcessor = "<< devProp.maxThreadsPerMultiProcessor<<std::endl;
                std::cout<<"[INFO]: max ThreadsDim 3D            = "<< devProp.maxThreadsDim[0]<<" "<<devProp.maxThreadsDim[1]<<" "<<devProp.maxThreadsDim[2]<<std::endl;
                std::cout<<"[INFO]: max Grid Size 3D             = "<< devProp.maxGridSize[0]<<" "<<devProp.maxGridSize[1]<<" "<<devProp.maxGridSize[2]<<std::endl;
                std::cout<<"[INFO]: warpSize:                    = "<< devProp.warpSize << "\n";
                std::cout<<"[INFO]: regsPerBlock:                = "<< devProp.regsPerBlock << "\n";
                std::cout<<"[INFO]: concurrentKernels:           = "<< devProp.concurrentKernels << "\n";
                std::cout<<"[INFO]: total Global Mem             = "<< devProp.totalGlobalMem<<std::endl;
                std::cout<<"[INFO]: shared Mem Per Block         = "<< devProp.sharedMemPerBlock<<std::endl;
    }

    HIP_CHECK(hipSetDevice(0));
    std::cout<<std::endl;
    //END::INFO HIP AMD
}


void scanInformationSystem()
{
	int Value;
	std::cout <<"\n";
	std::cout << "[INFO]: Scan Information System..."<<"\n";
	Value=std::system("lscpu>InfoSystemCPU.txt");
	Value=std::system("lshw -C display>InfoSystemGPU.txt");
	std::cout <<"\n";
}

void getInformationCPU()
{
	std::cout <<"\n";
	std::cout << "[INFO]: Inormation CPU"<<"\n";
	readFileViewInformation("InfoSystemCPU.txt");
	std::cout <<"\n";
}

void getInformationGPU()
{
	std::cout <<"\n";
	std::cout << "[INFO]: Inormation GPU"<<"\n";
	readFileViewInformation("InfoSystemGPU.txt");
	std::cout <<"\n";
}

#ifdef USE_MPI
void getMpiInformation(int argc, char *argv[])
{
	//BEGIN::INFO MPI
	bool qFullInfoSystem=false;
    MPI_Init(NULL, NULL);
    int world_rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name,&name_len);

    if (world_rank == 0) { 
	  	std::cout <<"\n";
      	int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
      	std::cout << "[INFO]: Name worlds processor: "<<processor_name<<"\n";
      	std::cout << "[INFO]: Nb CPU available: "<<numCPU<< "\n";
      	std::cout <<"\n";
      	std::cout << "[INFO]: Scan..."<<"\n";
    }
    std::cout << "[INFO]: rank: "<<world_rank<<" out of "<<world_size<<"\n";
    MPI_Finalize();
	//END::INFO MPI
}
#endif


/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

void TestLevel001()
{
	const int matrixSize=2000;
    bool qView=true;
    qView=false;
	std::chrono::steady_clock::time_point t_begin,t_end;
	long int t_laps;
	
	Matrix MatA; 
    MatA = build_init_matrix(matrixSize,matrixSize); 
    //une matrice définie positive est une matrice positive inversible. XtMX>0.
	if (qView) { writeMatrix(MatA);	}
	
    std::cout << "\n";

    std::cout << "[INFO]: Method Serial"<< "\n";
	t_begin = std::chrono::steady_clock::now();
	Matrix MatU_Serial=getCholeskySerial(MatA);
	t_end = std::chrono::steady_clock::now();
	if (qView) { writeMatrix(MatU_Serial); }
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";
	
    /*
	t_begin = std::chrono::steady_clock::now();
	Matrix MatU_gpu1=getCholeskyGPUVers1(MatA);
	t_end = std::chrono::steady_clock::now();
	if (qView) { writeMatrix(MatU_gpu1); }
	checkSolution(MatU_Serial,MatU_gpu1);
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";
    */
	
    std::cout << "[INFO]: Method GPU HIP AMD"<< "\n";
	t_begin = std::chrono::steady_clock::now();
	Matrix MatU_gpu2=getCholeskyGPUVers2(MatA);
	t_end = std::chrono::steady_clock::now();
	if (qView) { writeMatrix(MatU_gpu2); }
	checkSolution(MatU_Serial,MatU_gpu2);
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";

    std::cout << "[INFO]: Method pthread"<< "\n";
    int NbThread=2;
    t_begin = std::chrono::steady_clock::now();
    Matrix MatU_pthreads=getCholesky_pthreads(MatA,NbThread);
    t_end = std::chrono::steady_clock::now();
    if (qView) { writeMatrix(MatU_pthreads); }
	checkSolution(MatU_Serial,MatU_pthreads);
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
	std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
    std::cout << "\n";
    std::cout << "\n";
    
    if (1==0) {
        std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
        Matrix MatU_gpu2t=matrix_tanspose(MatU_gpu2);
        Matrix MatT=matrix_product(MatU_gpu2t,MatU_gpu2);
        if (qView) { writeMatrix(MatT); }
        checkSolution(MatA,MatT);
        free(MatU_gpu2t.data);
        free(MatT.data);
    }
	
	free(MatU_Serial.data);
	//free(MatU_gpu1.data);
	free(MatU_gpu2.data);
	free(MatU_pthreads.data);
	free(MatA.data);
}

void TestLevel001beta()
{
	bool qView=true;
	//qView=false;
	const int matrixSize=64;
	Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize); 

	if (qView) { 
        std::cout<<"\n";
        std::cout << "Matrix A ===>\n\n";	writeMatrix2(MatA); std::cout << "\n";
    }  

	Matrix MatU=getCholeskyGPUVers2(MatA);
	
	if (qView) { 
        std::cout<<"\n";
        std::cout << "Matrix U ===>\n\n";	writeMatrix2(MatU); std::cout << "\n";
    }

	std::cout<<"[INFO]: CheckSolution"<<"\n";
    Matrix MatUt=matrix_transpose_GPU(MatU);
    Matrix MatT=matrix_product_GPU(MatUt,MatU);
    checkSolution_GPU(MatA,MatT);

	saveMatrixToImagePGM("MatU.pgm",MatU);
	saveMatrixToImageBMP("MatU.bmp",MatU);

	free(MatU.data);
	free(MatUt.data);
	free(MatA.data);
}

void TestLevel002()
{
	std::ofstream myfile;
	myfile.open ("Data.csv");
	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	//qCTRL=false;

	myfile <<"DimMatrix,ModeSerial,ModeGpuHipAMD,pthread"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 1; i <= 22; i++)
    {
		const int matrixSize=i*500;
		Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize); 
		myfile <<matrixSize<<",";


			std::cout << "[INFO]: Method Serial"<< "\n";
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_Serial=getCholeskySerial(MatA);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_Serial); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";

			myfile<<t_laps<<",";


			std::cout << "[INFO]: Method GPU HIP AMD"<< "\n";
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_gpu2=getCholeskyGPUVers2(MatA);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_gpu2); }
			if (qCTRL) { checkSolution(MatU_Serial,MatU_gpu2); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";

			myfile<<t_laps<<",";


			std::cout << "[INFO]: Method pthread"<< "\n";
			int NbThread=9;
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_pthreads=getCholesky_pthreads(MatA,NbThread);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_pthreads); }
			if (qCTRL) {checkSolution(MatU_Serial,MatU_pthreads); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";
		
			myfile<<t_laps;

			free(MatU_Serial.data);
			//free(MatU_gpu1.data);
			free(MatU_gpu2.data);
			free(MatU_pthreads.data);
			free(MatA.data);

			sleep(2);
		myfile<<"\n";
	}
	myfile.close();
}


void TestLevel003()
{
	std::ofstream myfile;
	myfile.open ("DataScaling.csv");

	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix,nbCPU,dt"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 0; i <= 5; i++)
    {
		int NbThread=pow(2,i);
		const int matrixSize=NbThread*100;

		Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize); 
		myfile <<matrixSize<<","<<NbThread<<",";

			std::cout << "[INFO]: Method pthread"<< "\n";
			
			t_begin = std::chrono::steady_clock::now();
			Matrix MatU_pthreads=getCholesky_pthreads(MatA,NbThread);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_pthreads); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";			
			myfile<<t_laps;
			free(MatU_pthreads.data);
			free(MatA.data);
			sleep(2);
		myfile<<"\n";
	}
	myfile.close();
}



void TestLevel004()
{
	std::ofstream myfile;
	myfile.open ("DataTpB.csv");
	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix";	for (int k = 3; k <= 9; k++) { myfile<<","<<pow(2,k); } myfile <<"\n";
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 1; i <= 22; i++)
    {
		const int matrixSize=i*500;
		Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize); 
		myfile <<matrixSize;
			std::cout << "[INFO]: Method GPU HIP AMD"<< "\n";

			for (int k = 3; k <= 9; k++)
   			{
				myfile<<",";
				int threads_per_block=pow(2,k);
				t_begin = std::chrono::steady_clock::now();
				Matrix MatU_gpu2=getCholeskyGPUVers3(MatA,threads_per_block);
				t_end = std::chrono::steady_clock::now();
				if (qView) { writeMatrix(MatU_gpu2); }
				t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
				std::cout << "[INFO]: i="<<i<<" threads_per_block "<<threads_per_block<<" Elapsed microseconds inside: "<<t_laps<< " us\n";
				std::cout << "\n";
				myfile<<t_laps;
				free(MatU_gpu2.data);
				
			}
		myfile<<"\n";
		free(MatA.data);
		sleep(1);
	}
	myfile.close();
}


void TestLevel005()
{
	long int t_laps;
    bool qView=true;
	const int r=2,c=r+4;
	Matrix MatA, MatR; 
	MatA = build_index_matrix(r,c);
	MatR = matrix_transpose_GPU(MatA);
	std::cout << "\n"; 
	std::cout << "Test Transpose GPU\n"; 
	if (qView) { std::cout << "Matrix A\n";	writeMatrix(MatA); std::cout << "\n"; }
	if (qView) { std::cout << "Matrix R\n"; writeMatrix(MatR); std::cout << "\n"; }
	free(MatA.data);
	free(MatR.data);
}

void TestLevel006()
{
	long int t_laps;
    bool qView=true;
	const int r=5,c=r;
	Matrix MatA, MatR; 
	MatA = build_index_matrix(r,c);
	MatR = matrix_lower_triangular_GPU(MatA);
	std::cout << "\n"; 
	std::cout << "Test Transpose GPU\n"; 
	if (qView) { std::cout << "Matrix A\n";	writeMatrix(MatA); std::cout << "\n"; }
	if (qView) { std::cout << "Matrix R\n"; writeMatrix(MatR); std::cout << "\n"; }
	free(MatA.data);
	free(MatR.data);
}


void TestLevel007()
{
	long int t_laps;
	std::chrono::steady_clock::time_point t_begin,t_end;
    bool qView=true;
	const int r=9,c=r;
	Matrix MatA,MatK, MatR_CPU, MatR_GPU; 
	MatA = build_index_matrix(r,c);
	

	Matrix MatKI= kernel_identity(3);
	if (qView) { std::cout << "Matrix KI\n";	writeMatrix(MatKI); std::cout << "\n"; }

	MatK = kernel_LaplacianOfGaussian(3,1.0);
	if (qView) { std::cout << "Matrix K\n";	writeMatrix(MatK); std::cout << "\n"; }


	t_begin = std::chrono::steady_clock::now();
	MatR_CPU = getConvolutionCPU(MatA,MatK);
	t_end = std::chrono::steady_clock::now();
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
				std::cout << "[INFO]: Elapsed microseconds inside CPU: "<<t_laps<< " us\n";
				std::cout << "\n";

	MatA = build_index_matrix(r,c);

	t_begin = std::chrono::steady_clock::now();
	MatR_GPU = getConvolutionGPUVers2(MatA,MatK);
	t_end = std::chrono::steady_clock::now();
	t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
				std::cout << "[INFO]: Elapsed microseconds inside GPU: "<<t_laps<< " us\n";
				std::cout << "\n";


	std::cout << "\n"; 
	std::cout << "Test Covolution\n"; 
	if (qView) { std::cout << "Matrix A\n";	writeMatrix(MatA); std::cout << "\n"; }
	if (qView) { std::cout << "Matrix R CPU\n"; writeMatrix(MatR_CPU); std::cout << "\n"; }
	if (qView) { std::cout << "Matrix R GPU\n"; writeMatrix(MatR_GPU); std::cout << "\n"; }

	free(MatA.data);
	free(MatK.data);
	free(MatKI.data);
	free(MatR_CPU.data);
	free(MatR_GPU.data);
}





#ifdef UseOpenMP
void TestOpenMP()
{
	std::ofstream myfile;
	myfile.open ("DataScalingOpeMP.csv");

	long int t_laps;
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix,nbCPU,dt"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

	for (int i = 0; i <= 7; i++)
    {
		int NbThread=pow(2,i);
		const int matrixSize=NbThread*10;

		Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize); 
		if (qView) { writeMatrix(MatA); }

		myfile <<matrixSize<<","<<NbThread<<",";
			std::cout << "[INFO]: Method OpenMP"<< "\n";
			std::cout << "[INFO]: NbThread :"<<NbThread<< "\n";
			t_begin = std::chrono::steady_clock::now();
			//Matrix MatU_OpenMP=getCholeskyOpenMPVers1(MatA,NbThread);
			Matrix MatU_OpenMP=getCholeskyOpenMPVers2(MatA,NbThread);
			t_end = std::chrono::steady_clock::now();
			if (qView) { writeMatrix(MatU_OpenMP); }
			t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
			std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
			std::cout << "\n";			
			myfile<<t_laps;
			
			if (1==1) {
				std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
				Matrix MatU_OpenMPt=matrix_tanspose(MatU_OpenMP);
				//Matrix MatT=matrix_product(MatU_OpenMPt,MatU_OpenMP);
				Matrix MatT=matrix_product_GPU(MatU_OpenMPt,MatU_OpenMP);
				//if (qView) { writeMatrix(MatT); }
				//checkSolution(MatA,MatT);
				checkSolution_GPU(MatA,MatT);
				std::cout << "\n";
				free(MatU_OpenMPt.data);
				free(MatT.data);
			}
	
			free(MatU_OpenMP.data);
			free(MatA.data);
			//sleep(2);
		myfile<<"\n";
	}
	myfile.close();
}
#endif



#ifdef USE_MPI
void TestMPI(int argc, char *argv[])
{
	getCholeskyMPIVers2(argc, argv);
}
#endif




int main(int argc, char *argv[])
 {
	//scanInformationSystem();``
	//getInformationCPU();
	//getInformationGPU();
    //getHipInformation();
	//getShortInformationGPU();
	//getMpiInformation(argc,argv);
	
	//TestMPI(argc,argv);
	//TestOpenMP();

	//TestLevel001();
	//TestLevel002();	
	//TestLevel003();
	//TestLevel004();
	//TestLevel005();
	//TestLevel006();
	//TestLevel007();
	//TestLevel001beta();
	//mandelbrot(512,512);
	
}
