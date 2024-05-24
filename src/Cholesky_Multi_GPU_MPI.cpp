
#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>


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


//Links Specx
#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"

#include "na.hpp"
#include "Tools.hpp"
//#include "Taskflow_HPC.hpp"

#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20

  
#include <cmath>

#include <mpi.h>


//Links HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
//#include <hip/hip_runtime.h>
//#define MAT_SIZE (16*1024)
//#define MAT_SIZE (16*512)
//#define MAT_SIZE (16*8)
#define MAT_SIZE (4*2)

#define BLOCK_SIZE 4
#define MAT_BLOCKS (MAT_SIZE/BLOCK_SIZE)
#define NUM_GPUS 2
#define NUM_GPUS_PER_NODE 3


#define COMPARE
#define NOCPUCMP

//#define PRINTMATRIX
//#define FULLINIT
//#define PRINTINPUTMATRIX
//#define MATDISP
//#define CPU



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




 typedef struct {
 	  unsigned int num_columns;
   	  unsigned int num_rows;
 	  unsigned int pitch; 
 	  double* elements;
  } Matrix;



/*********************************************************************************************************************************************************/
// BEGIN::INTRODUCTION
int check_if_symmetric                 (const Matrix M); 
int check_if_diagonal_dominant         (const Matrix M);
Matrix create_positive_definite_matrix (unsigned int, unsigned int);
Matrix allocate_matrix                 (int num_rows, int num_columns, int init);

void writeMatrix                       (const Matrix);
void copy_matrix_to_device             (Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device           (Matrix Mhost,   const Matrix Mdevice);
Matrix allocate_matrix_on_gpu          (const Matrix M);
// END::INTRODUCTION
/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
//BEGIN::TOOLS MEMORY TRANSFER HIP AMD GPU

Matrix allocate_matrix(int num_rows, int num_columns, int init) {
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (double *) malloc(size * sizeof (double));
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.elements[i] = 0;
        else
            M.elements[i] = (double) rand() / (double) RAND_MAX;
    }
    return M;
}


Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(double);
    hipMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}


void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(double);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    hipMemcpy(Mdevice.elements, Mhost.elements, size, hipMemcpyHostToDevice);
}

void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(double);
    hipMemcpy(Mhost.elements, Mdevice.elements, size, hipMemcpyDeviceToHost);
}
//END::TOOLS MEMORY TRANSFER HIP AMD GPU
/*********************************************************************************************************************************************************/

/*********************************************************************************************************************************************************/
// BEGIN::HIP AMD GPU

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

	hipLaunchKernelGGL(matrix_mult,grid, thread_block,0,0,gpu_C.elements,gpu_A.elements,gpu_B.elements,matrixSize,matrixSize); 

	copy_matrix_from_device(C,gpu_C);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.elements);
	hipFree(gpu_B.elements);
	hipFree(gpu_C.elements);

	return C;
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
	hipLaunchKernelGGL(matrix_equal,grid, thread_block,0,0,d_Q,gpu_A.elements,gpu_B.elements,matrixSize*matrixSize,deltaError); 
	hipMemcpy(h_Q,d_Q,sizeof(bool), hipMemcpyDeviceToHost);
	hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
	hipFree(gpu_A.elements);
	hipFree(gpu_B.elements);
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



/*********************************************************************************************************************************************************/
//BEGIN:: BUILD INIT MATRIX

Matrix create_positive_definite_matrix(unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (double *)malloc(size * sizeof(double));

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
	printf("Creating a %d x %d matrix with random numbers between [-.5, .5]...", num_rows, num_columns);
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.elements[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
       	printf("done. \n");
	// writeMatrix(M);
	// getchar();

	// Step 2: Make the matrix symmetric by adding its transpose to itself
	printf("Generating the symmetric matrix...");
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_columns;
	transpose.num_rows = num_rows; 
	size = transpose.num_rows * transpose.num_columns;
	transpose.elements = (double *)malloc(size * sizeof(double));

	for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			transpose.elements[i * M.num_rows + j] = M.elements[j * M.num_columns + i];
	// writeMatrix(transpose);

	for(i = 0; i < size; i++)
		M.elements[i] += transpose.elements[i];
	if (check_if_symmetric(M))
		printf("done. \n");
	else{ 
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	// Step 3: Make the diagonal entries large with respect to the row and column entries
	printf("Generating the positive definite matrix...");
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++){
			if(i == j) 
				M.elements[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if(check_if_diagonal_dominant(M))
		printf("done. \n");
	else{
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	free(transpose.elements);
	return M;
}


void writeMatrix(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
		{
			printf("%f ", M.elements[i*M.num_columns + j]);
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
            fprintf(FICH,"%f ", M.elements[i*M.num_columns + j]);
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
			myfile<<M.elements[i*M.num_columns + j];
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
			myfile>>M.elements[i*M.num_columns + j];
    }
    myfile.close();
	return M;
}

int check_if_symmetric(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
		for(unsigned int j = 0; j < M.num_columns; j++)
			if(M.elements[i * M.num_rows + j] != M.elements[j * M.num_columns + i]) return 0;
	return 1;
}

int check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j) sum += abs(M.elements[i * M.num_rows + j]);
		}
		if(diag_element <= sum) return 0;
	}
	return 1;
}

Matrix matrix_multiply(const Matrix A, const Matrix B) 
{
    Matrix C;
    C.num_columns = C.pitch = A.num_columns;
    C.num_rows = A.num_rows;
    unsigned int size = C.num_rows * C.num_columns;
    C.elements = (double *) malloc(size * sizeof (double));

    for (unsigned int i = 0; i < A.num_columns; i++)
        for (unsigned int j = 0; j < B.num_rows; j++) {
            double sum = 0.0f;
            for (unsigned int k = 0; k < A.num_columns; k++) {
                double a = A.elements[i * A.num_columns + k];
                double b = B.elements[k * B.num_rows + j];
                sum += a * b;
            }
            C.elements[i * B.num_rows + j] = (double) sum;
        }
    return C;
}

Matrix matrix_tanspose(const Matrix M) 
{
  Matrix R= allocate_matrix(M.num_columns,M.num_rows,0);
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.elements[i * M.num_rows + j] = M.elements[j * M.num_columns + i];
  return R;
}


void matrix_copy_elements(Matrix R,const Matrix M) 
{
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.elements[i * M.num_rows + j] = M.elements[i * M.num_rows + j];
}


Matrix matrix_product(const Matrix A, const Matrix B) 
{
    Matrix C;
    C.num_columns = C.pitch = A.num_columns;
    C.num_rows = A.num_rows;
    unsigned int size = C.num_rows * C.num_columns;
    C.elements = (double *) malloc(size * sizeof (double));

    for (unsigned int i = 0; i < A.num_columns; i++)
        for (unsigned int j = 0; j < B.num_rows; j++) {
            double sum = 0.0f;
            for (unsigned int k = 0; k < A.num_columns; k++) {
                double a = A.elements[i * A.num_columns + k];
                double b = B.elements[k * B.num_rows + j];
                sum += a * b;
            }
            C.elements[i * B.num_rows + j] = (double) sum;
        }
    return C;
}


void matrix_lower_triangular(Matrix M) 
{
    int i, j;
    for (i = 0; i < M.num_rows; i++)
        for (j = 0; j < i; j++)
            M.elements[i * M.num_rows + j] = 0.0;
}



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
    unsigned res = compareArrays(MatRef.elements, MatRes.elements,MatRef.num_rows);
    printf("[INFO]:	%s\n", (1 == res) ? "WELL DONE PASSED :-)" : "FAILED");
}


//END:: BUILD INIT MATRIX
/*********************************************************************************************************************************************************/






/*********************************************************************************************************************************************************/




/*********************************************************************************************************************************************************/

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


/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/
/*********************************************************************************************************************************************************/

bool isFileExist(std::string ch)
{
    std::ifstream myfile;
    myfile.open(ch); bool qOK=false;
    if (myfile) { qOK=true; }
    myfile.close();
    return (qOK);
}




int nprocs,myrank;


__global__ void d_choldc_topleft(double (*m)[MAT_SIZE], int boffset)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double topleft[BLOCK_SIZE][BLOCK_SIZE+1];
    topleft[ty][tx]=m[ty+(boffset/NUM_GPUS)*BLOCK_SIZE][tx+BLOCK_SIZE*boffset];
    __syncthreads();
    double diagelem,fac;

    for (int k=0;k<BLOCK_SIZE;k++)
      {
        __syncthreads();
        fac=1./sqrt(topleft[k][k]);
        __syncthreads();
        if ((ty==k)&&(tx>=k)) { topleft[ty][tx]=(topleft[ty][tx])*fac; }
        __syncthreads();
        if ((tx>=ty)&&(ty>k)) { topleft[ty][tx]=topleft[ty][tx]-topleft[k][ty]*topleft[k][tx]; }
      }
    __syncthreads();
    if (tx>=ty) { m[ty+(boffset/NUM_GPUS)*BLOCK_SIZE][tx+BLOCK_SIZE*boffset]=topleft[ty][tx]; }

}


__global__ void d_choldc_strip(double (*m)[MAT_SIZE],int blockoffset)
{
  int boffx = blockIdx.x+blockoffset+1; 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ double topleftt[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double workingmat[BLOCK_SIZE][BLOCK_SIZE+1];
  topleftt[tx][ty]   = m[ty+(blockoffset/NUM_GPUS)*BLOCK_SIZE][tx+blockoffset*BLOCK_SIZE];
  workingmat[ty][tx] = m[ty+(blockoffset/NUM_GPUS)*BLOCK_SIZE][tx+boffx*BLOCK_SIZE];
  __syncthreads();

  if (ty==0)
    {
      for (int k=0;k<BLOCK_SIZE;k++)
      {
        double dotprod=0.0;
        for (int m=0;m<k;m++) { dotprod+=topleftt[k][m]*workingmat[m][tx];}
        workingmat[k][tx]=(workingmat[k][tx]-dotprod)/topleftt[k][k];
      }
    }
  __syncthreads();
  m[ty+(blockoffset/NUM_GPUS)*BLOCK_SIZE][tx+boffx*BLOCK_SIZE]=workingmat[ty][tx];
}




__global__ void d_choldc_diagupdate(double (*m)[MAT_SIZE],double (*wsm)[MAT_SIZE],int startingblock)  
{
  int boffx = NUM_GPUS*blockIdx.x+startingblock; 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ double topt[BLOCK_SIZE][BLOCK_SIZE+1];
  topt[tx][ty]=wsm[ty][tx+boffx*BLOCK_SIZE];
  __syncthreads();
  double matrixprod=0.0;
  if (tx>=ty)  
  { 
    for (int kk=0;kk<BLOCK_SIZE;kk++) { matrixprod+=topt[ty][kk]*topt[tx][kk]; }
    __syncthreads();
    m[ty+(boffx/NUM_GPUS)*BLOCK_SIZE][tx+boffx*BLOCK_SIZE]-=matrixprod; 
  }
}



// this kernel takes the results of the above ones and applies them to the 
//rest of the matrix...
__global__ void d_choldc_hiupdate(double (*m)[MAT_SIZE], double (*wsm)[MAT_SIZE], int startingblock)  
{

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int boffy=NUM_GPUS*blockIdx.x+startingblock;
  int boffx=boffy+1;

  // the +1's stop shared memory bank conflicts when accessing down columns
  // There are already no shared bank conflicts when accessing by row

  __shared__ double leftt[BLOCK_SIZE][BLOCK_SIZE+1];
  __shared__ double rightt[BLOCK_SIZE][BLOCK_SIZE+1];

  // now read in the data, always from top right

  int tmpx,tmpy;
  tmpy=__mul24(boffy,BLOCK_SIZE);

  // note the tmpy in the latter term to ensure we get the
  // correct common matrix for the row
  leftt[tx][ty]=wsm[ty][tx+tmpy];
  for (;boffx<MAT_BLOCKS;boffx++) 
  {
    tmpx=__mul24(boffx,BLOCK_SIZE);
    rightt[tx][ty]=wsm[ty][tx+tmpx];
    __syncthreads();
    double matrixprod=0.0;
    // ty,tx'th thread works out (ty,tx) cmpt of the product...
    for (int kk=0;kk<BLOCK_SIZE;kk++) { matrixprod+=leftt[ty][kk]*rightt[tx][kk];}
    __syncthreads();
    m[ty+(boffy/NUM_GPUS)*BLOCK_SIZE][tx+tmpx]-=matrixprod;
  }

}



void cpumatdisp(double (*mat)[MAT_SIZE],double (*diag)[MAT_SIZE])
{
  int i,j;
  printf("CPU output: \n");
  for (j=0;j<MAT_SIZE;j++)
  {
      for (i=0;i<MAT_SIZE;i++) { printf("%7.4f ",mat[j][i]); }
      printf("\n");
    }
  printf("\n");
  for (i=0;i<MAT_SIZE;i++) { printf("%7.4f ",diag[0][i]); }
  printf("\n");
}


void matdisp(double (*matptr)[MAT_SIZE])
{
  double mat[MAT_SIZE][MAT_SIZE];

  unsigned int mat_size=MAT_SIZE*MAT_SIZE*sizeof(double);

  int i,j;
  hipError_t error;
  //hipThreadSynchronize();
  hipDeviceSynchronize();

  //    printf("In matdisp, matptr=%p.\n\n",matptr);

  hipMemcpy(mat,matptr, mat_size,hipMemcpyDeviceToHost);
  //	error=hipGetLastError();
  //	printf("In mat disp, Error code %d: %s.\n",error,hipGetErrorString(error));

  //hipThreadSynchronize();
  hipDeviceSynchronize();

  printf("\n");

    for (j=0;j<MAT_SIZE;j++)
    {
      for (i=0;i<MAT_SIZE;i++) { printf("%7.4f ",mat[j][i]); }
      printf("\n");
    }

  printf("\n");


  //hipThreadSynchronize();
  hipDeviceSynchronize();
}





void choldc(double (*mat)[MAT_SIZE])
{
      volatile clock_t gputime;
      double workingblockrow[BLOCK_SIZE][MAT_SIZE];
      double (*d_mat)[MAT_SIZE];
      unsigned int mat_size=(MAT_SIZE/NUM_GPUS)*MAT_SIZE*sizeof(double);
      double (*d_workingblockrow)[MAT_SIZE];
      unsigned int workingblockrow_size=BLOCK_SIZE*MAT_SIZE*sizeof(double);
      hipError_t error; 
      dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
      dim3 stripgrid;
      dim3 diaggrid;
      dim3 higrid;
      int worker,workstartrow;
      int j=MAT_BLOCKS; // doesn't change
      int i=MAT_BLOCKS; // counts down; the no. of block cols still to do 
      int tmp1,tmp2,tmp3,tmp4; // for indexing calculations
      int pushforward;
      int firstdiagblock; 

      hipMalloc((void**) &d_mat,mat_size);
      hipMemcpy(d_mat, mat, mat_size,hipMemcpyHostToDevice);
      hipMalloc((void**) &d_workingblockrow,workingblockrow_size);

      error=hipGetLastError();
      printf("     Error code from process %i is %d: %s.\n",myrank,error,hipGetErrorString(error)); 
      gputime=clock();

    #ifdef MATDISP
      matdisp(d_mat);
    #endif

      while(i>2)
        {
          worker=(j-i)%nprocs;
          workstartrow=((j-i)/nprocs)*BLOCK_SIZE;
          //      printf("worker for i=%i is %i.\n",i,worker);
          stripgrid.x=i-1; stripgrid.y=1; stripgrid.z=1;
          // tmp3 shall contain the total no of blockrows a process will handle 
          tmp1=(MAT_BLOCKS)/nprocs; tmp2=(MAT_BLOCKS)%nprocs; 
          if (myrank<tmp2) { tmp3=tmp1+1; } else { tmp3=tmp1; }
          // tmp4 shall contain the number of irrelevant blockrows
          // for a process...
          tmp1=(j-i+1)/nprocs;  tmp2=(j-i+1)%nprocs; 
          if (myrank<tmp2){ tmp4=tmp1+1; } else { tmp4=tmp1; }
          diaggrid.x=tmp3-tmp4; diaggrid.y=1; diaggrid.z=1;
          // now subtracting 1 off the total because hiupdate
          // doesn't operate on the last blockrow 
          tmp1=(MAT_BLOCKS-1)/nprocs;  tmp2=(MAT_BLOCKS-1)%nprocs; 
          if (myrank<tmp2){ tmp3=tmp1+1; } else { tmp3=tmp1; }
          higrid.x=tmp3-tmp4; higrid.y=1; higrid.z=1;

          if (myrank==worker)
          {
              //	printf("rank %i about to work...\n",myrank);
              //d_choldc_topleft<<<1,threads>>>(d_mat,j-i);
              hipLaunchKernelGGL(d_choldc_topleft,1,threads,0,0,d_mat,j-i);
              //d_choldc_strip<<<stripgrid,threads>>>(d_mat,j-i);
              hipLaunchKernelGGL(d_choldc_strip,stripgrid,threads,0,0,d_mat,j-i);
              // the following could possibly be replaced by an appropriate
              // call to hipMemcpy2D, saving communication...
              hipMemcpy(workingblockrow,d_mat[workstartrow], workingblockrow_size, hipMemcpyDeviceToHost);
          }

          MPI_Bcast(workingblockrow,BLOCK_SIZE*MAT_SIZE,MPI_DOUBLE,worker,MPI_COMM_WORLD);
          hipMemcpy(d_workingblockrow,workingblockrow,workingblockrow_size,hipMemcpyHostToDevice);

          if (myrank>worker)  firstdiagblock=j-i+myrank-worker;
          if (myrank==worker) firstdiagblock=j-i+nprocs;
          if (myrank<worker)  firstdiagblock=j-i+nprocs+myrank-worker;

          if (diaggrid.x>0){
              //d_choldc_diagupdate<<<diaggrid,threads>>>(d_mat,d_workingblockrow,firstdiagblock);
              hipLaunchKernelGGL(d_choldc_diagupdate,diaggrid,threads,0,0,d_mat,d_workingblockrow,firstdiagblock);

              
          }
          if (higrid.x>0){
              //d_choldc_hiupdate<<<higrid,threads>>>(d_mat,d_workingblockrow,firstdiagblock);
              hipLaunchKernelGGL(d_choldc_hiupdate,higrid,threads,0,0,d_mat,d_workingblockrow,firstdiagblock);
          }
          i--;
        }

      if (j>1)
      {
          worker=(j-2)%nprocs;
          workstartrow=((j-2)/nprocs)*BLOCK_SIZE;

          if (myrank==worker){
              //	printf("rank %i about to work in part %i...\n",myrank,i);
              
              //d_choldc_topleft<<<1,threads>>>(d_mat,j-2);
              hipLaunchKernelGGL(d_choldc_topleft,1,threads,0,0,d_mat,j-2);
              //d_choldc_strip<<<1,threads>>>(d_mat,j-2);
              hipLaunchKernelGGL(d_choldc_strip,1,threads,0,0,d_mat,j-2);

              hipMemcpy(workingblockrow,d_mat[workstartrow],workingblockrow_size,hipMemcpyDeviceToHost); 
          }

          MPI_Bcast(workingblockrow,BLOCK_SIZE*MAT_SIZE,MPI_DOUBLE,worker,MPI_COMM_WORLD);
          hipMemcpy(d_workingblockrow,workingblockrow,workingblockrow_size,hipMemcpyHostToDevice);

              if (myrank>worker)  firstdiagblock=j-i+myrank-worker;
              if (myrank==worker) firstdiagblock=j-i+nprocs;
              if (myrank<worker)  firstdiagblock=j-i+nprocs+myrank-worker;

              // tmp3 shall contain the total no of blockrows a process will handle 
              tmp1=(MAT_BLOCKS)/nprocs; 
              tmp2=(MAT_BLOCKS)%nprocs; 
              if (myrank<tmp2) { tmp3=tmp1+1; } else { tmp3=tmp1; }

              // tmp4 shall contain the number of irrelevant blockrows
              // for a process...
              tmp1=(j-i+1)/nprocs; 
              tmp2=(j-i+1)%nprocs; 
              if (myrank<tmp2) { tmp4=tmp1+1; } else { tmp4=tmp1; }

              diaggrid.x=tmp3-tmp4; diaggrid.y=1; diaggrid.z=1;
              if (diaggrid.x>0){
                  //d_choldc_diagupdate<<<diaggrid,threads>>>(d_mat,d_workingblockrow,firstdiagblock);
                  hipLaunchKernelGGL(d_choldc_diagupdate,diaggrid,threads,0,0,d_mat,d_workingblockrow,firstdiagblock);
              }
              i--;
        }

      //  printf("rank %i out of the if...\n",myrank);
      worker=(j-1)%nprocs;
      if (myrank==worker){
        // printf("rank %i about to work in the last topleft\n",myrank);
        //d_choldc_topleft<<<1,threads>>>(d_mat,j-1);
        hipLaunchKernelGGL(d_choldc_topleft,1,threads,0,0,d_mat,j-1);
      }
      //hipThreadSynchronize();
	  hipDeviceSynchronize();
      gputime=clock()-gputime;
      printf("kernel time from process %i=%f s.\n",myrank,gputime/1.e6f);
      hipMemcpy(mat,d_mat, mat_size,hipMemcpyDeviceToHost);
      hipFree(d_mat);
      hipFree(d_workingblockrow);
      //hipThreadSynchronize();
	  hipDeviceSynchronize();
      error=hipGetLastError();
      printf("     Error code from process %i is %d: %s.\n",myrank,error,hipGetErrorString(error)); 

}




int main(int argc, char** argv)
{

    //mpirun -np 2 ./main
    //printf("NUM_GPUS=%i.\n",NUM_GPUS);
    //printf("NUM_GPUS_PER_NODE=%i.\n",NUM_GPUS_PER_NODE);

    double x [(MAT_SIZE/NUM_GPUS)][MAT_SIZE];
	  double diagonal[1][MAT_SIZE];
    double *bigxptr=NULL;
    double *bigx=NULL;
    double *xptr=NULL;
    int i,j,devcnt,mydev;
    int tmp;
    clock_t maincputime,maingputime;
    hipError_t error;

    Matrix MatA=allocate_matrix(MAT_SIZE,MAT_SIZE,0);

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    std::cout<<"[INFO]: rank "<<myrank<<" nprocs "<<nprocs<<"\n";
    if (nprocs!=NUM_GPUS) {
      std::cout<<"[INFO]: disagreement between NUM_GPUS and nprocs, from rank "<<myrank<<"\n";
      MPI_Abort(MPI_COMM_WORLD,-1);
      MPI_Finalize();
      return -1;
    }

    hipGetDeviceCount(&devcnt);
    std::cout<<"[INFO]: hip device count= "<<devcnt<<"\n";
    if (devcnt!=NUM_GPUS_PER_NODE) {
      std::cout<<"[INFO]: disagreement between NUM_GPUS_PER_NODE and devcnt on rank "<<myrank<<"\n";
      MPI_Abort(MPI_COMM_WORLD,-1);
      MPI_Finalize();
      return -1;
    }



    mydev=myrank%devcnt;
    hipSetDevice(mydev);
    error=hipGetLastError();
    std::cout<<"[INFO]: Error code from process "<<myrank<<" is "<<error<<": "<<hipGetErrorString(error)<<"\n";
    hipGetDevice(&tmp);
    std::cout<<"[INFO]: hip device for process "<<myrank<<" is "<<tmp<<"\n";
    std::cout<<"\n";

    if (myrank==0) {

        MatA = create_positive_definite_matrix(MAT_SIZE,MAT_SIZE); 
        std::cout<<"\n";

        std::cout<<"[INFO]: In from proc 0\n\n";
        bigx=(double*)calloc(sizeof(double),MAT_SIZE*MAT_SIZE);
        if (bigx==NULL) printf("Problem with allocation.\n");
        memset(bigx,0,sizeof(double)*MAT_SIZE*MAT_SIZE);
        std::cout<<"[INFO]: Initializing test matrix...\n";     
        std::cout<<"[INFO]: MAT_SIZE="<<MAT_SIZE<<"\n";

        //for (i=0;i<MAT_SIZE; i++) { bigx[i*MAT_SIZE+i]=i+1.f; }
        for (int i=0;i<MAT_SIZE;i++)
        {
          for (int j=0;j<MAT_SIZE;j++) { bigx[i*MAT_SIZE+j]=MatA.elements[i * MAT_SIZE + j]; }
        }
        std::cout<<"[INFO]: View matrix A\n";
        writeMatrix(MatA); 
        std::cout << "\n";

        //for (i=0;i<MAT_SIZE;i+=100) { printf("m[%i][%i]=%f\n",i,i,bigx[i*MAT_SIZE+i]); }
        std::cout << "=====================================================================\n";		
        printf("MAT_SIZE=%i.\n",MAT_SIZE);
        printf("MAT_SIZE/(NUM_GPUS*BLOCK_SIZE)=%i.\n",MAT_SIZE/(NUM_GPUS*BLOCK_SIZE));
        std::cout << "=====================================================================\n";		
        std::cout << "\n";		
    }

    MPI_Barrier(MPI_COMM_WORLD);
    

    for (i=0; i<MAT_SIZE/(NUM_GPUS*BLOCK_SIZE); i++)
    {
        if (myrank==0){
          bigxptr=&(bigx[NUM_GPUS*BLOCK_SIZE*i*MAT_SIZE]);
          std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";		
          printf("%p, %f, %f.\n",bigxptr,*bigxptr,bigxptr[3*MAT_SIZE+3]);
          std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";		
          std::cout << "\n";
        }
        else {
          bigxptr=NULL;
        }   
        xptr=&(x[BLOCK_SIZE*i][0]);
        printf("From rank %d, %p, %f, %f.\n",myrank,xptr,*xptr,xptr[3]);
        std::cout << "\n";
        MPI_Scatter((void*)bigxptr,BLOCK_SIZE*MAT_SIZE,MPI_DOUBLE,(void*)xptr,BLOCK_SIZE*MAT_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
  
    std::cout << "\n";
    std::cout << "[INFO]: Cholesky factorizing...\n";

    if (myrank==0){
      /*
      std::cout<<"[INFO]: CTRL Matrix\n";
      for (int i=0;i<MAT_SIZE;i++)
        {
          for (int j=0;j<MAT_SIZE;j++) { printf("%.2f ",bigx[i*MAT_SIZE+j]); }
          std::cout << "\n";
        }
        std::cout << "\n";
      */
    }

    if (1==1) {
              maincputime=clock();
              maincputime=clock()-maincputime;
              std::cout << "[INFO]: GPU proper..."<< "\n";
              std::cout << "\n";
              maingputime=clock();
              choldc(x);
              maingputime=clock()-maingputime;
              std::cout<<"[INFO]: maingputime="<<maingputime<<" maincputime="<<maincputime<<"\n";

      //  printf("x[%d][%d]=%f.\n",MAT_SIZE-1,MAT_SIZE-1,x[MAT_SIZE-1][MAT_SIZE-1]);

      for (i=0;i<MAT_SIZE/(NUM_GPUS*BLOCK_SIZE);i++){
          if (myrank==0){
            bigxptr=&(bigx[NUM_GPUS*BLOCK_SIZE*i*MAT_SIZE]);
            //    printf("%p, %f, %f.\n",bigxptr,*bigxptr,bigxptr[3*MAT_SIZE+3]);
          }
          else { bigxptr=NULL; }
          xptr=&(x[BLOCK_SIZE*i][0]);
          //    printf("From rank %d, %p, %f, %f.\n",myrank,xptr,*xptr,xptr[3]);
          MPI_Gather((void*)xptr,BLOCK_SIZE*MAT_SIZE,MPI_DOUBLE,(void*)bigxptr,BLOCK_SIZE*MAT_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);
      }

    }
 



    if (myrank==0) {
      std::cout<<"[INFO]: ============================================================="<< "\n";
      std::cout<<"[INFO]: View matrix U results\n";

      Matrix MatL = allocate_matrix(MAT_SIZE,MAT_SIZE,0);

      //for (i=0;i<MAT_SIZE;i+=100) { printf("d[%d]=%f.\n",i,bigx[i*MAT_SIZE+i]); }
       printf("\n");
        for (i=0;i<MAT_SIZE;i++)
        {
          for (j=0;j<MAT_SIZE;j++) { 
            printf("%.2f ",bigxptr[i*MAT_SIZE+j]); 
            MatL.elements[i * MAT_SIZE + j]=bigxptr[i*MAT_SIZE + j]; 
            //MatL.elements[i * MAT_SIZE + j]=bigx[i*MAT_SIZE + j]; 
          }
          printf("\n");
        }
        printf("\n");

      matrix_lower_triangular(MatL);
      //writeMatrix(MatL); 

      			if (1==1) {
              std::cout << "[INFO]: Controle matrix product if A=tU*U \n";
              Matrix MatLt=matrix_tanspose(MatL);
              //Matrix MatT=matrix_product(MatU_OpenMPt,MatU_OpenMP);
              //Matrix MatT=matrix_product_GPU(MatL,MatLt);
              Matrix MatT=matrix_product_GPU(MatLt,MatL);

               std::cout<<"Matrix A\n";
               writeMatrix(MatA); 
               std::cout << "\n";

              std::cout << "Matrix R=tL.L\n";
              writeMatrix(MatT); 
              std::cout << "\n";
              //checkSolution(MatA,MatT);
              checkSolution_GPU(MatA,MatT);
              std::cout << "\n";
              free(MatLt.elements);
              free(MatT.elements);
			    }

      free(MatL.elements);
      std::cout << "[INFO]: FINISHED"<< "\n";
      free(bigx);
    }

    MPI_Finalize();

    return 0;
}

