
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
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}

void saveMatrix(const Matrix M, char *filename) 
{
    FILE* FICH = fopen(filename,"w");
    for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++)
            fprintf(FICH,"%f ", M.elements[i * M.num_rows + j]);
        fprintf(FICH,"\n");
    }
    fprintf(FICH,"\n");
    fclose(FICH);
}

void readMatrix(const Matrix M, char *filename) 
{

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



void getCholeskyMPIVers2(int argc, char *argv[])
{
    int n=atoi(argv[1]);
    bool qView=atoi(argv[2]);
    Matrix MatA=allocate_matrix(n,n,0);
    Matrix MatM=allocate_matrix(n,n,0);
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
      MatA = create_positive_definite_matrix(n,n);
      matrix_copy_elements(MatM,MatA);
      if (qView) { writeMatrix(MatA);    }
      t_begin = std::chrono::steady_clock::now();
    }

    MPI_Barrier(MPI_COMM_WORLD);


    for (int k=0; k<n; k++)
    {
      if (world_rank_mpi == 0){
        for (int j=0; j<k; j++)
        {
          MatM.elements[k * n + k]-=(MatM.elements[k * n + j] * MatM.elements[k * n + j]);
        }
        MatM.elements[k * n + k]=sqrt(MatM.elements[k * n + k]);
        
        for (int p=1; p<n-k; p++)
        {
          for (int c=0; c<=k; c++)
          {
            for (int r=c; r<n; r++)
            {
              MPI_Send(&MatM.elements[r * n + c], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
          }
          MPI_Recv(&MatM.elements[( p + k )* n + k], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (k==n-1)
        {
          for(int i=0; i<n-1; i++)
          {
            for(int j=i+1; j<n; j++)
            {
              MatM.elements[i * n + j]=0;
            }
          }
        }
      }

      else if (world_rank_mpi<n-k){
        for (int c=0; c<=k; c++)
        {
          for (int r=c; r<n; r++)
          {
            MPI_Recv(&MatM.elements[r * n + c], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
        
        for (int g=0; g<k; g++) //calculating non-diagonal elements concurrently
        {
          MatM.elements[(k+world_rank_mpi) * n + k]-=MatM.elements[k * n + g] * MatM.elements[(k+world_rank_mpi) * n + g];
        }
        MatM.elements[(k+world_rank_mpi) * n + k]/=MatM.elements[k * n + k];
        MPI_Send(&MatM.elements[(k+world_rank_mpi) * n + k], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
                  //Matrix MatT=matrix_product(MatM,MatMt);
                  Matrix MatT=matrix_product_GPU(MatM,MatMt);
                  //if (qView) { writeMatrix(MatT); }
                  //checkSolution(MatA,MatT);
                  checkSolution_GPU(MatA,MatT);
                  free(MatMt.elements);
                  free(MatT.elements);
            }
            
          }
          
        free(MatM.elements);
    }
    MPI_Finalize();
}






int main(int argc, char** argv) {
    //mpirun -np 32 ./main 10 1
    //mpirun -np 32 ./main dimMatrix qView
   getCholeskyMPIVers2(argc, argv);
}  


