#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <thread>
#include <vector>
#include <array>
#include <typeinfo>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <pthread.h>

#include <algorithm> 
#include <string>
#include <utility>
#include <functional>
#include <future>
#include <cassert>
#include <chrono>
#include <type_traits>
#include <list>
#include <ranges>


//#include <execution> //C++20
#include "SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpConsumerThread.hpp"
#include "SpComputeEngine.hpp"
#include "Speculation/SpSpeculativeModel.hpp"


//Links Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>




//Links
#include "na.hpp"

/*
//Under Feelpp

#include <specx/Data/SpDataAccessMode.hpp>
#include <specx/Legacy/SpRuntime.hpp>
#include <specx/Task/SpPriority.hpp>
#include <specx/Task/SpProbability.hpp>
#include <specx/Utils/SpArrayView.hpp>
#include <specx/Utils/SpTimer.hpp>
#include <specx/Utils/small_vector.hpp>
#include <specx/Utils/SpBufferDataView.hpp>
#include <specx/Utils/SpHeapBuffer.hpp>
#include <specx/Utils/SpUtils.hpp>
#include <specx/Utils/SpConsumerThread.hpp>
#include <specx/Legacy/SpRuntime.hpp>

#include <napp/na.hpp>
*/


//Links mpi
//#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

//Links omp
#define USE_OpenMP
#ifdef USE_OpenMP
	#include <omp.h>
#endif



//#define COMPILE_WITH_CUDA
#define COMPILE_WITH_HIP

#define USE_GPU_HIP  // <**** temporary

//#define UseCUDA
#define UseHIP


//Links HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"


//Links CUDA
//#include "cuda_runtime.h"
//#include "cuda.h"


//#define COMPILE_WITH_CXX_20

#define MODE_NO_THREAD 0
#define MODE_THREAD 1
#define MODE_ASYNC 2
#define MODE_SPECX 3

#define MODE_HIP  30
#define MODE_CUDA 31


// Some macro functions for the AMD HIP GPU
#define HIP_KERNEL_NAME(...) __VA_ARGS__

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


#define CUDA_CHECK(command) {               \
  cudaError_t status = command;             \
  if (status!=hipSuccess) {                \
    std::cerr <<"Error: CUDA reports "<< cudaGetErrorString(status)<< std::endl; \
    std::abort(); } }


#ifdef NDEBUG
    #define CUDA_ASSERT(x) x
#else
    #define CUDA_ASSERT(x) (assert((x)==cudaSuccess))
#endif



 #define _param(...) _parameters=Frontend::parameters(__VA_ARGS__)

 

//================================================================================================================================
// Get Information System CPU and GPU
//================================================================================================================================

namespace LEM {

void readFileViewInformation(char *filename) 
{
	FILE* FICH = NULL;
    int c = 0;
	FICH = fopen(filename, "r");
    if (FICH != NULL) { do { c = fgetc(FICH); printf("%c",c); } while (c != EOF); fclose(FICH); }
}


void scanInformationSystem()
{
	int Value;
	std::cout <<"\n";
	std::cout << "[INFO]: Scan Information System..."<<"\n";
	Value=std::system("lscpu>InfoSystemCPU.txt");
	Value=std::system("lshw -C display>InfoSystemGPU.txt");
	std::cout <<"\n";
    std::cout <<"\n";
}

void getInformationCPU()
{
	std::cout <<"\n";
	std::cout << "[INFO]: Information CPU"<<"\n";
    std::cout <<"\n";
	readFileViewInformation("InfoSystemCPU.txt");
	std::cout <<"\n";
    std::cout <<"\n";
}

void getInformationGPU()
{
	std::cout <<"\n";
	std::cout << "[INFO]: Information GPU"<<"\n";
    std::cout <<"\n";
	readFileViewInformation("InfoSystemGPU.txt");
	std::cout <<"\n";
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
      	std::cout << "[INFO]: MPI Name worlds processor: "<<processor_name<<"\n";
      	std::cout << "[INFO]: MPI Nb CPU available: "<<numCPU<< "\n";
      	std::cout <<"\n";
      	std::cout << "[INFO]: MPI Scan..."<<"\n";
    }
    std::cout << "[INFO]: MPI Rank: "<<world_rank<<" out of "<<world_size<<"\n";
    MPI_Finalize();
	//END::INFO MPI
}
#endif

#ifdef USE_OpenMP
void getOpenMPInformation()
{
    std::cout << "[INFO]: OpenMP Nb num procs: "<<omp_get_num_procs( )<< "\n";
    std::cout << "[INFO]: OpenMP Nb max threads: "<<omp_get_max_threads()<< "\n";
}
#endif

void getShortInformationGPU()
{
	int deviceCount=0;
	std::cout <<"\n";
	std::cout << "[INFO]: Information GPU"<<"\n";

	//#ifdef COMPILE_WITH_HIP && UseHIP
    #ifdef UseHIP
		hipGetDeviceCount(&deviceCount);
		if (deviceCount>0) {
			std::cout << "[INFO]: Number of available GPUs AMD: " << deviceCount << "\n";
			for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
				hipSetDevice(deviceId);
				std::cout << "[INFO]: GPU " << deviceId << " initialized and resources allocated." << "\n";
			}
		}
	#endif

	#ifdef COMPILE_WITH_CUDA && useCUDA
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
    //#ifdef COMPILE_WITH_HIP && UseHIP
    #ifdef UseHIP
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
    //(...)
    HIP_CHECK(hipSetDevice(0));
    std::cout<<std::endl;
    //END::INFO HIP AMD
    #endif
}

void getCudaInformation()
{
    //Nota: no code fusion because if hybrid CUDA and HIP system used
    //BEGIN::INFO CUDA NVIDIA
    #ifdef COMPILE_WITH_CUDA && UseCUDA
    std::cout<<std::endl;
    int numDevices=0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    std::cout<<"[INFO]: Get numDevice                = "<<numDevices<<"\n";
    int deviceID=0;
    CUDA_CHECK(cudaGetDevice(&deviceID));
    std::cout<<"[INFO]: Get deviceID activated       = "<<deviceID<<"\n";
    deviceID=0;
    cudaSetDevice(deviceID);

    hipDeviceProp_t devProp;
    for (int i = 0; i < numDevices; i++)
    {
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaGetDeviceProperties(&devProp,i));
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
     //(...)
    CUDA_CHECK(cudaSetDevice(0));
    std::cout<<std::endl;
    //END::INFO CUDA NVIDIA
    #endif
}


void getInformationSystem()
{
    std::cout<<"[INFO]: ======================================================================================== "<<"\n";
    std::cout<<"[INFO]: Get Information System "<<"\n";
    getInformationCPU();
    scanInformationSystem();
    getInformationGPU();

    //#ifdef COMPILE_WITH_HIP && UseHIP
    #ifdef UseHIP
    getHipInformation();
    #endif

    #ifdef COMPILE_WITH_CUDA && UseCUDA
    getCudaInformation();
    #endif
    std::cout<<"[INFO]: ======================================================================================== "<<"\n";
    std::cout<<"[INFO]: "<<"\n";
}

}//END::namespace LEM

//=======================================================================================================================
// Meta function tools allowing you to process an expression defined in Task
//=======================================================================================================================

constexpr auto& _parameters = NA::identifier<struct parameters_tag>;
constexpr auto& _task = NA::identifier<struct task_tag>;


constexpr auto& _kernel = NA::identifier<struct kernel_tag>;
constexpr auto& _range  = NA::identifier<struct range_tag>;
constexpr auto& _links  = NA::identifier<struct links_tag>;


namespace Backend{

    template<typename ...T, size_t... I>
    auto extractParametersAsTuple( std::tuple<T...> && t, std::index_sequence<I...>)
    {
        return std::forward_as_tuple( std::get<I>(t).getValue()...);
    }

    struct Runtime{
        template <typename ... Ts>
        void task(Ts && ... ts ) {
            auto t = std::make_tuple( std::forward<Ts>(ts)... );
            auto callback = std::get<sizeof...(Ts) - 1>(t);
            auto parameters = extractParametersAsTuple( std::move(t), std::make_index_sequence<sizeof...(Ts)-1>{} );
            std::apply( callback, std::move(parameters) );
        }
    };

    template <typename T,bool b>
    class SpData
    {
        static_assert(std::is_reference<T>::value,
                    "The given type must be a reference");
    public:
        using value_type = T;
        static constexpr bool isWrite = b;

        template <typename U, typename = std::enable_if_t<std::is_convertible_v<U,T>> >
        constexpr explicit SpData( U && u ) : M_val( std::forward<U>(u) ) {}

        constexpr value_type getValue() { return M_val; }
    private:
        value_type M_val;
    };

    template <typename T>
    auto spRead( T && t )
    {
        return SpData<T,false>{ std::forward<T>( t ) };
    }
    template <typename T>
    auto spWrite( T && t )
    {
        return SpData<T,true>{ std::forward<T>( t ) };
    }

    template<typename T>
    auto toSpData( T && t )
    {
        if constexpr ( std::is_const_v<std::remove_reference_t<T>> )
            return spRead( std::forward<T>( t ) );
        else
            return spWrite( std::forward<T>( t ) );
    }

    template<typename ...T, size_t... I>
    auto makeSpDataHelper( std::tuple<T...>& t, std::index_sequence<I...>)
    {
        return std::make_tuple( toSpData(std::get<I>(t))...);
    }
    template<typename ...T>
    auto makeSpData( std::tuple<T...>& t ){
        return makeSpDataHelper<T...>(t, std::make_index_sequence<sizeof...(T)>{});
    }

    template<typename T>
    auto toSpDataSpecx( T && t )
    {
        if constexpr ( std::is_const_v<std::remove_reference_t<T>> )
            return SpRead(std::forward<T>( t ));
        else
            return SpWrite(std::forward<T>( t ));
    }

    template<typename ...T, size_t... I>
    auto makeSpDataHelperSpecx( std::tuple<T...>& t, std::index_sequence<I...>)
    {
        return std::make_tuple( toSpDataSpecx(std::get<I>(t))...);
    }
    template<typename ...T>
    auto makeSpDataSpecx( std::tuple<T...>& t ){
        return makeSpDataHelperSpecx<T...>(t, std::make_index_sequence<sizeof...(T)>{});
    }



    template<typename T>
    auto toSpDataSpecxGPU( T && t )
    {
        if constexpr ( std::is_const_v<std::remove_reference_t<T>> )
            return SpRead(std::forward<T>( t ));
        else
            //return SpWrite(std::forward<T>( t ));
            return SpCommutativeWrite(std::forward<T>( t ));
    }

    template<typename ...T, size_t... I>
    auto makeSpDataHelperSpecxGPU( std::tuple<T...>& t, std::index_sequence<I...>)
    {
        return std::make_tuple( toSpDataSpecxGPU(std::get<I>(t))...);
    }
    template<typename ...T>
    auto makeSpDataSpecxGPU( std::tuple<T...>& t ){
        return makeSpDataHelperSpecxGPU<T...>(t, std::make_index_sequence<sizeof...(T)>{});
    }



}


namespace Frontend
{
    template <typename ... Ts>
    auto parameters(Ts && ... ts)
    {
        return std::forward_as_tuple( std::forward<Ts>(ts)... );
    }
}

//================================================================================================================================
// Tools to manage memories between CPU and GPU
//================================================================================================================================

#ifdef UseHIP
struct VectorBuffer {
      unsigned int dimension; 
      unsigned int dimensionSizeof; 
 	  unsigned int pitch; 
 	  double* data;

      /////////////////////////////////////////////////////////////

      class DataDescr {
        std::size_t size;
        public:
            explicit DataDescr(const std::size_t inSize = 0) : size(inSize){}

            auto getSize() const{
                return size;
            }
        };

        using DataDescriptor = DataDescr;

        std::size_t memmovNeededSize() const{
            return dimensionSizeof;
        }

        template <class DeviceMemmov>
            auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size){
                assert(size == dimensionSizeof);
                double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
                mover.copyHostToDevice(doubleDevicePtr, data, dimensionSizeof);
                return DataDescr(dimension);
            }

        template <class DeviceMemmov>
            void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size, const DataDescr& /*inDataDescr*/){
                assert(size == dimensionSizeof);
                double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
                mover.copyDeviceToHost(data, doubleDevicePtr, dimensionSizeof);
            }
};


template<typename ptrtype>
    struct bufferGraphGPU {
        unsigned int size; 
        ptrtype* data;
        ptrtype* deviceBuffer;

    void memoryInit(int dim)
    {
        size=dim; data=(ptrtype *)malloc(sizeof(ptrtype) * size);
    }

    void memmovHostToDevice()
    {
        hipMalloc((void **) &deviceBuffer, sizeof(ptrtype) * size);
        hipMemcpy(deviceBuffer,data,sizeof(ptrtype) * size, hipMemcpyHostToDevice);
    };

    void memmovDeviceToHost()
    {
        hipMemcpy(data,deviceBuffer,sizeof(ptrtype) * size, hipMemcpyDeviceToHost);
        hipFree(deviceBuffer);
    }

};
 #endif





//================================================================================================================================
// CLASS Task: Provide a family of multithreaded functions...
//================================================================================================================================

// Nota: The objective is to provide a range of tools in the case of using a single variable in multithreading.
// In the case of work with several variables use the class TasksDispatchComplex.

//#define USE_jthread



#ifdef UseHIP
    template<typename Kernel, typename Input, typename Output>
    __global__ void OP_IN_KERNEL_GPU_1D(const Kernel kernel_function, int n,Input* in, Output* out)
    {
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        if(i<n) {
            kernel_function(i, in, out);
        }
    }

    template<typename Kernel,typename Input>
    __global__ void OP_IN_KERNEL_LAMBDA_GPU_1D(Kernel op,Input *A, int nb) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        if (idx < nb)
            op(idx,A);
        //__syncthreads();
    }

    template<typename Kernel,typename Input>
    __global__ void OP_IN_KERNEL_LAMBDA_GPU_1D2I(Kernel op,Input *A,Input *B, int nb) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        if (idx < nb)
            op(idx,A,B);
        //__syncthreads();
    }
 #endif






void *WorkerInNumCPU(void *arg) {
    // Function used to run a task on a given CPU number.
    std::function<void()> *func = (std::function<void()>*)arg;
    (*func)();
    pthread_exit(NULL);
}


namespace LEM {

class Task
{
    private:
        int M_nbThTotal;                                       // variable indicator of the total number of threads available
        std::string M_FileName;                                // name of the recording file that will be used for the debriefing
        template <typename ... Ts>
        auto parameters(Ts && ... ts);                         // meta function to use multiple variables
        bool M_qEmptyTask;                                     // variable indicator if there is no task
        bool M_qFlagDetachAlert;                               // flag internal variable indicator for activation of the detach module
        
        //SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> M_mytg;    // Specx TaskGraph function 
        SpTaskGraph<SpSpeculativeModel::SP_MODEL_1> M_mytg;    // Specx TaskGraph function 
        SpComputeEngine M_myce;                                // Specx engine function 

        std::vector<int>               M_idType;                // This vector saves the thread type according to the task id number
        std::vector<int>               M_numTaskStatus;         // This vector saves the thread type status according to the task id number
        std::vector<std::future<bool>> M_myfutures;             // Vector table of std::future
        std::vector<std::future<bool>> M_myfuturesdetach;       // Vector table of std::future detach
        std::vector<std::thread>       M_mythreads;             // Vector table of std::thread
        std::vector<pthread_t>         M_mypthread_t;           // Vector table of pthread_t

        #ifdef COMPILE_WITH_CXX_20
            std::vector<std::jthread>      myjthreads;              // Vector table of <std::jthread
        #endif

        //pthread_t M_mypthread_t[100]; 
        pthread_attr_t                 M_mypthread_attr_t;      // An attribute set to supply to pthread_create()
        std::vector<int>               M_mypthread_cpu;         // Vector table of thread with
        std::mutex                     M_mymtx;                 // Mutex
        //std::promise<int> promise0;
        std::chrono::steady_clock::time_point M_t_begin,M_t_end;

        //BEGIN::GPU part

        //END::GPU part


        // variables indicatrices utilisées par les fonctions set et get. Voir les descriptions ci-dessous.
        long int M_t_laps;
        bool M_qFirstTask;
        int  M_idk;
        int  M_numLevelAction;
        int  M_nbThreadDetach;
        bool M_qReady;

        bool M_qCUDA;
        bool M_qHIP;

        int  M_nbTh;
        int  M_numTypeTh;

        bool M_qDetach;
        bool M_qYield;
        bool M_qDeferred;
        bool M_qUseIndex;

        bool M_qViewChrono;
        bool M_qInfo;
        bool M_qSave;

        //BEGIN::GPU part
        int  M_numBlocksGPU;
        int  M_nThPerBckGPU;
        int  M_numOpGPU;
        //END::GPU part

        template <typename ... Ts> 
            auto common(Ts && ... ts);

        void init(); // Function that initializes all variables we need
        

    public:

        //BEGIN::No copy and no move
        Task(const Task&) = delete;
        Task(Task&&) = delete;
        Task& operator=(const Task&) = delete;
        Task& operator=(Task&&) = delete;
        //END::No copy and no move

        //BEGIN::Small functions and variables to manage initialization parameters
        
        void setDetach           (bool b)  {  M_qDetach     = b; } // This function allows you to indicate to the next task whether it should be detached.
        void setYield            (bool b)  {  M_qYield      = b; } // This function allows you to indicate to the next task whether it should be yield.
        void setDeferred         (bool b)  {  M_qDeferred   = b; } // This function allows you to indicate to the next task whether it should be deferred. Used only for std::async part
        void setUseIndex         (bool b)  {  M_qUseIndex   = b; } // 
        void setSave             (bool b)  {  M_qSave       = b; } // Indicates whether to save debriefing data
        void setInfo             (bool b)  {  M_qInfo       = b; } // Allows Indicates whether we must display progress information during the execution of the class.
        void setViewChrono       (bool b)  {  M_qViewChrono = b; } // Allows Indicates whether to display time measurement information from the stopwatch.
        


        bool isDetach            () const  {  return(M_qDetach);     } // Indicates the state of the boolean variable isDetach
        bool isYield             () const  {  return(M_qYield);      } // Indicates the state of the boolean variable isYield 
        bool isDeferred          () const  {  return(M_qDeferred);   } // Indicates the state of the boolean variable isDeferred
        bool isSave              () const  {  return(M_qSave);       } // Indicates the state of the boolean variable isSave
        bool isInfo              () const  {  return(M_qInfo);       } // Indicates the state of the boolean variable isInfo 
        bool isViewChrono        () const  {  return(M_qViewChrono); } // Indicates the state of the boolean variable isViewChrono


        void setNbThread         (int v)   { M_nbTh=std::min(v,M_nbThTotal); } // Fix the desired number of threads
        int  getNbMaxThread      ()        { M_nbThTotal=std::thread::hardware_concurrency(); return(M_nbThTotal); } // Gives the maximum number of threads
        int  getNbThreads        () const  { int val=M_nbTh; if (M_numTypeTh==3) { val=static_cast<int>(M_myce.getCurrentNbOfWorkers()); } return val; } // Gives the number of threads used
        int  getNbCpuWorkers     () const  { int val=M_nbTh; if (M_numTypeTh==3) { val=static_cast<int>(M_myce.getNbCpuWorkers()); } return val; } //Gives the number of CpuWorkers used

        auto getIdThread         (int i);  // Gives the memory address of the thread used
        int  getNbThreadPerBlock (int i);  // Gives the number of threads used per GPU block

        void setNumOpGPU         (int v)  {  M_numOpGPU = v; } 


        long int  getTimeLaps ()        { return M_t_laps; } // Gives the time laps simulation


        #ifdef COMPILE_WITH_CUDA
            int getNbCudaWorkers() const {
                return static_cast<int>(M_myce.getNbCudaWorkers());
                } // Returns the total number of Cuda cards
        #endif
        #ifdef COMPILE_WITH_HIP
            int getNbHipWorkers() const {
                return static_cast<int>(M_myce.getNbHipWorkers());
            } // Returns the total number of Hip cards
        #endif
            
        void setFileName(std::string s) { M_FileName=s; } 
        //END::Small functions and variables to manage initialization parameters


        void subTask(const int nbThread,const int nbBlocks,int M_numTypeThread);
           

        Task(void);  // Constructor, initializes the default variables that are defined in the init() function.
        ~Task(void); // Destructor is invoked automatically whenever an object is going to be destroyed. Closes the elements properly


        #if defined(COMPILE_WITH_HIP) || defined(COMPILE_WITH_CUDA)
            #ifdef COMPILE_WITH_HIP
                explicit Task(const int nbThread,const int nbBlocks,int M_numTypeThread):M_mytg(),M_myce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers()) // Class constructor in classic mode
                {
                    std::cout<<"[INFO]: WELCOME TO GPU:HIP"<<"\n";
                    subTask(nbThread,nbBlocks,M_numTypeThread);
                }


                explicit Task(const int nbThread,int M_numTypeThread):M_mytg(),M_myce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers()) // Class constructor in classic mode
                {
                    std::cout<<"[INFO]: WELCOME TO GPU:HIP"<<"\n";
                    subTask(nbThread,1,M_numTypeThread);
                }


            #endif

            #ifdef COMPILE_WITH_CUDA
                explicit Task(const int nbThread,const int nbBlocks,int M_numTypeThread):M_mytg(),M_myce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers()) // Class constructor in classic mode
                {
                    std::cout<<"[INFO]: WELCOME TO GPU:CUDA"<<"\n";
                    subTask(nbThread,nbBlocks,M_numTypeThread);
                }
            #endif
        #else
            explicit Task(const int nbThread,int M_numTypeThread):M_mytg(),M_myce(SpWorkerTeamBuilder::TeamOfCpuWorkers(nbThread)) // Class constructor in classic mode
            {
                std::cout<<"[INFO]: WELCOME TO CPU"<<"\n";
                subTask(nbThread,1,M_numTypeThread);
            }
        #endif

        template <class ClassFunc> void execOnWorkers(ClassFunc&& func) { M_myce.execOnWorkers(std::forward<ClassFunc>(func)); } //Execute a ClassFunc on workers
  
        void setSpeculationTest(std::function<bool(int,const SpProbability&)> inFormula){
            if (M_numTypeTh==3) { M_mytg.setSpeculationTest(std::move(inFormula)); }
        }

        template <typename ... Ts>
        void add( Ts && ... ts ); // This main function allows you to add a task

            template <typename ... Ts>
                void addTaskSimple( Ts && ... ts ); // This subfunction allows you to add a simple task

            template <typename ... Ts>
                void addTaskSpecx( Ts && ... ts ); // This subfunction allows you to add a specx task

            template <typename ... Ts>
                void addTaskAsync( Ts && ... ts ); // This subfunction allows you to add a std::async task

            template <typename ... Ts>
                void addTaskMultithread( Ts && ... ts ); // This subfunction allows you to add a multithread task

            
            #ifdef COMPILE_WITH_CXX_20
            template <typename ... Ts>
                void addTaskjthread( Ts && ... ts ); // This subfunction allows you to add a jthread task. Only works under C++20
            #endif

        #ifdef COMPILE_WITH_HIP
            template <typename ... Ts>
                void add_GPU( Ts && ... ts); // This main function allows you to add a task O:CPU Normal  1:SpHip  2:SpCuda  3:GPU to CPU
        #endif

        template <class InputIterator,typename ... Ts>
            void for_each(InputIterator first, InputIterator last,Ts && ... ts); // This function allows you to apply the same treatment to a set of elements of a task.

        
        template <typename ... Ts>
            void add(int numCPU,Ts && ... ts);  // Add a task on specific CPU number

        template<typename FctDetach>
            auto add_detach(FctDetach&& func) -> std::future<decltype(func())>; // Add a detach thread task

        template <typename ... Ts>
            void runInCPUs(const std::vector<int> & numCPU,Ts && ... ts); // Execute all tasks on specific CPU number
        
        void run();                     // Execution of all added tasks. 
        void close();                   // Memory cleanup of all variables used before closing the class.
        void debriefingTasks();         // Wrote a report on execution times and generated .dot .svg files regarding Specx and .csv to save the times.
        void getInformation();          // Provides all information regarding graphics cards (CUDA and HIP)
        void getGPUInformationError();  // Provides error types from the GPU (CUDA and HIP)

        
        //GPU-AMD-CUDA

        #ifdef USE_GPU_HIP
        // set of functions allowing you to use eigen under Hip gpu.
        template<typename Kernel, typename Input, typename Output>
            void run_gpu_1D(const Kernel& kernel_function,dim3 blocks,int n,const Input& in,Output& out);
        template<typename Kernel, typename Input, typename Output>
            void run_gpu_2D(const Kernel& kernel_function,dim3 blocks,int n,const Input& in,Output& out);
        template<typename Kernel, typename Input, typename Output>
            void run_gpu_3D(const Kernel& kernel_function,dim3 blocks,int n,const Input& in,Output& out);
        // set of functions allowing you to use eigen under Hip cpu. Juste to control the results
        template<typename Kernel, typename Input, typename Output>
            void run_cpu_1D(const Kernel& kernel_function, int n, const Input& in, Output& out);
        #endif

        #ifdef UseHIP

        #endif

        #if defined(COMPILE_WITH_HIP) || defined(COMPILE_WITH_CUDA)

        template <class... ParamsTy>
            void addTaskSpecxPureGPU(ParamsTy&&...params); 

        template <typename ... Ts>
            void addTaskSpecxGPU( Ts && ... ts); // This subfunction allows you to add a specx task

        #endif

};



Task::Task()
{
    init();
}

Task::~Task()
{
    //Add somes   
    if ((M_numTypeTh== 3) && (M_numLevelAction==3)) {  M_myce.stopIfNotAlreadyStopped(); } 
    if ((M_numTypeTh==33) && (M_numLevelAction==3)) {  M_myce.stopIfNotAlreadyStopped(); } // <== [ ] see if we really need it
    //Specx
}


void Task::init()
{
    M_nbThTotal=std::thread::hardware_concurrency();
    M_nbTh             = M_nbThTotal;
    M_qInfo            = true;
    M_qSave            = false;
    M_qDeferred        = false;
    M_numTypeTh        = 0;
    M_qUseIndex        = false;
    M_FileName         ="NoName";
    M_qFirstTask       = true;
    M_idk              = 0;
    M_numLevelAction   = 0;
    M_qCUDA            = false;
    M_qHIP             = false;
    M_qEmptyTask       = true;
    M_qFlagDetachAlert = false;
    M_nbThreadDetach   = 0;
    M_qReady           = false;
    M_qYield           = false;
    M_numBlocksGPU     = 128; //<- see after

    M_numOpGPU         = 0;
    
    M_mythreads.clear();
    M_myfutures.clear();
    
    #ifdef COMPILE_WITH_CXX_20
    myjthreads.clear();
    #endif
}


void Task::subTask(const int nbThread,const int nbBlocks,int M_numTypeThread)
{
    M_numBlocksGPU=nbBlocks;
    M_nbTh=nbThread;
    M_numTypeTh = M_numTypeThread;
    M_idk=0; M_numTaskStatus.clear(); M_qDetach=0; M_nbThreadDetach=0;
    M_idType.clear(); 
    M_t_begin   = std::chrono::steady_clock::now();
    M_qDeferred = false;
    M_qReady    = false;
                
    if (M_numTypeTh==0) { } // No Thread
    if (M_numTypeTh==1) { } // multithread
    if (M_numTypeTh==2) { } // std::async
    if (M_numTypeTh==3) { M_mytg.computeOn(M_myce); } // Specx

    if (M_numTypeTh==10) { pthread_attr_init(&M_mypthread_attr_t); } //pthread

    if (M_numTypeTh==33) { 
        #ifdef COMPILE_WITH_HIP
            //static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,"should be stdvec"); //will see after
            M_mytg.computeOn(M_myce);
        #endif
    } // Specx GPU
}

int Task::getNbThreadPerBlock(int i)
{
    int numDevices=0;
    #ifdef COMPILE_WITH_HIP
        hipGetDeviceCount(&numDevices);
        hipDeviceProp_t devProp;
        if ((i>=0) && (i<numDevices))
        {
            HIP_CHECK(hipGetDeviceProperties(&devProp,i));
            return(devProp.maxThreadsPerBlock);
        }
    #endif

    #ifdef COMPILE_WITH_CUDA
        cudaGetDeviceCount(&numDevices);
        cudaDeviceProp_t devProp;
        if ((i>=0) && (i<numDevices))
        {
            cudaGetDeviceProperties(&devProp,i);
            return(devProp.maxThreadsPerBlock);
        }
    #endif
}



void Task::getInformation()
{
    // Provides all information regarding graphics cards CUDA and AMD
    if (M_qInfo)
    {
        if (M_numTypeTh== 0) { std::cout<<"[INFO]: Mode No Thread\n"; }
        if (M_numTypeTh== 1) { std::cout<<"[INFO]: Mode Multithread\n"; }
        if (M_numTypeTh== 2) { std::cout<<"[INFO]: Mode Std::async\n"; }
        if (M_numTypeTh== 3) { std::cout<<"[INFO]: Mode Specx\n"; }

        if (M_numTypeTh==10) { std::cout<<"[INFO]: Mode Thread in CPU\n"; }
        M_nbThTotal=getNbMaxThread(); 
        std::cout<<"[INFO]: Nb max Thread="<<M_nbThTotal<<"\n";
        getInformationSystem();
    }
}


void Task::getGPUInformationError()
{
    #ifdef COMPILE_WITH_CUDA
        cudaError_t num_err = cudaGetLastError();
        if (num_err != cudaSuccess) {
            std::cout<<"[INFO]: hip Error Name ="<<cudaGetErrorName(num_err)<<" "<<cudaGetErrorString(num_err)<<"\n";
        }
        num_err = cudaDeviceSynchronize();
        if (num_err != cudaSuccess) {
            std::cout<<"[INFO]: hip Error Name="<<cudaGetErrorName(err)<<" "<<cudaGetErrorString(num_err)<<"\n";
        }
    #endif

    #ifdef COMPILE_WITH_HIP
        hipError_t num_err = hipGetLastError();
        if (num_err != hipSuccess) {
            std::cout<<"[INFO]: hip Error Name ="<<hipGetErrorName(num_err)<<" "<<hipGetErrorString(num_err)<<"\n";
        }
        num_err = hipDeviceSynchronize();
        if (num_err != hipSuccess) {
            std::cout<<"[INFO]: hip Error Name="<<hipGetErrorName(num_err)<<" "<<hipGetErrorString(num_err)<<"\n";
        }
    #endif
}


#ifdef USE_GPU_HIP
template<typename Kernel, typename Input, typename Output>
void Task::run_gpu_1D(const Kernel& kernel_function,dim3 blocks,int n, const Input& in, Output& out)
{
    bool qInfo_GPU_process=true;
    if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
    std::chrono::steady_clock::time_point M_t_begin_all_GPU_process,M_t_end_all_GPU_process;
    std::chrono::steady_clock::time_point M_t_begin_inside,M_t_end_inside;

    M_t_begin_all_GPU_process = std::chrono::steady_clock::now();
        typename Input::Scalar*  d_in;
        typename Output::Scalar* d_out;
        std::ptrdiff_t in_bytes  = in.size()  * sizeof(typename Input::Scalar);
        std::ptrdiff_t out_bytes = out.size() * sizeof(typename Output::Scalar);
        
        HIP_ASSERT(hipMalloc((void**)(&d_in),  in_bytes));
        HIP_ASSERT(hipMalloc((void**)(&d_out), out_bytes));
        
        HIP_ASSERT(hipMemcpy(d_in,  in.data(),  in_bytes,  hipMemcpyHostToDevice));
        HIP_ASSERT(hipMemcpy(d_out, out.data(), out_bytes, hipMemcpyHostToDevice));
        
        //dim3 blocks(128);
        //dim3 grids( (n+int(blocks.x)-1)/int(blocks.x) );
        int num_blocks = (n + blocks.x - 1) / blocks.x;
	    dim3 thread_block(blocks.x, 1, 1);
	    dim3 grid(num_blocks,1,1);

        hipDeviceSynchronize();
            M_t_begin_inside = std::chrono::steady_clock::now();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(
                OP_IN_KERNEL_GPU_1D<Kernel,typename std::decay<decltype(*d_in)>::type,typename std::decay<decltype(*d_out)>::type>), 
                        grid, thread_block, 0, 0, kernel_function, n, d_in, d_out);

            M_t_end_inside = std::chrono::steady_clock::now();
        getGPUInformationError();

        hipMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  hipMemcpyDeviceToHost);
        hipMemcpy(out.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
        HIP_ASSERT(hipFree(d_in));
        HIP_ASSERT(hipFree(d_out));    
    M_t_end_all_GPU_process = std::chrono::steady_clock::now();

    long int M_t_laps3= std::chrono::duration_cast<std::chrono::microseconds>(M_t_end_inside - M_t_begin_inside).count();
    long int M_t_laps2= std::chrono::duration_cast<std::chrono::microseconds>(M_t_end_all_GPU_process - M_t_begin_all_GPU_process).count();
    if (qInfo_GPU_process) {
        std::cout << "[INFO]: Elapsed microseconds inside: "<<M_t_laps3<< " us\n";    
        std::cout << "[INFO]: Elapsed microseconds inside + memory copy: "<<M_t_laps2<< " us\n";
        std::cout << "[INFO]: nb grids: "<<grid.x<<" "<<grid.y<<" "<<grid.z<< "\n";
        std::cout << "[INFO]: nb block: "<<thread_block.x<<" "<<thread_block.y<<" "<<thread_block.z<< "\n";
    }
}

template<typename Kernel, typename Input, typename Output>
void Task::run_gpu_2D(const Kernel& kernel_function,dim3 blocks,int n, const Input& in, Output& out)
{
    if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
}

template<typename Kernel, typename Input, typename Output>
void Task::run_gpu_3D(const Kernel& kernel_function,dim3 blocks,int n, const Input& in, Output& out)
{
    if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
}


template<typename Kernel, typename Input, typename Output>
void Task::run_cpu_1D(const Kernel& kernel_function, int n, const Input& in, Output& out)
{
  if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
  for(int i=0; i<n; i++)
    kernel_function(i, in.data(), out.data());
}
#endif







#if defined(COMPILE_WITH_HIP) || defined(COMPILE_WITH_CUDA)

template <class... ParamsTy>
void Task::addTaskSpecxPureGPU(ParamsTy&&...params)
{
      M_mytg.task(std::forward<ParamsTy>(params)...);
}


template <typename ... Ts>
void Task::addTaskSpecxGPU( Ts && ... ts)
{
    //=33
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    
    //std::cout<<"GET Task="<<std::get<0>(args)<<"\n";
    //std::cout<<"GET Parameters="<<std::get<0>(parameters)<<"\n";



    auto tpBackend=Backend::makeSpDataSpecxGPU( parameters );
    int NbtpBackend=std::tuple_size<decltype(tpBackend)>::value;
    std::cout <<"Size Parameters="<<NbtpBackend<< std::endl;

    auto LambdaExpression=std::make_tuple(task);
    int NbLambdaExpression=std::tuple_size<decltype(LambdaExpression)>::value;
    std::cout <<"Size Tuple Task="<<NbLambdaExpression<< std::endl;

    //auto tpHip=std::tuple_cat(tpBackend,LambdaExpression);  

    //M_mytg.task(std::get<0>(tpBackend),std::get<1>(tpBackend),SpHip(task));


    //std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tpHip);    
    //addTaskSpecxPureGPU(tpHip);   



    if (!M_qDetach)
    {
        if (M_numOpGPU==0) { 
            
            auto tp=std::tuple_cat( Backend::makeSpDataSpecxGPU( parameters ), std::make_tuple( task ));  
            std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tp);       
        } //CPU Normal


        if (M_numOpGPU==1)
        { 
            //auto tp=std::tuple_cat( Backend::makeSpDataSpecxGPU( parameters ), SpHip(std::make_tuple(task))); 
            //std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tp);
        } //op in Hip


/*

        if (M_numOpGPU==2) { 
            auto tp=std::tuple_cat( Backend::makeSpDataSpecxGPU( parameters ), SpCuda(std::make_tuple( task )));   
            std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tp);  
        } //op in Cuda

        if (M_numOpGPU==3) { 
            auto tp=std::tuple_cat( Backend::makeSpDataSpecx( parameters ), SpCPU(std::make_tuple( task )));
            std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tp);
        } //op in GPU to CPU
*/
        usleep(0); std::atomic_int counter(0);
    }
    else
    {
        addTaskAsync(std::forward<Ts>(ts)...);
    }
}
#endif





template <typename ... Ts>
auto Task::parameters(Ts && ... ts)
{
    return std::forward_as_tuple( std::forward<Ts>(ts)... );
}

template <typename ... Ts>
    auto Task::common(Ts && ... ts)
{
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    auto tp=std::tuple_cat(  Backend::makeSpData( parameters ), std::make_tuple( task ) );
    return(tp);
}


template <typename ... Ts>
void Task::addTaskSimple( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
    std::apply( [&runtime](auto... args){ runtime.task(args...); }, tp );
}


template <typename ... Ts>
void Task::addTaskMultithread( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
	auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
	};

    if (!M_qDetach)
    {
        std::thread th(LamdaTransfert);
        M_mythreads.push_back(std::move(th));
    }
    else
    {
        if (M_qInfo) { std::cout<<"[INFO]: detach in process...\n"; }
        M_myfuturesdetach.emplace_back(add_detach(LamdaTransfert)); 
    }
    usleep(1);
}


template <typename ... Ts>
void Task::addTaskAsync( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
    auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
		};

    if (!M_qDetach)
    {
        if (M_qDeferred) { M_myfutures.emplace_back(std::async(std::launch::deferred,LamdaTransfert));}
        else             { M_myfutures.emplace_back(std::async(std::launch::async,LamdaTransfert)); }
    }
    else
    {
        if (M_qInfo) { std::cout<<"[INFO]: detach in process...\n"; }
        M_myfuturesdetach.emplace_back(add_detach(LamdaTransfert)); 
    }

    usleep(1);
}



template <typename ... Ts>
void Task::addTaskSpecx( Ts && ... ts )
{
    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    auto tp=std::tuple_cat( 
					Backend::makeSpDataSpecx( parameters ), 
					std::make_tuple( task ) 
				);
    if (!M_qDetach)
    {
        std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tp);
        usleep(0); std::atomic_int counter(0);
    }
    else
    {
        addTaskAsync(std::forward<Ts>(ts)...);
    }
}

#ifdef COMPILE_WITH_CXX_20
template <typename ... Ts>
void Task::addTaskjthread( Ts && ... ts )
{
    auto tp=common(std::forward<Ts>(ts)...);
    Backend::Runtime runtime;
	auto LamdaTransfert = [&]() {
			std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
	};

    if (!M_qDetach)
    {
        std::jthread th(LamdaTransfert);
        myjthreads.push_back(std::move(th));
    }
    else
    {
        if (M_qInfo) { std::cout<<"[INFO]: detach in process...\n"; }
        M_myfuturesdetach.emplace_back(add_detach(LamdaTransfert)); 
    }
    usleep(1);
}
#endif


template <typename ... Ts>
void Task::add( Ts && ... ts )
{
    M_numLevelAction=1;
    M_qEmptyTask=false;
    M_idk++; M_idType.push_back(M_numTypeTh); M_numTaskStatus.push_back(M_qDetach);
    if (M_qDetach) { M_qFlagDetachAlert=true; M_nbThreadDetach++; }
    if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
    //std::cout<<"M_numTypeTh="<<M_numTypeTh<<"\n";
    switch(M_numTypeTh) {
        case 1: addTaskMultithread(std::forward<Ts>(ts)...); //multithread
        break;
        case 2: addTaskAsync(std::forward<Ts>(ts)...); //std::async
        break;
        case 3: addTaskSpecx(std::forward<Ts>(ts)...); //Specx
        break;
        #ifdef COMPILE_WITH_CXX_20
            case 4: addTaskjthread(std::forward<Ts>(ts)...); //std::jthread
            break;
        #endif
        default: addTaskSimple(std::forward<Ts>(ts)...); //No Thread
    }
    M_qDetach=false;
    //std::cout<<"[INFO] :ADD OK"<<std::endl;
}

#ifdef COMPILE_WITH_HIP
template <typename ... Ts>
void Task::add_GPU( Ts && ... ts)
{
    M_numLevelAction=1;
    M_qEmptyTask=false;
    M_idk++; M_idType.push_back(M_numTypeTh); M_numTaskStatus.push_back(M_qDetach);
    if (M_qDetach) { M_qFlagDetachAlert=true; M_nbThreadDetach++; }
    if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
    switch(M_numTypeTh) {
        case 33: addTaskSpecxGPU(std::forward<Ts>(ts)...); //Specx GPU
        break;
    }
    M_qDetach=false;
}
#endif


auto Task::getIdThread(int i)
{
    if ((i>0) && (i<M_idType.size()))
    {
        int nb=M_idType.size();
        int numM_idType=M_idType[i];
        int k=0; int ki=-1; bool qOn=true;
        while (qOn)
        {
            if (M_idType[k]==numM_idType) { ki++; }
            k++; if (k>i) { qOn=false; }
        }

        switch(numM_idType) {
            case 1: return(M_mythreads[ki].get_id());//multithread
            break;
            case 2:  //std::async
            break;
            case 3: //Specx
            break;
            #ifdef COMPILE_WITH_CXX_20
            case 4: return(myjthreads[ki].get_id()); //std::jtread
            break;
            #endif

        }
    }
    return(M_mythreads[0].get_id());
}


template <class InputIterator,typename ... Ts>
    void Task::for_each(InputIterator first, InputIterator last,Ts && ... ts)
{
    M_qUseIndex=true; //Iterator used
    M_numLevelAction=1;
    M_qEmptyTask=false;
    if (M_qFirstTask) { M_t_begin = std::chrono::steady_clock::now(); M_qFirstTask=false;}
    for ( ; first!=last; ++first )
    {
        M_idk++; M_idType.push_back(M_numTypeTh);
        auto const& ivdk = *first;
        //std::cout <<ivdk;

        auto args = NA::make_arguments( std::forward<Ts>(ts)... );
        auto && task = args.get(_task);
        auto && parameters = args.get_else(_parameters,std::make_tuple());
        if (M_qUseIndex) { std::get<0>(parameters)=std::cref(ivdk); } 
        auto tp=std::tuple_cat( 
					Backend::makeSpData( parameters ), 
					std::make_tuple( task ) 
				);

        Backend::Runtime runtime;

        if (M_numTypeTh==0) {
            std::apply( [&runtime](auto... args){ runtime.task(args...); }, tp );
        }

        if (M_numTypeTh==1) {
            auto LamdaTransfert = [&]() {
			    std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
            };
            std::thread th(LamdaTransfert);
            M_mythreads.push_back(std::move(th));
            usleep(1);
        }

        if (M_numTypeTh==2) {
            auto LamdaTransfert = [&]() {
			    std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
		    };

            if (M_qDeferred) { M_myfutures.emplace_back(std::async(std::launch::deferred,LamdaTransfert));}
            else             { M_myfutures.emplace_back(std::async(std::launch::async,LamdaTransfert)); }
            usleep(1);
        }

        if (M_numTypeTh==3) {
            auto tp=std::tuple_cat( 
					Backend::makeSpDataSpecx( parameters ), 
					std::make_tuple( task ) 
			);
            std::apply([&](auto &&... args) { M_mytg.task(args...).setTaskName("Op("+std::to_string(M_idk)+")"); },tp);
            usleep(0); std::atomic_int counter(0);
        }

        #ifdef COMPILE_WITH_CXX_20
        if (M_numTypeTh==4) {
            auto LamdaTransfert = [&]() {
			    std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
            return true; 
            };
            std::jthread th(LamdaTransfert);
            myjthreads.push_back(std::move(th));
            usleep(1);
        }
        #endif

    }
}


void Task::run()
{
    if (M_qEmptyTask) { std::cout<<"[INFO]: Run failed empty task\n"; exit(0); }
    M_numLevelAction=2;
    if (M_qInfo) { std::cout<<"[INFO]: Run\n"; }
    if (M_numTypeTh==0) { } //No Thread
    if (M_numTypeTh==1) { 
        for (std::thread &t : M_mythreads) { t.join();} 
    } //multithread

    if (M_numTypeTh==2) { 
        for( auto& r : M_myfutures){ auto a =  r.get(); }; 
    } //std::async

    if (M_numTypeTh==3) { 
        /*promise0.set_value(0);*/ 
        M_mytg.waitAllTasks();
    } //Specx

    #ifdef COMPILE_WITH_CXX_20
    if (M_numTypeTh==4) { 
        for (std::jthread &t : myjthreads) { t.join();} 
    } //std::jthread
    #endif

    if (M_numTypeTh==10) { 
        for (int i = 0; i < M_mypthread_cpu.size(); i++) { 
            std::cout<<"[INFO]: Joint "<<i<<"\n";
            pthread_join(M_mypthread_t[i], NULL); 
        }
        //pthread_attr_destroy(&M_mypthread_attr_t);
    } //In CPU

     if (M_numTypeTh==33) { 
        #ifdef COMPILE_WITH_HIP
            M_mytg.waitAllTasks();
        #endif
    } //Specx GPU


    M_mythreads.clear();
    M_myfutures.clear();
    #ifdef COMPILE_WITH_CXX_20
    myjthreads.clear();
    #endif

    M_numTaskStatus.clear();
    M_idType.clear();

    if (M_qInfo) { std::cout<<"[INFO]: All Tasks Accomplished\n"; }
    M_qEmptyTask=true;
}

void Task::close()
{
    if (M_qFlagDetachAlert) {
        if (M_qInfo) { std::cout<<"[INFO]: Detach processes are still running...\n"; }
        if (M_qInfo) { std::cout<<"[INFO]: Please wait before closing.\n"; }

        if ((M_numTypeTh>0) && (M_numTypeTh<10))
        {
            if (M_myfuturesdetach.size()>0)
            {
                for( auto& r : M_myfuturesdetach) {
                    //std::cout <<"Detach Status=" <<r.valid() << '\n';
                    r.wait(); 
                }
            }
        }      
    }

    M_numLevelAction=3;
    if (M_numTypeTh== 0) {  } //No Thread
    if (M_numTypeTh== 1) {  } //multithread
    if (M_numTypeTh== 2) {  } //std::async
    if (M_numTypeTh== 3) { M_myce.stopIfNotAlreadyStopped(); } //Specx
    if (M_numTypeTh==33) { M_myce.stopIfNotAlreadyStopped(); } //Specx GPU <== see if we really need it

    if (!M_qFirstTask) { M_t_end = std::chrono::steady_clock::now(); M_qFirstTask=true;}

    if (M_myfuturesdetach.size()>0) { M_myfuturesdetach.clear(); }

    if (M_qInfo) { std::cout<<"[INFO]: Close All Tasks and process\n"; }
    M_idk=0;
}

void Task::debriefingTasks()
{
    M_t_laps=std::chrono::duration_cast<std::chrono::microseconds>(M_t_end - M_t_begin).count();

    if (M_qViewChrono) {  
        if (M_qInfo) { std::cout << "[INFO]: Elapsed microseconds: "<<M_t_laps<< " us\n"; }
    }

    if (M_qSave)
    {
        if ((M_numTypeTh==3) || (M_numTypeTh==33)) { 
            std::cout << "[INFO]: Save "<<M_FileName<< "\n";
            M_mytg.generateDot(M_FileName+".dot",true); 
            M_mytg.generateTrace(M_FileName+".svg",true); 
        }

        std::ofstream myfile;
        myfile.open (M_FileName+".csv");
        myfile << "Elapsed microseconds,"<< M_t_laps<<"\n";
        myfile << "Nb max Thread,"<< M_nbThTotal<<"\n";
        myfile << "Nb Thread used,"<< M_nbTh<<"\n";
        myfile << "Nb Thread Detach used,"<<M_nbThreadDetach<<"\n";
        myfile << "Mode,"<< M_numTypeTh<<"\n";
        myfile.close();
    }
} 


template<typename FctDetach>
auto Task::add_detach(FctDetach&& func) -> std::future<decltype(func())>
{
    auto task   = std::packaged_task<decltype(func())()>(std::forward<FctDetach>(func));
    auto future = task.get_future();
    std::thread(std::move(task)).detach();
    return std::move(future);
}

template <typename ... Ts>
void Task::add(int numCPU,Ts && ... ts)
{
    //Run in num CPU
    M_mypthread_cpu.push_back(numCPU);
    pthread_t new_thread;
    

    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    auto tp=std::tuple_cat( 
            Backend::makeSpData( parameters ), 
                      std::make_tuple( task ) 
    );
    auto LamdaTransfert = [&]() {
                std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
                return true; 
    };


    std::function<void()> func =LamdaTransfert;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(numCPU, &cpuset);
    pthread_attr_setaffinity_np(&M_mypthread_attr_t, sizeof(cpuset), &cpuset);
    int ret=pthread_create(&new_thread,&M_mypthread_attr_t,WorkerInNumCPU,&func);
    if (ret) { std::cerr << "Error in creating thread" << std::endl; }

    M_mypthread_t.push_back(new_thread);

    //pthread_attr_destroy(&M_mypthread_attr_t);

    //vectorOfThreads.resize(NUM_THREADS);
}




template <typename ... Ts>
void Task::runInCPUs(const std::vector<int> & numCPU,Ts && ... ts)
{
    int M_nbTh=numCPU.size();
    pthread_t thread_array[M_nbTh];
    pthread_attr_t pta_array[M_nbTh];

    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && task = args.get(_task);
    auto && parameters = args.get_else(_parameters,std::make_tuple());
    Backend::Runtime runtime;
    M_qUseIndex=true;
    
    for (int i = 0; i < M_nbTh; i++) {
        int const& M_idk = i;
        if (M_qUseIndex) { std::get<0>(parameters)=M_idk; }
        auto tp=std::tuple_cat( 
                Backend::makeSpData( parameters ), 
                        std::make_tuple( task ) 
        );

        auto LamdaTransfert = [&]() {
                    std::apply([&runtime](auto... args){ runtime.task(args...); }, tp);
                    return true; 
        };
        std::function<void()> func =LamdaTransfert;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(numCPU[i], &cpuset);
        std::cout<<"Num CPU="<< numCPU[i] <<" activated"<<std::endl;
        pthread_attr_init(&pta_array[i]);
        pthread_attr_setaffinity_np(&pta_array[i], sizeof(cpuset), &cpuset);
        if (pthread_create(&thread_array[i],&pta_array[i],WorkerInNumCPU,&func)) { std::cerr << "Error in creating thread" << std::endl; }
    }

    for (int i = 0; i < M_nbTh; i++) {
            pthread_join(thread_array[i], NULL);
    }

    for (int i = 0; i < M_nbTh; i++) {
            pthread_attr_destroy(&pta_array[i]);
    }
}

}



//================================================================================================================================
// CLASS Task: Provide to manipulate graph Hip/Cuda hybrid system
//================================================================================================================================

#ifdef UseHIP
    template<typename Kernel,typename Input>
    __global__ void OP_IN_KERNEL_GRAPH_LAMBDA_GPU_1D(Kernel op,Input *A,int iBegin,int iEnd) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        if ((idx>=iBegin) && (idx<iEnd))
            op(idx,A);
        //__syncthreads();
    }
#endif



//--------------------------------------------------------------------------------------------------------------------------------
namespace LEMGPU {

class Task
{
    private:
        std::string M_FileName;
        int         M_nbTh;
        int         M_numBlocksGPU;
        int         M_nThPerBckGPU;
        bool        M_q_graph;
        bool        M_qViewInfo;
        bool        M_qSave;
        long int    M_time_laps;
        bool        M_qDeviceReset;

        std::vector<int>  M_ListGraphDependencies;

        
        std::chrono::steady_clock::time_point M_t_begin,M_t_end;

        //#ifdef COMPILE_WITH_HIP && UseHIP
        #ifdef UseHIP
            hipGraph_t                          hip_graph;
            hipGraphExec_t                      hip_graphExec;
            hipStream_t                         hip_graphStream;
            hipKernelNodeParams                 hip_nodeParams;
            std::vector<hipGraphNode_t>         M_hipGraphNode_t;     
        #endif

        #ifdef UseCUDA
            cudaGraph_t                          cuda_graph;
            cudaGraphExec_t                      cuda_graphExec;
            cudaStream_t                         cuda_graphStream;
            cudaKernelNodeParams                 cuda_nodeParams;
            std::vector<cudaGraphNode_t>         M_cudaGraphNode_t;     
        #endif
            

    public:
        

        Task();
        ~Task();

        void setSave         (bool b)        {  M_qSave = b; }
        void setViewInfo     (bool b)        {  M_qViewInfo = b; }
        void setDeviceReset  (bool b)        {  M_qDeviceReset = b; }
        bool isSave          () const        {  return(M_qSave); }
        bool isDeviceReset   () const        {  return(M_qDeviceReset); }
        void setFileName     (std::string s) {  M_FileName=s; }

        void setDeviceHIP    (int v);
        void setDeviceCUDA   (int v);
           


        void open(int nbBlock,int NbTh);

        template<typename Kernel, typename Input, typename Output>
            void add(const Kernel& kernel_function,
                     int numElems,
                     int iBegin,int iEnd,
                     Input* buffer,
                     Output* hostbuffer,
                     std::vector<int> links);

        template<typename Kernel, typename Input, typename Output>
            void add_cuda(const Kernel& kernel_function,
                     int numElems,
                     int iBegin,int iEnd,
                     Input* buffer,
                     Output* hostbuffer,
                     std::vector<int> links);

        template<typename Kernel, typename Input>
            void single_hip(const Kernel& kernel_function,
                                int numElems,
                                int iBegin,int iEnd,
                                Input* buffer);

        template<typename Kernel, typename Input>
            void single_cuda(const Kernel& kernel_function,
                                int numElems,
                                int iBegin,int iEnd,
                                Input* buffer);

        void run();
        void close();
        void debriefing(); 
};

Task::Task()
{
    M_nbTh         = 1;
    M_numBlocksGPU = 1;
    M_q_graph      = false;
    M_qViewInfo    = true;
    M_time_laps    = 0;
    M_FileName     = "NoName";
    M_qSave        = true;
    M_qDeviceReset = false;
    M_ListGraphDependencies.clear();
}

Task::~Task()
{
    M_ListGraphDependencies.clear();
}

void Task::debriefing()
{
    if (M_qViewInfo) { 
        std::cout<<"<=====================================================================>"<<"\n"; 
        std::cout<<"[INFO]: Debriefing"<<"\n";  
        std::cout<<"[INFO]: Elapsed microseconds : "<<M_time_laps<< " us\n";
        std::cout<<"[INFO]: List Graph Dependencie  >>> [";
        for (int i = 0; i < M_ListGraphDependencies.size(); i++) { std::cout<<M_ListGraphDependencies[i];}
        std::cout<<"] <<<\n";
    }

    if (M_qSave)
    {
        if (M_qViewInfo) { std::cout<<"[INFO]: Save Informations"<<"\n"; }   
        std::ofstream myfile;
        myfile.open (M_FileName+".csv");
        myfile << "Elapsed microseconds,"<<M_time_laps<<"\n";
        myfile << "Nb Thread,"<< M_nbTh<<"\n";
        myfile << "Nb Block used,"<<M_numBlocksGPU<<"\n";
        myfile << "Nb Th/Block,"<<M_nThPerBckGPU<<"\n";
        myfile << "List Graph Dependencie,";
        for (int i = 0; i < M_ListGraphDependencies.size(); i++) { myfile <<M_ListGraphDependencies[i];}
        myfile <<"\n";
        myfile.close();
   }
   if (M_qViewInfo) { std::cout<<"<=====================================================================>"<<"\n"; }
}

void Task::setDeviceHIP(int v) 
{
    #ifdef UseHIP
    int numDevices=0; hipGetDeviceCount(&numDevices);
    if ((v>=0) && (v<=numDevices)) { hipSetDevice(v); }
    #endif
}

void Task::setDeviceCUDA(int v) 
{
    #ifdef UseCUDA
    int numDevices=0; cudaGetDeviceCount(&numDevices);
    if ((v>=0) && (v<=numDevices)) { cudaSetDevice(v); }
    #endif
}

void Task::open(int nbBlock,int NbTh)
{
    M_q_graph=false;
    M_nbTh= NbTh;
    M_numBlocksGPU=nbBlock;
    M_nThPerBckGPU=M_nbTh/M_numBlocksGPU;
    if (M_qViewInfo)
    {
        std::cout<<"<=====================================================================>"<<"\n";
        std::cout<<"[INFO]: Open Graph"<<"\n";
        std::cout<<"[INFO]: nb Thread        : "<<M_nbTh<<"\n";
        std::cout<<"[INFO]: nb Block         : "<<M_numBlocksGPU<<"\n";
        std::cout<<"[INFO]: Thread Per Block : "<<M_nThPerBckGPU<<"\n";
        std::cout<<"<=====================================================================>"<<"\n";
    }
    
    //#ifdef COMPILE_WITH_HIP && UseHIP
    #ifdef UseHIP       
        hipGraphCreate(&hip_graph, 0);
        hip_nodeParams = {0};
        memset(&hip_nodeParams, 0, sizeof(hip_nodeParams));
    #endif

    #ifdef UseCUDA       
        cudaGraphCreate(&cuda_graph, 0);
        cuda_nodeParams = {0};
        memset(&cuda_nodeParams, 0, sizeof(cuda_nodeParams));
    #endif
    
}




template<typename Kernel, typename Input>
    void Task::single_hip(const Kernel& kernel_function,
                                int numElems,
                                int iBegin,int iEnd,
                                Input* buffer)
{   
    #ifdef UseHIP  
        int block_size = 512; block_size =M_numBlocksGPU;   
        typename std::decay<decltype(buffer)> *deviceBuffer;
        int sz=sizeof(typename std::decay<decltype(buffer)>) * numElems;
        hipMalloc((void **) &deviceBuffer,sz);
        hipMemcpy(deviceBuffer,buffer,sz, hipMemcpyHostToDevice);
        if (iEnd>numElems) { iEnd=numElems; }
        if (iBegin<0)      { iBegin=0; }
        int num_blocks = (numElems + block_size - 1) / block_size;
        dim3 thread_block(block_size, 1, 1);
        dim3 grid(num_blocks, 1);
        hipLaunchKernelGGL(OP_IN_KERNEL_GRAPH_LAMBDA_GPU_1D,grid,thread_block,0,0,kernel_function,buffer,iBegin,iEnd);
        hipMemcpy(buffer,deviceBuffer,sz, hipMemcpyDeviceToHost);
        hipFree(deviceBuffer);
    #endif
}

template<typename Kernel, typename Input>
    void Task::single_cuda(const Kernel& kernel_function,
                                int numElems,
                                int iBegin,int iEnd,
                                Input* buffer)
{   
    #ifdef UseCUDA
        int block_size = 512; block_size =M_numBlocksGPU;   
        typename std::decay<decltype(buffer)> *deviceBuffer;
        int sz=sizeof(typename std::decay<decltype(buffer)>) * numElems;
        cudaMalloc((void **) &deviceBuffer,sz);
        cudaMemcpy(deviceBuffer,buffer,sz, hipMemcpyHostToDevice);
        if (iEnd>numElems) { iEnd=numElems; }
        if (iBegin<0)      { iBegin=0; }
        int num_blocks = (numElems + block_size - 1) / block_size;
        dim3 thread_block(block_size, 1, 1);
        dim3 grid(num_blocks, 1);
        OP_IN_KERNEL_GRAPH_LAMBDA_GPU_1D<<<gridthread_block>>>(kernel_function,buffer,iBegin,iEnd);
        cudaMemcpy(buffer,deviceBuffer,sz, hipMemcpyDeviceToHost);
        cudaFree(deviceBuffer);
    #endif
}



template<typename Kernel, typename Input, typename Output>
    void Task::add(const Kernel& kernel_function,
                                int numElems,
                                int iBegin,int iEnd,
                                Input* buffer,
                                Output* hostbuffer,
                                std::vector<int> links)
{            
    #ifdef UseHIP  
        bool qFlag=false;
        //BEGIN::Init new node
        hipGraphNode_t newKernelNode; M_hipGraphNode_t.push_back(newKernelNode);
        memset(&hip_nodeParams, 0, sizeof(hip_nodeParams));

        if (M_qViewInfo) { std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<"\n"; }
        //if (M_qViewInfo) { std::cout<<"[INFO]: M_hipGraphNode_t="<<M_hipGraphNode_t.size()<<" : "<<M_hipGraphNode_t[M_hipGraphNode_t.size()-1]<<"\n"; }
        if (M_qViewInfo) { std::cout<<"[INFO]: Num Graph Node = "<<M_hipGraphNode_t.size()<<"\n"; }

        //CRTL range
        if (iEnd>numElems) { iEnd=numElems; }
        if (iBegin<0)      { iBegin=0; }

        //printf("[%x]\n",M_hipGraphNode_t[M_hipGraphNode_t.size()-1]);

        hip_nodeParams.func   =  (void *)OP_IN_KERNEL_GRAPH_LAMBDA_GPU_1D<Kernel,Input>;
        hip_nodeParams.gridDim        = dim3(M_numBlocksGPU, 1, 1);
        hip_nodeParams.blockDim       = dim3(M_nThPerBckGPU, 1, 1);
        hip_nodeParams.sharedMemBytes = 0;
        void *inputs[4];
        inputs[0]                     = (void *)&kernel_function;
        inputs[1]                     = (void *)&buffer;
        inputs[2]                     = (void *)&iBegin;
        inputs[3]                     = (void *)&iEnd;
        hip_nodeParams.kernelParams   = inputs;
        
        if (M_q_graph) { hip_nodeParams.extra          = NULL; }
        else           { hip_nodeParams.extra          = nullptr; }
        //END::Init new node

        //BEGIN::Dependencies part
        unsigned int nbElemLinks               = links.size();
        unsigned int nbElemKernelNode          = M_hipGraphNode_t.size();
        std::vector<hipGraphNode_t> dependencies;
        M_ListGraphDependencies.push_back(M_hipGraphNode_t.size()-1);
        for (int i = 0; i < nbElemLinks; i++) { 
            if (links[i]==-1) { qFlag=true; }
            if (links[i]!=-1) {
                dependencies.push_back(M_hipGraphNode_t[links[i]]); 
                M_ListGraphDependencies.push_back(links[i]);
            }
        }

        if (M_qViewInfo) {
            std::cout<<"[INFO]: Nb Elem Links  = "<<nbElemLinks<<"\n";
            std::cout<<"[INFO]: Link dependencies with >>> [";
                for (auto v: dependencies) { std::cout << v << " "; } std::cout<<"] <<<\n";
        }
        //END::Dependencies part

        //BEGIN::Add Node to kernel GPU
        if (M_q_graph) { hipGraphAddKernelNode(&M_hipGraphNode_t[M_hipGraphNode_t.size()-1],hip_graph,dependencies.data(),nbElemLinks, &hip_nodeParams); }
        else           { hipGraphAddKernelNode(&M_hipGraphNode_t[M_hipGraphNode_t.size()-1],hip_graph,nullptr,0, &hip_nodeParams); }
        //END::Add Node to kernel GPU

        M_q_graph=true;

        //BEGIN::Final node kernel GPU
        if (qFlag)
        {
            if (M_qViewInfo) {
                std::cout<<"[INFO]:"<<"\n";
                std::cout<<"[INFO]: List >>> [";
                for (auto v: M_hipGraphNode_t) { std::cout << v << " "; } std::cout<<"] <<<\n";
            }
            hipGraphNode_t copyBuffer;
            if (M_qViewInfo) { std::cout<<"[INFO]: Last M_hipGraphNode_t="<<M_hipGraphNode_t.size()<<" : "<<M_hipGraphNode_t[M_hipGraphNode_t.size()-1]<<"\n"; }
            std::vector<hipGraphNode_t> finalDependencies = { M_hipGraphNode_t[ M_hipGraphNode_t.size()-1] };
            hipGraphAddMemcpyNode1D(&copyBuffer,
                                    hip_graph,
                                    dependencies.data(),
                                    dependencies.size(),
                                    hostbuffer,
                                    buffer,
                                    numElems  * sizeof(typename std::decay<decltype(hostbuffer)>::type),
                                    hipMemcpyDeviceToHost);
        }
        //END::Final node kernel GPU

        dependencies.clear();
     #endif
}


template<typename Kernel, typename Input, typename Output>
    void Task::add_cuda(const Kernel& kernel_function,
                                int numElems,
                                int iBegin,int iEnd,
                                Input* buffer,
                                Output* hostbuffer,
                                std::vector<int> links)
{            
    #ifdef UseCUDA  
        bool qFlag=false;
        //BEGIN::Init new node
        cudaGraphNode_t newKernelNode; M_cudaGraphNode_t.push_back(newKernelNode);
        memset(&cuda_nodeParams, 0, sizeof(cuda_nodeParams));

        if (M_qViewInfo) { std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<"\n"; }
        //if (M_qViewInfo) { std::cout<<"[INFO]: M_cudaGraphNode_t="<<M_cudaGraphNode_t.size()<<" : "<<M_cudaGraphNode_t[M_cudaGraphNode_t.size()-1]<<"\n"; }
        if (M_qViewInfo) { std::cout<<"[INFO]: Num Graph Node = "<<M_cudaGraphNode_t.size()<<"\n"; }

        //CRTL range
        if (iEnd>numElems) { iEnd=numElems; }
        if (iBegin<0)      { iBegin=0; }

        //printf("[%x]\n",M_cudaGraphNode_t[M_cudaGraphNode_t.size()-1]);

        cuda_nodeParams.func   =  (void *)OP_IN_KERNEL_GRAPH_LAMBDA_GPU_1D<Kernel,Input>;
        cuda_nodeParams.gridDim        = dim3(M_numBlocksGPU, 1, 1);
        cuda_nodeParams.blockDim       = dim3(M_nThPerBckGPU, 1, 1);
        cuda_nodeParams.sharedMemBytes = 0;
        void *inputs[4];
        inputs[0]                     = (void *)&kernel_function;
        inputs[1]                     = (void *)&buffer;
        inputs[2]                     = (void *)&iBegin;
        inputs[3]                     = (void *)&iEnd;
        cuda_nodeParams.kernelParams   = inputs;
        
        if (M_q_graph) { cuda_nodeParams.extra          = NULL; }
        else           { cuda_nodeParams.extra          = nullptr; }
        //END::Init new node

        //BEGIN::Dependencies part
        unsigned int nbElemLinks               = links.size();
        unsigned int nbElemKernelNode          = M_cudaGraphNode_t.size();
        std::vector<cudaGraphNode_t> dependencies;
        M_ListGraphDependencies.push_back(M_cudaGraphNode_t.size()-1);
        for (int i = 0; i < nbElemLinks; i++) { 
            if (links[i]==-1) { qFlag=true; }
            if (links[i]!=-1) {
                dependencies.push_back(M_cudaGraphNode_t[links[i]]); 
                M_ListGraphDependencies.push_back(links[i]);
            }
        }

        if (M_qViewInfo) {
            std::cout<<"[INFO]: Nb Elem Links  = "<<nbElemLinks<<"\n";
            std::cout<<"[INFO]: Link dependencies with >>> [";
                for (auto v: dependencies) { std::cout << v << " "; } std::cout<<"] <<<\n";
        }
        //END::Dependencies part

        //BEGIN::Add Node to kernel GPU
        if (M_q_graph) { cudaGraphAddKernelNode(&M_cudaGraphNode_t[M_cudaGraphNode_t.size()-1],cuda_graph,dependencies.data(),nbElemLinks, &cuda_nodeParams); }
        else           { cudaGraphAddKernelNode(&M_cudaGraphNode_t[M_cudaGraphNode_t.size()-1],cuda_graph,nullptr,0, &cuda_nodeParams); }
        //END::Add Node to kernel GPU

        M_q_graph=true;

        //BEGIN::Final node kernel GPU
        if (qFlag)
        {
            if (M_qViewInfo) {
                std::cout<<"[INFO]:"<<"\n";
                std::cout<<"[INFO]: List >>> [";
                for (auto v: M_cudaGraphNode_t) { std::cout << v << " "; } std::cout<<"] <<<\n";
            }
            cudaGraphNode_t copyBuffer;
            if (M_qViewInfo) { std::cout<<"[INFO]: Last M_cudaGraphNode_t="<<M_cudaGraphNode_t.size()<<" : "<<M_cudaGraphNode_t[M_cudaGraphNode_t.size()-1]<<"\n"; }
            std::vector<cudaGraphNode_t> finalDependencies = { M_cudaGraphNode_t[ M_cudaGraphNode_t.size()-1] };
            cudaGraphAddMemcpyNode1D(&copyBuffer,
                                    cuda_graph,
                                    dependencies.data(),
                                    dependencies.size(),
                                    hostbuffer,
                                    buffer,
                                    numElems  * sizeof(typename std::decay<decltype(hostbuffer)>::type),
                                    cudaMemcpyDeviceToHost);
        }
        //END::Final node kernel GPU

        dependencies.clear();
     #endif
}



void Task::run()
{
    if (M_qViewInfo) { std::cout<<"<=====================================================================>"<<"\n"; }
    if (M_qViewInfo) { std::cout<<"[INFO]: Run Graph"<<"\n"; }    
    M_t_begin = std::chrono::steady_clock::now();
    //Run HIP Graph
    if (M_q_graph) {   
        //#ifdef COMPILE_WITH_HIP && UseHIP
        #ifdef UseHIP
            hipGraphInstantiate      (&hip_graphExec, hip_graph, nullptr, nullptr, 0);
            hipStreamCreateWithFlags (&hip_graphStream, hipStreamNonBlocking);
            hipGraphLaunch           (hip_graphExec, hip_graphStream);
            hipStreamSynchronize     (hip_graphStream);
        #endif
        #ifdef UseCUDA
            cudaGraphInstantiate      (&cuda_graphExec, cuda_graph, nullptr, nullptr, 0);
            cudaStreamCreateWithFlags (&cuda_graphStream, cudaStreamNonBlocking);
            cudaGraphLaunch           (cuda_graphExec, cuda_graphStream);
            cudaStreamSynchronize     (cuda_graphStream);
        #endif
    }
    M_t_end = std::chrono::steady_clock::now();
    M_time_laps= std::chrono::duration_cast<std::chrono::microseconds>(M_t_end - M_t_begin).count();
    if (M_qViewInfo) { std::cout<<"<=====================================================================>"<<"\n"; }
}

void Task::close()
{
    if (M_q_graph) {   //HIP Graph
        //#ifdef COMPILE_WITH_HIP && UseHIP
        #ifdef UseHIP
            hipGraphExecDestroy     (hip_graphExec);
            hipGraphDestroy         (hip_graph);
            hipStreamDestroy        (hip_graphStream);
            if (M_qDeviceReset)     { hipDeviceReset(); }   // Not be used if Spex Hip AMD activated
        #endif
        #ifdef UseCUDA
            cudaGraphExecDestroy     (cuda_graphExec);
            cudaGraphDestroy         (cuda_graph);
            cudaStreamDestroy        (cuda_graphStream);
            if (M_qDeviceReset)      { cudaDeviceReset(); }  // Not be used if Spex Cuda NVidia activated
        #endif
        if (M_qViewInfo) { std::cout<<"[INFO]: Close Graph Hip"<<"\n"; }
    }
}



template<typename T>
class PtrTask:public Task
{
    private:
        #ifdef UseHIP
        bufferGraphGPU<T> BUFFER;   
        #endif 
        unsigned int bufferSizeBytes;  
    public:
        PtrTask();
        ~PtrTask();

        void set(int nb,T *v);
        void get(T *v);

        template<typename Kernel>
            void add(const Kernel& kernel_function,int iBegin,int iEnd,std::vector<int> links);

        template <typename ... Ts>
            void addTest( Ts && ... ts); 
};


template<typename T>
PtrTask<T>::PtrTask()
{
    //...
}

template<typename T>
PtrTask<T>::~PtrTask()
{
    //...
}

template<typename T>
void PtrTask<T>::set(int nb,T *v)
{
    #ifdef UseHIP
    BUFFER.memoryInit(nb);  bufferSizeBytes=nb*sizeof(T);
    std::memcpy(&BUFFER.data, &v,bufferSizeBytes); 
    BUFFER.memmovHostToDevice();
    #endif
}

template<typename T>
void PtrTask<T>::get(T *v)
{
    #ifdef UseHIP
    BUFFER.memmovDeviceToHost();
    std::memcpy(&v,&BUFFER.data,bufferSizeBytes); 
    #endif
}

template<typename T>
template<typename Kernel>
void PtrTask<T>::add(const Kernel& kernel_function,int iBegin,int iEnd,std::vector<int> links)
{
    #ifdef UseHIP
    Task::add(kernel_function,BUFFER.size,iBegin,iEnd,BUFFER.deviceBuffer,BUFFER.data,links);
    #endif
}


template<typename T>
template <typename ... Ts>
void PtrTask<T>::addTest( Ts && ... ts)
{
    //(_kernel=(op1),_range=(0,nbElements),_links=(0)); 

    auto args = NA::make_arguments( std::forward<Ts>(ts)... );
    auto && kernel = args.get(_kernel);
    auto && range  = args.get(_range);
    auto && links  = args.get(_links);
    auto tp1=std::make_tuple( kernel );
    auto tp2=std::make_tuple( range );
    auto tp3=std::make_tuple(links);
    auto tp=std::tuple_cat(tp1,tp2,tp3);

    //CTRL
    int Nbtp1=std::tuple_size<decltype(tp1)>::value;
    int Nbtp2=std::tuple_size<decltype(tp2)>::value;
    int Nbtp3=std::tuple_size<decltype(tp3)>::value;
    int Nbtp=std::tuple_size<decltype(tp)>::value;
    std::cout <<"==> Size Parameters tp1..3"<<Nbtp1<<" "<<Nbtp2<<" "<<Nbtp3<< "tp="<<Nbtp<<std::endl;

  

    #ifdef UseHIP
        //std::apply(add,tp);
        //std::apply([&](auto &&... args) { add(args...); },tp);
    #endif
    
}


//--------------------------------------------------------------------------------------------------------------------------------
} //End namespace



namespace LEMGPUI {

struct Task {
    enum class state_t      { capture, update };
    void add_kernel_node    (size_t key, hipKernelNodeParams params, hipStream_t s);
    void update_kernel_node (size_t key, hipKernelNodeParams params);
    state_t state()         { return _state; }
    ~Task();

private:
    std::unordered_map<size_t, hipGraphNode_t> _node_map;
    state_t _state;
    hipGraph_t _graph;
    hipGraphExec_t _graph_exec;
    bool _instantiated = false;
    static void begin_capture(hipStream_t s);
    void end_capture  (hipStream_t s);
    void launch_graph (hipStream_t s);

public:
    bool _always_recapture = false;
    template<class Obj>
    void wrap(Obj &o, hipStream_t s);
};


Task::~Task() {
    if (_instantiated) {
        hipGraphDestroy(_graph);
        hipGraphExecDestroy(_graph_exec);
        _instantiated = false;
    }
}

void Task::begin_capture(hipStream_t s) { hipStreamBeginCapture(s, hipStreamCaptureModeGlobal); }

void Task::end_capture(hipStream_t s) {
    if (_instantiated) { hipGraphDestroy(_graph); }
    hipStreamEndCapture(s, &_graph);
    bool need_instantiation;

    if (_instantiated) {
        hipGraphExecUpdateResult updateResult;
        hipGraphNode_t errorNode;
        hipGraphExecUpdate(_graph_exec, _graph, &errorNode, &updateResult);
        if (_graph_exec == nullptr || updateResult != hipGraphExecUpdateSuccess) {
            hipGetLastError();
            if (_graph_exec != nullptr) { hipGraphExecDestroy(_graph_exec); }
            need_instantiation = true;
        } else {
            need_instantiation = false;
        }
    } else {
        need_instantiation = true;
    }

    if (need_instantiation) {
        hipGraphInstantiate(&_graph_exec, _graph, nullptr, nullptr, 0);
    }
    _instantiated = true;
}

template<class Obj>
void Task::wrap(Obj &o, hipStream_t s) 
{
    if (!_always_recapture && _instantiated) {
        _state = state_t::update;
        o(*this, s);
    } else {
        _state = state_t::capture;
        begin_capture(s);
        o(*this, s);
        end_capture(s);
    }
    launch_graph(s);
}

void Task::launch_graph(hipStream_t s) {
    if (_instantiated) { hipGraphLaunch(_graph_exec, s);}
}

void Task::add_kernel_node(size_t key, hipKernelNodeParams params, hipStream_t stream)
{
    hipStreamCaptureStatus capture_status;
    hipGraph_t graph;
    const hipGraphNode_t *deps;
    size_t dep_count;
    hipStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &graph, &deps, &dep_count);
    hipGraphNode_t new_node;
    hipGraphAddKernelNode(&new_node, graph, deps, dep_count, &params);
    _node_map[key] = new_node;
    hipStreamUpdateCaptureDependencies(stream, &new_node, 1, 1);
}

void Task::update_kernel_node(size_t key, hipKernelNodeParams params)
{
    hipGraphExecKernelNodeSetParams(_graph_exec, _node_map[key], &params);
}

} //End namespace LEMGPUI




//================================================================================================================================
// THE END.
//================================================================================================================================
