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


#define _param(...) _parameters=Sbtask::parameters(__VA_ARGS__)

//Links external
#include "na.hpp"


//Links internal
#include "hardware.hpp"
#include "taskgpu.hpp"
#include "taskcpu.hpp"

//links internal scheduling management

//================================================================================================================================
// THE END.
//================================================================================================================================

