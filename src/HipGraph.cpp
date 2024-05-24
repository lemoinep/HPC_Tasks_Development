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
#include "hip/hip_runtime.h"
//#include "roctx.h"
//#include "roctracer_ext.h"
#include "hipblas.h"
#include "hiprand.h"

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

//Links Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

//#include "na.hpp"
//#include "Tools.hpp"
//#include "Taskflow_HPC.hpp"

#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20


/*********************************************************************************************************************************************************/
// BEGIN::HIP AMD GPU

    template <typename PtrType>
    __global__ void op_kernel_000_001(PtrType *buffer, unsigned int numElems) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < numElems)
            buffer[tid] = buffer[tid]+1;
    }

    //BEGIN::SUB Node Level 1
    template <typename PtrType>
    __global__ void op_kernel_001_001(PtrType *buffer, unsigned int numElems) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < numElems/3) buffer[tid] += 2;
    }

    template <typename PtrType>
    __global__ void op_kernel_001_002(PtrType *buffer, unsigned int numElems) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if ((tid>=numElems/3) && (tid<(2*numElems/3))) buffer[tid] *= 4;
    }

    template <typename PtrType>
    __global__ void op_kernel_001_003(PtrType *buffer, unsigned int numElems) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid>=(2*numElems/3)) buffer[tid] *= 5;
    }
    //END::SUB Node Level 1

    //BEGIN::SUB Node Level 2
    template <typename PtrType>
        __global__ void op_kernel_002_001(PtrType *buffer, unsigned int numElems) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid < numElems) buffer[tid] += 1;
    }
    //END::SUB Node Level 2

    //BEGIN::SUB Node Level 3
    template <typename PtrType>
        __global__ void op_kernel_003_001(PtrType *buffer, unsigned int numElems) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid < numElems) buffer[tid] -= 3;
    }
    //END::SUB Node Level 3

// END::HIP AMD GPU
/*********************************************************************************************************************************************************/


/*********************************************************************************************************************************************************/
// BEGIN::HIP TOOLS

template <typename PtrType> 
void write_data(void *data,const int nb) 
{
    PtrType *buffer = (PtrType *)data;
    std::cout << "[INFO]: V=[";
    for (unsigned int i = 0; i < nb; ++i) {
        std::cout << buffer[i]; if (i<nb-1) { std::cout << ","; }
    }
    std::cout << "]\n";
}

void runGraphGPU(hipGraph_t &Graph, hipGraphExec_t &graphExec,hipStream_t &graphStream)
 {
    hipGraphInstantiate      (&graphExec, Graph, nullptr, nullptr, 0);
    hipStreamCreateWithFlags (&graphStream, hipStreamNonBlocking);
    hipGraphLaunch           (graphExec, graphStream);
    hipStreamSynchronize     (graphStream);
}


void destroyGraphGPU(hipGraph_t &Graph, hipGraphExec_t &graphExec,hipStream_t &graphStream)
{
    hipGraphExecDestroy     (graphExec);
    hipGraphDestroy         (Graph);
    hipStreamDestroy        (graphStream);
    hipDeviceReset          ();
}


template <typename PtrType>
void createGraphGPU(hipGraph_t &Graph, hipGraphExec_t &graphExec,
                    hipStream_t &graphStream, PtrType *buffer,
                    unsigned int numElems, PtrType *hostBuffer,
                    int numThreads,
                    int numBlocks
                    )

{

    //             Node000-A
    //          /     ¡       \
    //   Node001-A Node001-B Node001-C
    //          \      ¡      /
    //             Node002-A
    //                ¡
    //             Node003-A

    int nThPerBck=numThreads/numBlocks;

    hipGraphCreate(&Graph, 0);

    hipGraphNode_t KernelNode_000_001;
    hipKernelNodeParams nodeParams = {0};
    memset(&nodeParams, 0, sizeof(nodeParams));

    void *inputs[2]; //two input for each GPU functions
    inputs[0] = (void *)&buffer;
    inputs[1] = (void *)&numElems;  

    nodeParams.func           = (void *)op_kernel_000_001<PtrType>;
    nodeParams.gridDim        = dim3(numBlocks, 1, 1);
    nodeParams.blockDim       = dim3(nThPerBck, 1, 1);
    nodeParams.sharedMemBytes = 0;
    nodeParams.kernelParams   = inputs;
    nodeParams.extra          = nullptr;

    hipGraphAddKernelNode(&KernelNode_000_001, Graph, nullptr, 0, &nodeParams);

        hipGraphNode_t KernelNode_001_001;
        memset(&nodeParams, 0, sizeof(nodeParams));
        nodeParams.func           = (void *)op_kernel_001_001<PtrType>;
        nodeParams.gridDim        = dim3(numBlocks, 1, 1);
        nodeParams.blockDim       = dim3(nThPerBck, 1, 1);
        nodeParams.sharedMemBytes = 0;
        inputs[0]                 = (void *)&buffer;
        inputs[1]                 = (void *)&numElems;
        nodeParams.kernelParams   = inputs;
        nodeParams.extra          = NULL;
        hipGraphAddKernelNode(&KernelNode_001_001, Graph, &KernelNode_000_001, 1, &nodeParams);

        hipGraphNode_t KernelNode_001_002;
        memset(&nodeParams, 0, sizeof(nodeParams));
        nodeParams.func           = (void *)op_kernel_001_002<PtrType>;
        nodeParams.gridDim        = dim3(numBlocks, 1, 1);
        nodeParams.blockDim       = dim3(nThPerBck, 1, 1);
        nodeParams.sharedMemBytes = 0;
        inputs[0]                 = (void *)&buffer;
        inputs[1]                 = (void *)&numElems;
        nodeParams.kernelParams   = inputs;
        nodeParams.extra          = NULL;
        hipGraphAddKernelNode(&KernelNode_001_002, Graph, &KernelNode_000_001, 1, &nodeParams);


        hipGraphNode_t KernelNode_001_003;
        memset(&nodeParams, 0, sizeof(nodeParams));
        nodeParams.func           = (void *)op_kernel_001_003<PtrType>;
        nodeParams.gridDim        = dim3(numBlocks, 1, 1);
        nodeParams.blockDim       = dim3(nThPerBck, 1, 1);
        nodeParams.sharedMemBytes = 0;
        inputs[0]                 = (void *)&buffer;
        inputs[1]                 = (void *)&numElems;
        nodeParams.kernelParams   = inputs;
        nodeParams.extra          = NULL;
        hipGraphAddKernelNode(&KernelNode_001_003, Graph, &KernelNode_000_001, 1, &nodeParams);


        hipGraphNode_t KernelNode_002_001;
        memset(&nodeParams, 0, sizeof(nodeParams));
        nodeParams.func           = (void *)op_kernel_002_001<PtrType>;
        nodeParams.gridDim        = dim3(numBlocks, 1, 1);
        nodeParams.blockDim       = dim3(nThPerBck, 1, 1);
        nodeParams.sharedMemBytes = 0;
        inputs[0]                 = (void *)&buffer;
        inputs[1]                 = (void *)&numElems;
        nodeParams.kernelParams   = inputs;
        nodeParams.extra          = NULL;
        hipGraphNode_t dependencies2[3];
        dependencies2[0] = KernelNode_001_001;
        dependencies2[1] = KernelNode_001_002;
        dependencies2[2] = KernelNode_001_003;
        //hipGraphAddMemFreeNode(&KernelNode_002_001,Graph,dependencies2,3,&nodeParams); <== detach
        hipGraphAddKernelNode(&KernelNode_002_001, Graph,dependencies2,3, &nodeParams);
         
        hipGraphNode_t KernelNode_003_001;
        memset(&nodeParams, 0, sizeof(nodeParams));
        nodeParams.func           = (void *)op_kernel_003_001<PtrType>;
        nodeParams.gridDim        = dim3(numBlocks, 1, 1);
        nodeParams.blockDim       = dim3(nThPerBck, 1, 1);
        nodeParams.sharedMemBytes = 0;
        inputs[0]                 = (void *)&buffer;
        inputs[1]                 = (void *)&numElems;
        nodeParams.kernelParams   = inputs;
        nodeParams.extra          = NULL;
        hipGraphNode_t dependencies3[1];
        dependencies3[0] = KernelNode_002_001;
        hipGraphAddKernelNode(&KernelNode_003_001, Graph,dependencies3,1, &nodeParams);

        hipGraphNode_t copyBuffer;   
        std::vector<hipGraphNode_t> dependencies = { KernelNode_003_001 };
        hipGraphAddMemcpyNode1D(&copyBuffer, Graph,dependencies.data(),dependencies.size(),hostBuffer, buffer, numElems*sizeof(PtrType), hipMemcpyDeviceToHost);
}

// END::HIP TOOLS
/*********************************************************************************************************************************************************/





int main() 
{
    std::cout << "\n";

    hipGraph_t     graph;
    hipGraphExec_t graphExec;
    hipStream_t    graphStream;

    unsigned int numBlocks       = 10;
    unsigned int numThreads      = 70;
    
    unsigned int numElems        = numThreads;
    unsigned int bufferSizeBytes = numElems * sizeof(int);
    unsigned int hostBuffer[numElems];
    memset(hostBuffer, 0, bufferSizeBytes);

    std::cout << "[INFO]: Init buffer\n";
    for(int i = 0; i < numElems; i++) { hostBuffer[i] = 1; }
    unsigned int *deviceBuffer;
    std::cout << "[INFO]: Write CPU buffer\n";
    write_data<int>(hostBuffer,numThreads);
    std::cout << "[INFO]: Copy buffer CPU to GPU\n";
    hipMalloc(&deviceBuffer, bufferSizeBytes);
    hipMemcpy(deviceBuffer,hostBuffer,bufferSizeBytes, hipMemcpyHostToDevice);
    std::cout << "[INFO]: ...\n";
    std::cout << "[INFO]: Build Graph GPU\n";
    createGraphGPU(graph, graphExec, graphStream, deviceBuffer,numElems, hostBuffer,numThreads,numBlocks);
    std::cout << "[INFO]: ...\n";
    std::cout << "[INFO]: Execute Graph GPU\n";
    std::cout << "[INFO]: ...\n";
    runGraphGPU(graph, graphExec, graphStream);
    std::cout << "[INFO]: Copy buffer GPU to CPU\n";
    hipMemcpy(hostBuffer,deviceBuffer,bufferSizeBytes,hipMemcpyDeviceToHost); hipFree(deviceBuffer);
    std::cout << "[INFO]: Write CPU buffer\n";
    write_data<int>(hostBuffer,numThreads);
    std::cout << "[INFO]: ...\n";
    std::cout << "[INFO]: Destroy Graph GPU\n";
    destroyGraphGPU(graph, graphExec, graphStream);
    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";
}


