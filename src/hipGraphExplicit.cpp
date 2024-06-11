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

#include "na.hpp"
#include "Tools.hpp"
#include "Taskflow_HPC.hpp"

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



/*********************************************************************************************************************************************************/
// BEGIN::ADD NODE DEMO

hipGraph_t                          hip_graph;
hipGraphExec_t                      hip_graphExec;
hipStream_t                         hip_graphStream;
hipKernelNodeParams                 hip_nodeParams;
std::vector<hipGraphNode_t>         M_hipGraphNode_t;  
bool                                M_qhip_graph;
int  M_numBlocksGPU;
int  M_nThPerBckGPU;
int  M_numOpGPU;



void write_vector(std::string ch,double* v,int nb)
{
  std::cout<<"[INFO] :"<<ch<<"> ";
	for (int i = 0; i < nb; i++) { std::cout<<int(v[i]); }
  std::cout<<std::endl;
}



    template<typename Kernel,typename Input>
    __global__ void OP_IN_KERNEL_LAMBDA_GPU_1D(Kernel op,Input *A, int nb) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        if (idx < nb)
            op(idx,A);
        //__syncthreads();
    }


template<typename Kernel, typename Input, typename Output>
void add_hip_graph_1D(const Kernel& kernel_function,
                            dim3 blocks,
                            int numElems, 
                            Input* buffer,
                            Output* hostbuffer,
                            std::vector<int> links,
                            bool qFlag)
{

    bool qViewInfo=true;
    //qViewInfo=false;
        
    //BEGIN::Init new node
    hipGraphNode_t newKernelNode; M_hipGraphNode_t.push_back(newKernelNode);
    memset(&hip_nodeParams, 0, sizeof(hip_nodeParams));

    if (qViewInfo) { std::cout<<"[INFO]: M_hipGraphNode_t["<<M_hipGraphNode_t.size()<<"] : "<<M_hipGraphNode_t[M_hipGraphNode_t.size()-1]<<"\n"; }

    hip_nodeParams.func   =  (void *)OP_IN_KERNEL_LAMBDA_GPU_1D<Kernel,Input>;
    hip_nodeParams.gridDim        = dim3(M_numBlocksGPU, 1, 1);
    hip_nodeParams.blockDim       = dim3(M_nThPerBckGPU, 1, 1);
    hip_nodeParams.sharedMemBytes = 0;
    void *inputs[3];               
    inputs[0]                     = (void *)&kernel_function;
    inputs[1]                     = (void *)&buffer;
    inputs[2]                     = (void *)&numElems;
    hip_nodeParams.kernelParams   = inputs;
    
    if (M_qhip_graph) { hip_nodeParams.extra          = NULL; }
    else              { hip_nodeParams.extra          = nullptr; }
    //END::Init new node

    //BEGIN::Dependencies part
    unsigned int nbElemLinks               = links.size();
    unsigned int nbElemKernelNode          = M_hipGraphNode_t.size();
    std::vector<hipGraphNode_t> dependencies;
    for (int i = 0; i < nbElemLinks; i++) { dependencies.push_back(M_hipGraphNode_t[links[i]]); }
    if (qViewInfo) { 
        std::cout<<"[INFO]: nb Elem Links="<<nbElemLinks<<"\n"; 
        std::cout<<"[INFO]: link dependencies: "; 
            for (auto v: dependencies) { std::cout << v << " "; } std::cout<<"]\n";
    }
    //END::Dependencies part

    //BEGIN::Add Node to kernel GPU
    if (M_qhip_graph) { hipGraphAddKernelNode(&M_hipGraphNode_t[M_hipGraphNode_t.size()-1],hip_graph,dependencies.data(),nbElemLinks, &hip_nodeParams); }
    else { hipGraphAddKernelNode(&M_hipGraphNode_t[M_hipGraphNode_t.size()-1],hip_graph,nullptr,0, &hip_nodeParams); }
    //END::Add Node to kernel GPU

    M_qhip_graph=true;

    if (qFlag)        
    {
        if (qViewInfo) { 
            std::cout<<"[INFO]: list hipGraphNode_t: "; 
            for (auto v: M_hipGraphNode_t) { std::cout << v << " "; } std::cout<<"]\n";
        }
        hipGraphNode_t copyBuffer;   
        if (qViewInfo) { std::cout<<"[INFO]: Last M_hipGraphNode_t="<<M_hipGraphNode_t.size()<<" : "<<M_hipGraphNode_t[M_hipGraphNode_t.size()-1]<<"\n"; }
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
}

// END::ADD NODE
/*********************************************************************************************************************************************************/




void Test001()
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


void Test002()
{
    M_qhip_graph=false;
    std::cout << "\n";
    M_numBlocksGPU = 1;
    M_numOpGPU     = 10;    
    M_nThPerBckGPU =M_numOpGPU/M_numBlocksGPU;

    int nbElements        = M_numOpGPU;
    int bufferSizeBytes   = sizeof(double) * nbElements;
    double *hostBuffer    = (double *)malloc(bufferSizeBytes);
    double *deviceBuffer;

    for (int i = 0; i < nbElements; i++) { hostBuffer[i] = 1.0; }
    write_data<double>(hostBuffer,nbElements);

    hipMalloc((void **) &deviceBuffer, bufferSizeBytes);
    hipMemcpy(deviceBuffer,hostBuffer,bufferSizeBytes,hipMemcpyHostToDevice);

    auto op1=[ ] __device__ (auto i,auto a) { 
		    a[i]=a[i]+10;
		    return (true); 
	};

    auto op2=[ ] __device__ (auto i,auto a) { 
		    a[i]=a[i]*2;
		    return (true); 
	};

    auto op3=[ ] __device__ (auto i,auto a) { 
		    a[i]=a[i]-1;
		    return (true); 
	};

    auto op4=[ ] __device__ (auto i,auto a) { 
		    a[i]=a[i]+4;
		    return (true); 
	};


    std::cout<<"[INFO]: Init Graph"<<"\n";

    hipGraphCreate(&hip_graph, 0);
    hip_nodeParams = {0};
    memset(&hip_nodeParams, 0, sizeof(hip_nodeParams));

    std::cout<<"[INFO]: Add node in Graph"<<"\n";

    std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<"\n";
    std::vector<int> link0{ 0 };
    add_hip_graph_1D(op1,dim3(nbElements,1,1),nbElements,deviceBuffer,hostBuffer,link0,false);
    std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<"\n";
    std::vector<int> link1{ 0 };
    add_hip_graph_1D(op2,dim3(nbElements,1,1),nbElements,deviceBuffer,hostBuffer,link1,false);
    std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<"\n";
    std::vector<int> link2{ 1 };
    add_hip_graph_1D(op3,dim3(nbElements,1,1),nbElements,deviceBuffer,hostBuffer,link1,true);
    std::cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<"\n";


    std::cout<<"[INFO]: Run Graph"<<"\n";
    hipGraphInstantiate      (&hip_graphExec, hip_graph, nullptr, nullptr, 0);
    hipStreamCreateWithFlags (&hip_graphStream, hipStreamNonBlocking);
    hipGraphLaunch           (hip_graphExec, hip_graphStream);
    hipStreamSynchronize     (hip_graphStream);
    std::cout<<"\n";

    hipMemcpy(hostBuffer,deviceBuffer,bufferSizeBytes,hipMemcpyDeviceToHost);
    write_data<double>(hostBuffer,nbElements);

    hipGraphExecDestroy     (hip_graphExec);
    hipGraphDestroy         (hip_graph);
    hipStreamDestroy        (hip_graphStream);
    hipDeviceReset          ();

    std::cout<<"\n";
    std::cout << "[INFO]: Close Graph"<<"\n";
    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";

}

int main() 
{
    //Test001(); //OK it works
    Test002(); //OK it works
}


