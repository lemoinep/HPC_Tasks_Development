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


#include <execution> //C++20
//#include <coroutine> //C++20
//#include "CoroutineScheduler.hpp" //C++20

#define UseHIP

/*********************************************************************************************************************************************************/



namespace LEMGPUI {

#ifdef UseHIP

struct Task {
    enum class state_t      { capture, update };
    void add_kernel_node    (size_t key, hipKernelNodeParams params, hipStream_t s);
    void update_kernel_node (size_t key, hipKernelNodeParams params);
    state_t state()         { return M_state; }
    ~Task();

private:
    std::unordered_map<size_t, hipGraphNode_t> _node_map;
    state_t        M_state;
    hipGraph_t     M_graph;
    hipGraphExec_t M_graph_exec;
    bool M_qInstantiated = false;
    static void begin_capture  (hipStream_t stream);
    void end_capture           (hipStream_t stream);
    void launch_graph          (hipStream_t stream);

public:
    bool _always_recapture = false;
    template<class Obj>
        void wrap(Obj &o, hipStream_t stream);
};


Task::~Task() {
    if (M_qInstantiated) {
        hipGraphDestroy(M_graph);
        hipGraphExecDestroy(M_graph_exec);
        M_qInstantiated = false;
    }
}

void Task::begin_capture(hipStream_t stream) { hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal); }

void Task::end_capture(hipStream_t stream) {
    if (M_qInstantiated) { hipGraphDestroy(M_graph); }
    hipStreamEndCapture(stream, &M_graph);
    bool need_instantiation;

    if (M_qInstantiated) {
        hipGraphExecUpdateResult updateResult;
        hipGraphNode_t errorNode;
        hipGraphExecUpdate(M_graph_exec, M_graph, &errorNode, &updateResult);
        if (M_graph_exec == nullptr || updateResult != hipGraphExecUpdateSuccess) {
            hipGetLastError();
            if (M_graph_exec != nullptr) { hipGraphExecDestroy(M_graph_exec); }
            need_instantiation = true;
        } else {
            need_instantiation = false;
        }
    } else {
        need_instantiation = true;
    }

    if (need_instantiation) {
        hipGraphInstantiate(&M_graph_exec, M_graph, nullptr, nullptr, 0);
    }
    M_qInstantiated = true;
}

template<class Obj>
void Task::wrap(Obj &o, hipStream_t stream) 
{
    if (!_always_recapture && M_qInstantiated) {
        M_state = state_t::update;
        o(*this, stream);
    } 
    else
    {
        M_state = state_t::capture;
        begin_capture(stream);
        o(*this, stream);
        end_capture(stream);
    }
    launch_graph(stream);
}

void Task::launch_graph(hipStream_t stream) {
    if (M_qInstantiated) { hipGraphLaunch(M_graph_exec, stream);}
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
    hipGraphExecKernelNodeSetParams(M_graph_exec, _node_map[key], &params);
}

#endif

} //End namespace LEMGPUI




/*********************************************************************************************************************************************************/
// BEGIN::HIP AMD GPU

__global__ void shortKernel(float *out_d, const float *in_d, int N, float f){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) { 
      out_d[idx] = f * in_d[idx];
  }
}

__global__ void initKernel(float *ptr, int N, float f){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) { 
      ptr[idx] = f;
  }
}

// END::HIP AMD GPU
/*********************************************************************************************************************************************************/


constexpr int n_kernel = 3;
constexpr int n_iteration = 400000;


void run_kernels_graph(float *out_d, float *in_d, int size, float f, LEMGPUI::Task &g, hipStream_t s)
{
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;

  for(int i = 0; i < n_kernel; i++){
    hipKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(threads), 1, 1};
    params.gridDim = {static_cast<unsigned int>(blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void *>(shortKernel);
    void *args[] = {&out_d, &in_d, &size, &f};
    params.kernelParams = args;
    params.extra = nullptr;

    if (g.state() == LEMGPUI::Task::state_t::capture) {
      // Static kernels
      hipLaunchKernelGGL(shortKernel,blocks, threads, 0, s,out_d, in_d, size, 1.004f);
      hipLaunchKernelGGL(shortKernel,blocks, threads, 0, s,in_d, out_d, size, 1.004f);

      // kernels with dynamic parameter `f`
      // Add the kernel nodes
      g.add_kernel_node(i * 2 + 0, params, s);
      params.kernelParams[0] = &in_d;
      params.kernelParams[1] = &out_d;
      g.add_kernel_node(i * 2 + 1, params, s);
    } else if (g.state() == LEMGPUI::Task::state_t::update) {
      // Update the kernel nodes
      g.update_kernel_node(i * 2 + 0, params);
      params.kernelParams[0] = &in_d;
      params.kernelParams[1] = &out_d;
      g.update_kernel_node(i * 2 + 1, params);
    }
  } 
}

void run_kernels_no_graph(float *out_d, float *in_d, int size, float f, hipStream_t s)
{
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;

  for(int i = 0; i < n_kernel; i++){
    // Static kernels
    hipLaunchKernelGGL(shortKernel,blocks, threads, 0, s,out_d, in_d, size, 1.004f);
    hipLaunchKernelGGL(shortKernel,blocks, threads, 0, s,in_d, out_d, size, 1.004f);

    // kernels with dynamic parameter `f`
    hipLaunchKernelGGL(shortKernel,blocks, threads, 0, s,out_d, in_d, size, f);
    hipLaunchKernelGGL(shortKernel,blocks, threads, 0, s,in_d, out_d, size, f);
  } 
}

void run_init(float *ptr, int size, float f, hipStream_t s) {
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;
  hipLaunchKernelGGL(initKernel,blocks, threads, 0, s,ptr, size, f);
}



void Test001()
{
    printf("[INFO]: Start\n");
    LEMGPUI::Task _graph;
    LEMGPUI::Task _graph_always_recapture;

    _graph_always_recapture._always_recapture = true;

    // Set up memory, stream, events
    float *out_d = nullptr;
    float *in_d  = nullptr;

    int size = 1024*1000;
    size_t bytes = size * sizeof(float);

    hipMalloc(&out_d, bytes);
    hipMalloc(&in_d, bytes);
    hipStream_t stream;
    hipEvent_t start, stop;
    hipStreamCreate(&stream);
    hipEventCreate(&start);
    hipEventCreate(&stop);




//================================================================================================================================

    float scale = 2.0f;

    auto wrap_obj_graph = [&](LEMGPUI::Task &g, hipStream_t s) {
        run_kernels_graph(out_d, in_d, size, scale, g, s);
    };

    auto wrap_obj_no_graph = [&](LEMGPUI::Task &g, hipStream_t s) {
        run_kernels_no_graph(out_d, in_d, size, scale, stream);
    };

//================================================================================================================================


//================================================================================================================================
// BEGIN::Recapture-then-update with hip graph

    printf("[INFO]: Init Stream\n");
    run_init(out_d, size, 1.0f, stream);
    run_init(in_d, size, 1.0f, stream);

    printf("[INFO]: Running with    hip graph ('Recapture-then-update') ...\n");

    // Running the test with graph
    hipEventRecord(start, stream);

    for(int i = 0; i < n_iteration; i++){
        scale = 1.0f + i * 0.001f;
        _graph_always_recapture.wrap(wrap_obj_no_graph, stream);
    }

    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    float milliseconds;
    hipEventElapsedTime(&milliseconds, start, stop);

    printf("[INFO]: Running with    hip graph ('Recapture-then-update') took %6.2f ms\n", milliseconds);
    printf("[INFO]: Effective Bandwidth (GB/s): %f\n", size*n_iteration*4*3/milliseconds/1e6);

//END::Recapture-then-update with hip graph
//================================================================================================================================


//================================================================================================================================
//BEGIN::Combined Approach with hip graph 

    printf("[INFO]: Init Stream\n");
    run_init(out_d, size, 1.0f, stream);
    run_init(in_d, size, 1.0f, stream);

    printf("[INFO]: Running with    hip graph ('Combined Approach')     ...\n");

    // Running the test with graph
    hipEventRecord(start, stream);
    for(int i = 0; i < n_iteration; i++){
        scale = 1.0f + i * 0.001f;
        _graph.wrap(wrap_obj_graph, stream);
    }
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&milliseconds, start, stop);
    printf("[INFO]: Running with    hip graph ('Combined Approach')     took %6.2f ms\n", milliseconds);
    printf("[INFO]: Effective Bandwidth (GB/s): %f\n", size*n_iteration*4*3/milliseconds/1e6);


//END::Combined Approach with hip graph 
//================================================================================================================================


//================================================================================================================================
//BEGIN::without hip graph 

    printf("[INFO]: Init Stream\n");
    run_init(out_d, size, 1.0f, stream);
    run_init(in_d, size, 1.0f, stream);

    printf("[INFO]: Running without hip graph                           ...\n");

    hipEventRecord(start, stream);

    for(int i = 0; i < n_iteration; i++){
        scale = 1.0f + i * 0.001f;
        run_kernels_no_graph(out_d, in_d, size, scale, stream);
    }

    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&milliseconds, start, stop);

    printf("[INFO]: Running without hip graph                           took %6.2f ms\n", milliseconds);
    printf("[INFO]: Effective Bandwidth (GB/s): %f\n", size*n_iteration*4*3/milliseconds/1e6);

//END::without hip graph 
//================================================================================================================================


//================================================================================================================================
    hipStreamDestroy(stream);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(out_d);
    hipFree(in_d);
//================================================================================================================================

}


int main() 
{
    Test001(); //OK it works
    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";
}


