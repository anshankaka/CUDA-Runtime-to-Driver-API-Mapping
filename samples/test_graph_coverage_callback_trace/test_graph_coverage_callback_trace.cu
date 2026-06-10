#include "..\..\common\cupti_trace_helper.h"

#include <string.h>

__global__ void AddOneKernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] += 1.0f;
    }
}

static void CUDART_CB GraphHostNode(void *userData)
{
    int *flag = (int *)userData;
    *flag += 1;
}

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    CUDA_CALL(cudaSetDevice(0));
    const int n = 1 << 16;
    const size_t bytes = n * sizeof(float);
    float *host = (float *)malloc(bytes);
    float *dev = NULL;
    int hostFlag = 0;
    for (int i = 0; i < n; ++i)
    {
        host[i] = (float)i;
    }

    cudaStream_t stream = NULL;
    cudaEvent_t event = NULL;
    cudaGraph_t graph = NULL;
    cudaGraph_t clonedGraph = NULL;
    cudaGraphExec_t graphExec = NULL;
    cudaGraphNode_t emptyNode = NULL;
    cudaGraphNode_t memcpyNode = NULL;
    cudaGraphNode_t memsetNode = NULL;
    cudaGraphNode_t kernelNode = NULL;
    cudaGraphNode_t hostNode = NULL;
    cudaGraphNode_t eventRecordNode = NULL;
    cudaGraphNode_t eventWaitNode = NULL;

    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaEventCreate(&event));
    CUDA_CALL(cudaMalloc((void **)&dev, bytes));
    CUDA_CALL(cudaGraphCreate(&graph, 0));
    CUDA_CALL(cudaGraphAddEmptyNode(&emptyNode, graph, NULL, 0));
    CUDA_CALL(cudaGraphAddMemcpyNode1D(&memcpyNode, graph, &emptyNode, 1, dev, host, bytes, cudaMemcpyHostToDevice));

    cudaMemsetParams memsetParams = {0};
    memsetParams.dst = dev;
    memsetParams.value = 0;
    memsetParams.elementSize = 4;
    memsetParams.width = n;
    memsetParams.height = 1;
    CUDA_CALL(cudaGraphAddMemsetNode(&memsetNode, graph, &memcpyNode, 1, &memsetParams));

    void *kernelArgs[] = {&dev, (void *)&n};
    cudaKernelNodeParams kernelParams = {0};
    kernelParams.func = (void *)AddOneKernel;
    kernelParams.gridDim = dim3((n + 255) / 256);
    kernelParams.blockDim = dim3(256);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    CUDA_CALL(cudaGraphAddKernelNode(&kernelNode, graph, &memsetNode, 1, &kernelParams));

    cudaHostNodeParams hostParams = {0};
    hostParams.fn = GraphHostNode;
    hostParams.userData = &hostFlag;
    CUDA_CALL(cudaGraphAddHostNode(&hostNode, graph, &kernelNode, 1, &hostParams));
    CUDA_CALL(cudaGraphAddEventRecordNode(&eventRecordNode, graph, &hostNode, 1, event));
    CUDA_CALL(cudaGraphAddEventWaitNode(&eventWaitNode, graph, &eventRecordNode, 1, event));

    size_t nodeCount = 0;
    size_t edgeCount = 0;
    CUDA_CALL(cudaGraphGetNodes(graph, NULL, &nodeCount));
    CUDA_CALL(cudaGraphGetEdges(graph, NULL, NULL, &edgeCount));
    CUDA_CALL(cudaGraphGetRootNodes(graph, NULL, &nodeCount));

    cudaGraphNodeType nodeType;
    CUDA_CALL(cudaGraphNodeGetType(kernelNode, &nodeType));
    CUDA_CALL(cudaGraphKernelNodeGetParams(kernelNode, &kernelParams));
    CUDA_CALL(cudaGraphKernelNodeSetParams(kernelNode, &kernelParams));
    CUDA_CALL(cudaGraphMemsetNodeGetParams(memsetNode, &memsetParams));
    CUDA_CALL(cudaGraphMemsetNodeSetParams(memsetNode, &memsetParams));
    CUDA_CALL(cudaGraphHostNodeGetParams(hostNode, &hostParams));
    CUDA_CALL(cudaGraphHostNodeSetParams(hostNode, &hostParams));

    CUDA_CALL(cudaGraphClone(&clonedGraph, graph));
    cudaGraphNode_t clonedKernelNode = NULL;
    CUDA_CALL(cudaGraphNodeFindInClone(&clonedKernelNode, kernelNode, clonedGraph));

    CUDA_CALL(cudaGraphInstantiate(&graphExec, graph, 0));
    CUDA_CALL(cudaGraphUpload(graphExec, stream));
    CUDA_CALL(cudaGraphLaunch(graphExec, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    cudaGraphExecUpdateResultInfo updateInfo;
    memset(&updateInfo, 0, sizeof(updateInfo));
    CUDA_TRY(cudaGraphExecUpdate(graphExec, graph, &updateInfo));
    CUDA_CALL(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &kernelParams));
    CUDA_CALL(cudaGraphExecMemsetNodeSetParams(graphExec, memsetNode, &memsetParams));
    CUDA_CALL(cudaGraphExecHostNodeSetParams(graphExec, hostNode, &hostParams));
    unsigned long long execFlags = 0;
    CUDA_CALL(cudaGraphExecGetFlags(graphExec, &execFlags));

    CUDA_CALL(cudaGraphExecDestroy(graphExec));
    CUDA_CALL(cudaGraphDestroy(clonedGraph));
    CUDA_CALL(cudaGraphDestroy(graph));
    CUDA_CALL(cudaFree(dev));
    CUDA_CALL(cudaEventDestroy(event));
    CUDA_CALL(cudaStreamDestroy(stream));
    free(host);
    CUDA_CALL(cudaDeviceSynchronize());

    StopCuptiTrace(subscriber, hostFlag > 0 ? "graph coverage finished successfully" : "graph coverage host node not observed");
    return 0;
}
