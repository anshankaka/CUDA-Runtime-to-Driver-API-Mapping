#include "..\..\common\cupti_trace_helper.h"

#include <string.h>

__device__ float gGraphSymbol[128];

__global__ void GraphExtraKernel(float *data, int n, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] = value;
    }
}

static void CUDART_CB UserObjectDestructor(void *ptr)
{
    int *value = (int *)ptr;
    *value += 1;
}

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    CUDA_CALL(cudaSetDevice(0));
    const int n = 128;
    const size_t bytes = n * sizeof(float);
    float host[n];
    for (int i = 0; i < n; ++i)
    {
        host[i] = (float)i;
    }

    float *dev = NULL;
    CUDA_CALL(cudaMalloc((void **)&dev, bytes));

    cudaGraph_t graph = NULL;
    cudaGraph_t child = NULL;
    cudaGraph_t childOut = NULL;
    cudaGraph_t clone = NULL;
    cudaGraphExec_t exec = NULL;
    cudaGraphExec_t execFlags = NULL;
    cudaGraphExec_t execParams = NULL;
    cudaGraphNode_t emptyA = NULL;
    cudaGraphNode_t emptyB = NULL;
    cudaGraphNode_t tempNode = NULL;
    cudaGraphNode_t childEmpty = NULL;
    cudaGraphNode_t childNode = NULL;
    cudaGraphNode_t copyNode = NULL;
    cudaGraphNode_t toSymbolNode = NULL;
    cudaGraphNode_t fromSymbolNode = NULL;
    cudaGraphNode_t kernelNode = NULL;

    CUDA_CALL(cudaGraphCreate(&graph, 0));
    CUDA_CALL(cudaGraphAddEmptyNode(&emptyA, graph, NULL, 0));
    CUDA_CALL(cudaGraphAddEmptyNode(&emptyB, graph, NULL, 0));
    CUDA_CALL(cudaGraphAddEmptyNode(&tempNode, graph, NULL, 0));
    CUDA_CALL(cudaGraphDestroyNode(tempNode));

    CUDA_CALL(cudaGraphCreate(&child, 0));
    CUDA_CALL(cudaGraphAddEmptyNode(&childEmpty, child, NULL, 0));
    CUDA_CALL(cudaGraphAddChildGraphNode(&childNode, graph, &emptyA, 1, child));
    CUDA_CALL(cudaGraphChildGraphNodeGetGraph(childNode, &childOut));

    CUDA_CALL(cudaGraphAddMemcpyNode1D(&copyNode, graph, &childNode, 1, dev, host, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaGraphAddMemcpyNodeToSymbol(&toSymbolNode, graph, &copyNode, 1, gGraphSymbol, host, bytes, 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaGraphAddMemcpyNodeFromSymbol(&fromSymbolNode, graph, &toSymbolNode, 1, dev, gGraphSymbol, bytes, 0, cudaMemcpyDeviceToDevice));

    void *kernelArgs[] = {&dev, (void *)&n, (void *)&host[0]};
    cudaKernelNodeParams kernelParams;
    memset(&kernelParams, 0, sizeof(kernelParams));
    kernelParams.func = (void *)GraphExtraKernel;
    kernelParams.gridDim = dim3(1);
    kernelParams.blockDim = dim3(128);
    kernelParams.kernelParams = kernelArgs;
    CUDA_CALL(cudaGraphAddKernelNode(&kernelNode, graph, &fromSymbolNode, 1, &kernelParams));

    cudaGraphNode_t fromNodes[] = {emptyB};
    cudaGraphNode_t toNodes[] = {kernelNode};
    CUDA_CALL(cudaGraphAddDependencies(graph, fromNodes, toNodes, 1));
    size_t depCount = 0;
    CUDA_CALL(cudaGraphNodeGetDependencies(kernelNode, NULL, &depCount));
    CUDA_CALL(cudaGraphNodeGetDependentNodes(emptyB, NULL, &depCount));
    CUDA_CALL(cudaGraphRemoveDependencies(graph, fromNodes, toNodes, 1));

    cudaGraphEdgeData edgeData;
    memset(&edgeData, 0, sizeof(edgeData));
    CUDA_TRY(cudaGraphAddDependencies_v2(graph, fromNodes, toNodes, &edgeData, 1));
    CUDA_TRY(cudaGraphNodeGetDependencies_v2(kernelNode, NULL, NULL, &depCount));
    CUDA_TRY(cudaGraphNodeGetDependentNodes_v2(emptyB, NULL, NULL, &depCount));
    CUDA_TRY(cudaGraphRemoveDependencies_v2(graph, fromNodes, toNodes, &edgeData, 1));

    cudaMemcpy3DParms memcpyParams;
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    CUDA_CALL(cudaGraphMemcpyNodeGetParams(copyNode, &memcpyParams));
    CUDA_CALL(cudaGraphMemcpyNodeSetParams1D(copyNode, dev, host, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaGraphMemcpyNodeSetParamsToSymbol(toSymbolNode, gGraphSymbol, host, bytes, 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaGraphMemcpyNodeSetParamsFromSymbol(fromSymbolNode, dev, gGraphSymbol, bytes, 0, cudaMemcpyDeviceToDevice));

    cudaKernelNodeAttrValue attrValue;
    memset(&attrValue, 0, sizeof(attrValue));
    attrValue.sharedMemCarveout = 50;
    CUDA_TRY(cudaGraphKernelNodeSetAttribute(kernelNode, cudaKernelNodeAttributePreferredSharedMemoryCarveout, &attrValue));
    CUDA_TRY(cudaGraphKernelNodeGetAttribute(kernelNode, cudaKernelNodeAttributePreferredSharedMemoryCarveout, &attrValue));
    unsigned int enabled = 0;

    cudaGraphInstantiateParams instantiateParams;
    memset(&instantiateParams, 0, sizeof(instantiateParams));
    CUDA_CALL(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CALL(cudaGraphInstantiateWithFlags(&execFlags, graph, 0));
    CUDA_CALL(cudaGraphInstantiateWithParams(&execParams, graph, &instantiateParams));
    CUDA_TRY(cudaGraphNodeSetEnabled(exec, kernelNode, 1));
    CUDA_TRY(cudaGraphNodeGetEnabled(exec, kernelNode, &enabled));
    CUDA_TRY(cudaGraphExecMemcpyNodeSetParams1D(exec, copyNode, dev, host, bytes, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaGraphExecMemcpyNodeSetParamsToSymbol(exec, toSymbolNode, gGraphSymbol, host, bytes, 0, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaGraphExecMemcpyNodeSetParamsFromSymbol(exec, fromSymbolNode, dev, gGraphSymbol, bytes, 0, cudaMemcpyDeviceToDevice));

    int userValue = 1;
    cudaUserObject_t userObject = NULL;
    CUDA_CALL(cudaUserObjectCreate(&userObject, &userValue, UserObjectDestructor, 1, cudaUserObjectNoDestructorSync));
    CUDA_CALL(cudaUserObjectRetain(userObject, 1));
    CUDA_CALL(cudaGraphRetainUserObject(graph, userObject, 1, cudaGraphUserObjectMove));
    CUDA_CALL(cudaGraphReleaseUserObject(graph, userObject, 1));
    CUDA_CALL(cudaUserObjectRelease(userObject, 1));

    CUDA_CALL(cudaGraphClone(&clone, graph));
    CUDA_TRY(cudaGraphDebugDotPrint(graph, "graph_extra.dot", cudaGraphDebugDotFlagsVerbose));

    CUDA_CALL(cudaGraphExecDestroy(execParams));
    CUDA_CALL(cudaGraphExecDestroy(execFlags));
    CUDA_CALL(cudaGraphExecDestroy(exec));
    CUDA_CALL(cudaGraphDestroy(clone));
    CUDA_CALL(cudaGraphDestroy(child));
    CUDA_CALL(cudaGraphDestroy(graph));
    CUDA_CALL(cudaFree(dev));

    StopCuptiTrace(subscriber, "graph extra coverage finished successfully");
    return 0;
}
