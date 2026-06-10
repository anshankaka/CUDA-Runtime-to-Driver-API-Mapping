#include "..\..\common\cupti_trace_helper.h"

__global__ void FillKernel(float *data, int n, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] = value + (float)i;
    }
}

static void CUDART_CB HostCallback(void *userData)
{
    int *flag = (int *)userData;
    *flag = 1;
}

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    int leastPriority = 0;
    int greatestPriority = 0;
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    cudaStream_t normal = NULL;
    cudaStream_t priority = NULL;
    cudaEvent_t event = NULL;
    CUDA_CALL(cudaStreamCreate(&normal));
    CUDA_CALL(cudaStreamCreateWithPriority(&priority, cudaStreamNonBlocking, greatestPriority));
    CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    unsigned int flags = 0;
    int priorityValue = 0;
    unsigned long long streamId = 0;
    CUDA_CALL(cudaStreamGetFlags(priority, &flags));
    CUDA_CALL(cudaStreamGetPriority(priority, &priorityValue));
    CUDA_CALL(cudaStreamGetId(priority, &streamId));

    const int n = 1 << 18;
    const size_t bytes = n * sizeof(float);
    float *managed = NULL;
    float *asyncA = NULL;
    float *asyncB = NULL;
    int hostFlag = 0;

    CUDA_CALL(cudaMallocManaged((void **)&managed, bytes));
    CUDA_CALL(cudaStreamAttachMemAsync(normal, managed, 0, cudaMemAttachSingle));

    cudaMemPool_t defaultPool = NULL;
    cudaMemPool_t currentPool = NULL;
    CUDA_CALL(cudaDeviceGetDefaultMemPool(&defaultPool, 0));
    CUDA_CALL(cudaDeviceGetMemPool(&currentPool, 0));

    cudaMemPoolAttr attr = cudaMemPoolAttrReleaseThreshold;
    unsigned long long threshold = bytes * 4;
    CUDA_CALL(cudaMemPoolSetAttribute(defaultPool, attr, &threshold));
    threshold = 0;
    CUDA_CALL(cudaMemPoolGetAttribute(defaultPool, attr, &threshold));

    CUDA_CALL(cudaMallocAsync((void **)&asyncA, bytes, normal));
    CUDA_CALL(cudaMallocFromPoolAsync((void **)&asyncB, bytes, defaultPool, priority));
    CUDA_CALL(cudaMemsetAsync(asyncA, 0, bytes, normal));
    FillKernel<<<(n + 255) / 256, 256, 0, normal>>>(asyncA, n, 1.0f);
    CUDA_CALL(cudaPeekAtLastError());

    CUDA_CALL(cudaEventRecordWithFlags(event, normal, cudaEventRecordDefault));
    CUDA_CALL(cudaStreamWaitEvent(priority, event, 0));
    CUDA_CALL(cudaMemcpyAsync(asyncB, asyncA, bytes, cudaMemcpyDeviceToDevice, priority));
    CUDA_CALL(cudaLaunchHostFunc(priority, HostCallback, &hostFlag));
    CUDA_CALL(cudaStreamQuery(normal) == cudaErrorNotReady ? cudaSuccess : cudaGetLastError());
    CUDA_CALL(cudaStreamSynchronize(priority));

    CUDA_CALL(cudaFreeAsync(asyncA, normal));
    CUDA_CALL(cudaFreeAsync(asyncB, priority));
    CUDA_CALL(cudaStreamSynchronize(normal));
    CUDA_CALL(cudaStreamSynchronize(priority));
    CUDA_CALL(cudaMemPoolTrimTo(defaultPool, 0));

    CUDA_CALL(cudaFree(managed));
    CUDA_CALL(cudaEventDestroy(event));
    CUDA_CALL(cudaStreamDestroy(priority));
    CUDA_CALL(cudaStreamDestroy(normal));
    CUDA_CALL(cudaDeviceSynchronize());

    StopCuptiTrace(subscriber, hostFlag ? "stream pool coverage finished successfully" : "stream pool coverage host callback not observed");
    return 0;
}
