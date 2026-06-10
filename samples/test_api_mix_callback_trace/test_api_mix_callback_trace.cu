#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <stdio.h>
#include <stdlib.h>

#ifndef TRACE_LOG_FILE
#define TRACE_LOG_FILE "callback_trace.log"
#endif

static FILE *gTraceFile = NULL;

static const char *CallbackSiteName(CUpti_ApiCallbackSite site)
{
    return site == CUPTI_API_ENTER ? "ENTER" : (site == CUPTI_API_EXIT ? "EXIT" : "UNKNOWN");
}

static const char *DomainName(CUpti_CallbackDomain domain)
{
    return domain == CUPTI_CB_DOMAIN_DRIVER_API ? "DRIVER_API" :
           domain == CUPTI_CB_DOMAIN_RUNTIME_API ? "RUNTIME_API" : "OTHER";
}

static void CheckCupti(CUptiResult result, const char *call, const char *file, int line)
{
    if (result != CUPTI_SUCCESS)
    {
        const char *errstr = NULL;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "CUPTI error at %s:%d: %s failed with %s\n",
                file, line, call, errstr ? errstr : "unknown");
        exit(EXIT_FAILURE);
    }
}

static void CheckCuda(cudaError_t result, const char *call, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d: %s failed with %s\n",
                file, line, call, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#define CUPTI_CALL(call) CheckCupti((call), #call, __FILE__, __LINE__)
#define CUDA_CALL(call) CheckCuda((call), #call, __FILE__, __LINE__)

static void CUPTIAPI TraceCallback(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void *cbdata)
{
    (void)userdata;
    (void)cbid;

    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API)
    {
        return;
    }

    const CUpti_CallbackData *info = (const CUpti_CallbackData *)cbdata;
    uint64_t timestamp = 0;
    if (cuptiGetTimestamp(&timestamp) != CUPTI_SUCCESS)
    {
        return;
    }

    if (info->callbackSite == CUPTI_API_ENTER && info->correlationData != NULL)
    {
        *info->correlationData = timestamp;
    }

    uint64_t start = 0;
    uint64_t end = 0;
    if (info->callbackSite == CUPTI_API_ENTER)
    {
        start = timestamp;
    }
    else if (info->callbackSite == CUPTI_API_EXIT)
    {
        start = info->correlationData ? *info->correlationData : 0;
        end = timestamp;
    }
    else
    {
        return;
    }

    fprintf(gTraceFile,
            "[CUPTI] domain=%s callbacksite=%s function=%s start_time=%llu end_time=%llu correlation_id=%u\n",
            DomainName(domain),
            CallbackSiteName(info->callbackSite),
            info->functionName ? info->functionName : "(unknown)",
            (unsigned long long)start,
            (unsigned long long)end,
            info->correlationId);
    fflush(gTraceFile);
}

__global__ void ScaleAddKernel(const float *a, const float *b, float *c, int n, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] * scale + b[i];
    }
}

int main()
{
    gTraceFile = fopen(TRACE_LOG_FILE, "w");
    if (gTraceFile == NULL)
    {
        fprintf(stderr, "Failed to open %s\n", TRACE_LOG_FILE);
        return EXIT_FAILURE;
    }

    CUpti_SubscriberHandle subscriber = NULL;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)TraceCallback, NULL));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    fprintf(gTraceFile, "domain, callbacksite, function name, start time, end time, correlation ID\n");
    fflush(gTraceFile);

    int deviceCount = 0;
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    CUDA_CALL(cudaFree(0));

    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    float *hA = NULL;
    float *hB = NULL;
    float *hC = NULL;
    float *dA = NULL;
    float *dB = NULL;
    float *dC = NULL;
    cudaStream_t stream = NULL;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;

    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));
    CUDA_CALL(cudaMallocHost((void **)&hA, bytes));
    CUDA_CALL(cudaHostAlloc((void **)&hB, bytes, cudaHostAllocDefault));
    hC = (float *)malloc(bytes);
    if (hC == NULL)
    {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i)
    {
        hA[i] = (float)i;
        hB[i] = (float)(n - i);
    }

    CUDA_CALL(cudaMalloc((void **)&dA, bytes));
    CUDA_CALL(cudaMalloc((void **)&dB, bytes));
    CUDA_CALL(cudaMalloc((void **)&dC, bytes));
    CUDA_CALL(cudaMemsetAsync(dC, 0, bytes, stream));
    CUDA_CALL(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaEventRecord(start, stream));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ScaleAddKernel<<<blocks, threads, 0, stream>>>(dA, dB, dC, n, 0.5f);
    CUDA_CALL(cudaPeekAtLastError());

    CUDA_CALL(cudaEventRecord(stop, stream));
    CUDA_CALL(cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CALL(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dC));
    CUDA_CALL(cudaFreeHost(hA));
    CUDA_CALL(cudaFreeHost(hB));
    free(hC);
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaDeviceReset());

    CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    CUPTI_CALL(cuptiUnsubscribe(subscriber));

    fprintf(gTraceFile, "API mix finished successfully, elapsed_ms=%.3f\n", elapsedMs);
    fclose(gTraceFile);
    gTraceFile = NULL;
    return 0;
}
