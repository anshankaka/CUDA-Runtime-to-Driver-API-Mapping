#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <stdio.h>
#include <stdlib.h>

#ifndef TRACE_LOG_FILE
#define TRACE_LOG_FILE "callback_trace.log"
#endif

static FILE *gTraceFile = NULL;

static FILE *TraceOutput()
{
    return gTraceFile ? gTraceFile : stdout;
}

static const char *CallbackSiteName(CUpti_ApiCallbackSite site)
{
    switch (site)
    {
        case CUPTI_API_ENTER:
            return "ENTER";
        case CUPTI_API_EXIT:
            return "EXIT";
        default:
            return "UNKNOWN";
    }
}

static const char *DomainName(CUpti_CallbackDomain domain)
{
    switch (domain)
    {
        case CUPTI_CB_DOMAIN_DRIVER_API:
            return "DRIVER_API";
        case CUPTI_CB_DOMAIN_RUNTIME_API:
            return "RUNTIME_API";
        default:
            return "OTHER";
    }
}

static void CheckCupti(CUptiResult result, const char *func, const char *file, int line)
{
    if (result != CUPTI_SUCCESS)
    {
        const char *errstr = NULL;
        cuptiGetResultString(result, &errstr);
        fprintf(stderr, "CUPTI error at %s:%d: %s failed with %s\n",
                file, line, func, errstr ? errstr : "unknown error");
        exit(EXIT_FAILURE);
    }
}

#define CUPTI_CALL(call) CheckCupti((call), #call, __FILE__, __LINE__)

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
    CUptiResult timestampResult = cuptiGetTimestamp(&timestamp);
    if (timestampResult != CUPTI_SUCCESS)
    {
        return;
    }

    if (info->callbackSite == CUPTI_API_ENTER)
    {
        if (info->correlationData != NULL)
        {
            *info->correlationData = timestamp;
        }

        FILE *out = TraceOutput();
        fprintf(out, "[CUPTI] domain=%s callbacksite=%s function=%s start_time=%llu end_time=0 correlation_id=%u\n",
                DomainName(domain),
                CallbackSiteName(info->callbackSite),
                info->functionName ? info->functionName : "(unknown)",
                (unsigned long long)timestamp,
                info->correlationId);
        fflush(out);
    }
    else if (info->callbackSite == CUPTI_API_EXIT)
    {
        uint64_t startTimestamp = 0;
        if (info->correlationData != NULL)
        {
            startTimestamp = *info->correlationData;
        }

        FILE *out = TraceOutput();
        fprintf(out, "[CUPTI] domain=%s callbacksite=%s function=%s start_time=%llu end_time=%llu correlation_id=%u\n",
                DomainName(domain),
                CallbackSiteName(info->callbackSite),
                info->functionName ? info->functionName : "(unknown)",
                (unsigned long long)startTimestamp,
                (unsigned long long)timestamp,
                info->correlationId);
        fflush(out);
    }
}

__global__ void VectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    gTraceFile = fopen(TRACE_LOG_FILE, "w");
    if (gTraceFile == NULL)
    {
        fprintf(stderr, "Failed to open %s for writing.\n", TRACE_LOG_FILE);
        return EXIT_FAILURE;
    }

    CUpti_SubscriberHandle subscriber = NULL;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)TraceCallback, NULL));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    fprintf(gTraceFile, "domain, callbacksite, function name, start time, end time, correlation ID\n");
    fflush(gTraceFile);

    int vectorLen = 1024 * 1024;
    size_t size = vectorLen * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < vectorLen; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (vectorLen + threadsPerBlock - 1) / threadsPerBlock;

    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, vectorLen);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    CUPTI_CALL(cuptiUnsubscribe(subscriber));

    fprintf(gTraceFile, "VectorAdd finished successfully!\n");
    fclose(gTraceFile);
    gTraceFile = NULL;

    return 0;
}
