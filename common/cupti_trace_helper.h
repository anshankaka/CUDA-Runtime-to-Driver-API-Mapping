#pragma once

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

static cudaError_t TryCuda(cudaError_t result, const char *call, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA warning at %s:%d: %s returned %s\n",
                file, line, call, cudaGetErrorString(result));
        cudaGetLastError();
    }
    return result;
}

#define CUPTI_CALL(call) CheckCupti((call), #call, __FILE__, __LINE__)
#define CUDA_CALL(call) CheckCuda((call), #call, __FILE__, __LINE__)
#define CUDA_TRY(call) TryCuda((call), #call, __FILE__, __LINE__)

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

static CUpti_SubscriberHandle StartCuptiTrace()
{
    gTraceFile = fopen(TRACE_LOG_FILE, "w");
    if (gTraceFile == NULL)
    {
        fprintf(stderr, "Failed to open %s\n", TRACE_LOG_FILE);
        exit(EXIT_FAILURE);
    }

    CUpti_SubscriberHandle subscriber = NULL;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)TraceCallback, NULL));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    fprintf(gTraceFile, "domain, callbacksite, function name, start time, end time, correlation ID\n");
    fflush(gTraceFile);
    return subscriber;
}

static void StopCuptiTrace(CUpti_SubscriberHandle subscriber, const char *message)
{
    CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    CUPTI_CALL(cuptiUnsubscribe(subscriber));

    if (gTraceFile != NULL)
    {
        fprintf(gTraceFile, "%s\n", message ? message : "finished");
        fclose(gTraceFile);
        gTraceFile = NULL;
    }
}
