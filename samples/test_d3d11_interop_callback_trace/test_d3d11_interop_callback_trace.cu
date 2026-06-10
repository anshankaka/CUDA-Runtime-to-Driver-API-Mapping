#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include "..\..\common\cupti_trace_helper.h"

#include <cuda_d3d11_interop.h>
#include <stdio.h>
#include <string.h>

static void CheckHr(HRESULT result, const char *call, const char *file, int line)
{
    if (FAILED(result))
    {
        fprintf(stderr, "HRESULT error at %s:%d: %s failed with 0x%08lx\n",
                file, line, call, (unsigned long)result);
        exit(EXIT_FAILURE);
    }
}

#define HR_CALL(call) CheckHr((call), #call, __FILE__, __LINE__)

static IDXGIAdapter1 *FindCudaAdapter(int *cudaDevice)
{
    IDXGIFactory1 *factory = NULL;
    HR_CALL(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&factory));

    IDXGIAdapter1 *selected = NULL;
    for (UINT index = 0;; ++index)
    {
        IDXGIAdapter1 *adapter = NULL;
        HRESULT hr = factory->EnumAdapters1(index, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            break;
        }
        HR_CALL(hr);

        int candidateDevice = -1;
        cudaError_t cudaResult = CUDA_TRY(cudaD3D11GetDevice(&candidateDevice, adapter));
        if (cudaResult == cudaSuccess)
        {
            *cudaDevice = candidateDevice;
            selected = adapter;
            break;
        }

        adapter->Release();
    }

    factory->Release();

    if (selected == NULL)
    {
        fprintf(stderr, "No CUDA-compatible DXGI adapter was found for D3D11 interop.\n");
        exit(EXIT_FAILURE);
    }

    return selected;
}

static ID3D11Buffer *CreateInteropBuffer(ID3D11Device *device)
{
    D3D11_BUFFER_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.ByteWidth = 4096;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    ID3D11Buffer *buffer = NULL;
    HR_CALL(device->CreateBuffer(&desc, NULL, &buffer));
    return buffer;
}

static ID3D11Texture2D *CreateInteropTexture(ID3D11Device *device, UINT mipLevels)
{
    D3D11_TEXTURE2D_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.Width = 32;
    desc.Height = 32;
    desc.MipLevels = mipLevels;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    ID3D11Texture2D *texture = NULL;
    HR_CALL(device->CreateTexture2D(&desc, NULL, &texture));
    return texture;
}

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    int cudaDevice = -1;
    IDXGIAdapter1 *adapter = FindCudaAdapter(&cudaDevice);
    CUDA_CALL(cudaSetDevice(cudaDevice));

    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0};
    D3D_FEATURE_LEVEL createdLevel;
    ID3D11Device *device = NULL;
    ID3D11DeviceContext *context = NULL;

    HR_CALL(D3D11CreateDevice(
        adapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        NULL,
        0,
        levels,
        (UINT)(sizeof(levels) / sizeof(levels[0])),
        D3D11_SDK_VERSION,
        &device,
        &createdLevel,
        &context));

    unsigned int d3dCudaCount = 0;
    int d3dCudaDevices[8] = {0};
    CUDA_TRY(cudaD3D11GetDevices(&d3dCudaCount, d3dCudaDevices, 8, device, cudaD3D11DeviceListAll));

    ID3D11Buffer *buffer = CreateInteropBuffer(device);
    ID3D11Texture2D *texture = CreateInteropTexture(device, 1);
    ID3D11Texture2D *mipTexture = CreateInteropTexture(device, 2);

    cudaGraphicsResource_t bufferResource = NULL;
    cudaGraphicsResource_t textureResource = NULL;
    cudaGraphicsResource_t mipTextureResource = NULL;
    cudaGraphicsResource_t resources[3] = {NULL, NULL, NULL};
    int resourceCount = 0;

    if (CUDA_TRY(cudaGraphicsD3D11RegisterResource(&bufferResource, buffer, cudaGraphicsRegisterFlagsNone)) == cudaSuccess)
    {
        CUDA_TRY(cudaGraphicsResourceSetMapFlags(bufferResource, cudaGraphicsMapFlagsWriteDiscard));
        resources[resourceCount++] = bufferResource;
    }

    if (CUDA_TRY(cudaGraphicsD3D11RegisterResource(&textureResource, texture, cudaGraphicsRegisterFlagsSurfaceLoadStore)) == cudaSuccess)
    {
        resources[resourceCount++] = textureResource;
    }

    if (CUDA_TRY(cudaGraphicsD3D11RegisterResource(&mipTextureResource, mipTexture, cudaGraphicsRegisterFlagsSurfaceLoadStore)) == cudaSuccess)
    {
        resources[resourceCount++] = mipTextureResource;
    }

    if (resourceCount > 0)
    {
        cudaError_t mapResult = CUDA_TRY(cudaGraphicsMapResources(resourceCount, resources, 0));
        if (mapResult == cudaSuccess)
        {
            if (bufferResource != NULL)
            {
                void *devicePtr = NULL;
                size_t mappedSize = 0;
                CUDA_TRY(cudaGraphicsResourceGetMappedPointer(&devicePtr, &mappedSize, bufferResource));
            }

            if (textureResource != NULL)
            {
                cudaArray_t mappedArray = NULL;
                CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(&mappedArray, textureResource, 0, 0));
            }

            if (mipTextureResource != NULL)
            {
                cudaMipmappedArray_t mappedMipmappedArray = NULL;
                CUDA_TRY(cudaGraphicsResourceGetMappedMipmappedArray(&mappedMipmappedArray, mipTextureResource));
            }

            CUDA_TRY(cudaGraphicsUnmapResources(resourceCount, resources, 0));
        }
    }

    if (mipTextureResource != NULL)
    {
        CUDA_TRY(cudaGraphicsUnregisterResource(mipTextureResource));
    }
    if (textureResource != NULL)
    {
        CUDA_TRY(cudaGraphicsUnregisterResource(textureResource));
    }
    if (bufferResource != NULL)
    {
        CUDA_TRY(cudaGraphicsUnregisterResource(bufferResource));
    }

    mipTexture->Release();
    texture->Release();
    buffer->Release();
    context->Release();
    device->Release();
    adapter->Release();

    StopCuptiTrace(subscriber, "d3d11 interop coverage finished");
    return 0;
}
