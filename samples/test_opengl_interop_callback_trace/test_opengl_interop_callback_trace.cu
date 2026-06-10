#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

#include "..\..\common\cupti_trace_helper.h"

#include <cuda_gl_interop.h>
#include <stdio.h>
#include <string.h>

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif

typedef void(APIENTRY *GlGenBuffersFn)(GLsizei n, GLuint *buffers);
typedef void(APIENTRY *GlBindBufferFn)(GLenum target, GLuint buffer);
typedef void(APIENTRY *GlBufferDataFn)(GLenum target, ptrdiff_t size, const void *data, GLenum usage);
typedef void(APIENTRY *GlDeleteBuffersFn)(GLsizei n, const GLuint *buffers);

struct GlBufferFns
{
    GlGenBuffersFn GenBuffers;
    GlBindBufferFn BindBuffer;
    GlBufferDataFn BufferData;
    GlDeleteBuffersFn DeleteBuffers;
};

struct GlContext
{
    HINSTANCE instance;
    HWND window;
    HDC dc;
    HGLRC rc;
    const char *className;
};

static LRESULT CALLBACK HiddenWindowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
{
    return DefWindowProcA(window, message, wparam, lparam);
}

static void CheckWin(BOOL result, const char *call, const char *file, int line)
{
    if (!result)
    {
        fprintf(stderr, "Win32 error at %s:%d: %s failed with %lu\n",
                file, line, call, GetLastError());
        exit(EXIT_FAILURE);
    }
}

static void *LoadGlProc(const char *name)
{
    void *proc = (void *)wglGetProcAddress(name);
    if (proc == NULL || proc == (void *)1 || proc == (void *)2 || proc == (void *)3 || proc == (void *)-1)
    {
        HMODULE module = GetModuleHandleA("opengl32.dll");
        proc = module ? (void *)GetProcAddress(module, name) : NULL;
    }

    if (proc == NULL)
    {
        fprintf(stderr, "Failed to load OpenGL function %s\n", name);
        exit(EXIT_FAILURE);
    }

    return proc;
}

static void CheckGl(const char *operation, const char *file, int line)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        fprintf(stderr, "OpenGL error at %s:%d after %s: 0x%04x\n",
                file, line, operation, (unsigned int)error);
        exit(EXIT_FAILURE);
    }
}

#define WIN_CALL(call) CheckWin((call), #call, __FILE__, __LINE__)
#define GL_CALL(operation)          \
    do                              \
    {                               \
        operation;                  \
        CheckGl(#operation, __FILE__, __LINE__); \
    } while (0)

static GlContext CreateHiddenGlContext()
{
    GlContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.instance = GetModuleHandleA(NULL);
    ctx.className = "CuptiHiddenOpenGLWindow";

    WNDCLASSA wc;
    memset(&wc, 0, sizeof(wc));
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = HiddenWindowProc;
    wc.hInstance = ctx.instance;
    wc.lpszClassName = ctx.className;

    if (!RegisterClassA(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
    {
        fprintf(stderr, "RegisterClassA failed with %lu\n", GetLastError());
        exit(EXIT_FAILURE);
    }

    ctx.window = CreateWindowA(
        ctx.className,
        "CUPTI OpenGL Interop Hidden Window",
        WS_OVERLAPPEDWINDOW,
        0,
        0,
        1,
        1,
        NULL,
        NULL,
        ctx.instance,
        NULL);
    if (ctx.window == NULL)
    {
        fprintf(stderr, "CreateWindowA failed with %lu\n", GetLastError());
        exit(EXIT_FAILURE);
    }

    ctx.dc = GetDC(ctx.window);
    if (ctx.dc == NULL)
    {
        fprintf(stderr, "GetDC failed with %lu\n", GetLastError());
        exit(EXIT_FAILURE);
    }

    PIXELFORMATDESCRIPTOR pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pixelFormat = ChoosePixelFormat(ctx.dc, &pfd);
    if (pixelFormat == 0)
    {
        fprintf(stderr, "ChoosePixelFormat failed with %lu\n", GetLastError());
        exit(EXIT_FAILURE);
    }

    WIN_CALL(SetPixelFormat(ctx.dc, pixelFormat, &pfd));

    ctx.rc = wglCreateContext(ctx.dc);
    if (ctx.rc == NULL)
    {
        fprintf(stderr, "wglCreateContext failed with %lu\n", GetLastError());
        exit(EXIT_FAILURE);
    }

    WIN_CALL(wglMakeCurrent(ctx.dc, ctx.rc));
    return ctx;
}

static GlBufferFns LoadGlBufferFns()
{
    GlBufferFns fns;
    fns.GenBuffers = (GlGenBuffersFn)LoadGlProc("glGenBuffers");
    fns.BindBuffer = (GlBindBufferFn)LoadGlProc("glBindBuffer");
    fns.BufferData = (GlBufferDataFn)LoadGlProc("glBufferData");
    fns.DeleteBuffers = (GlDeleteBuffersFn)LoadGlProc("glDeleteBuffers");
    return fns;
}

static void DestroyGlContext(const GlContext *ctx)
{
    if (ctx->rc != NULL)
    {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(ctx->rc);
    }
    if (ctx->window != NULL && ctx->dc != NULL)
    {
        ReleaseDC(ctx->window, ctx->dc);
    }
    if (ctx->window != NULL)
    {
        DestroyWindow(ctx->window);
    }
    if (ctx->className != NULL)
    {
        UnregisterClassA(ctx->className, ctx->instance);
    }
}

int main()
{
    CUpti_SubscriberHandle subscriber = StartCuptiTrace();

    GlContext gl = CreateHiddenGlContext();
    GlBufferFns glBuffers = LoadGlBufferFns();

    unsigned int glCudaCount = 0;
    int glCudaDevices[8] = {0};
    cudaError_t glDeviceResult = CUDA_TRY(cudaGLGetDevices(&glCudaCount, glCudaDevices, 8, cudaGLDeviceListAll));
    if (glDeviceResult == cudaSuccess && glCudaCount > 0)
    {
        CUDA_CALL(cudaSetDevice(glCudaDevices[0]));
    }
    else
    {
        CUDA_CALL(cudaSetDevice(0));
    }

    int wglDevice = -1;
    CUDA_TRY(cudaWGLGetDevice(&wglDevice, NULL));

    const size_t bufferBytes = 4096;
    unsigned char bufferData[bufferBytes];
    memset(bufferData, 7, sizeof(bufferData));

    GLuint vbo = 0;
    glBuffers.GenBuffers(1, &vbo);
    GL_CALL(glBuffers.BindBuffer(GL_ARRAY_BUFFER, vbo));
    GL_CALL(glBuffers.BufferData(GL_ARRAY_BUFFER, (ptrdiff_t)bufferBytes, bufferData, GL_DYNAMIC_DRAW));
    GL_CALL(glBuffers.BindBuffer(GL_ARRAY_BUFFER, 0));

    const int width = 16;
    const int height = 16;
    unsigned char pixels[width * height * 4];
    memset(pixels, 128, sizeof(pixels));

    GLuint texture = 0;
    GL_CALL(glGenTextures(1, &texture));
    GL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels));
    GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));

    cudaGraphicsResource_t bufferResource = NULL;
    cudaGraphicsResource_t textureResource = NULL;
    cudaGraphicsResource_t resources[2] = {NULL, NULL};
    int resourceCount = 0;

    if (CUDA_TRY(cudaGraphicsGLRegisterBuffer(&bufferResource, vbo, cudaGraphicsRegisterFlagsNone)) == cudaSuccess)
    {
        CUDA_TRY(cudaGraphicsResourceSetMapFlags(bufferResource, cudaGraphicsMapFlagsWriteDiscard));
        resources[resourceCount++] = bufferResource;
    }

    if (CUDA_TRY(cudaGraphicsGLRegisterImage(&textureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)) == cudaSuccess)
    {
        resources[resourceCount++] = textureResource;
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
                cudaMipmappedArray_t mappedMipmappedArray = NULL;
                CUDA_TRY(cudaGraphicsSubResourceGetMappedArray(&mappedArray, textureResource, 0, 0));
                CUDA_TRY(cudaGraphicsResourceGetMappedMipmappedArray(&mappedMipmappedArray, textureResource));
            }

            CUDA_TRY(cudaGraphicsUnmapResources(resourceCount, resources, 0));
        }
    }

    if (textureResource != NULL)
    {
        CUDA_TRY(cudaGraphicsUnregisterResource(textureResource));
    }
    if (bufferResource != NULL)
    {
        CUDA_TRY(cudaGraphicsUnregisterResource(bufferResource));
    }

    GL_CALL(glDeleteTextures(1, &texture));
    glBuffers.DeleteBuffers(1, &vbo);
    DestroyGlContext(&gl);

    StopCuptiTrace(subscriber, "opengl interop coverage finished");
    return 0;
}
