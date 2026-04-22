#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Helper macro: check CUDA API calls and abort on error
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - "   \
                 << cudaGetErrorString(err) << endl;                           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel function (runs on GPU)
__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Host function
void vecAdd(float *A, float *B, float *C, int n)
{
    float *A_d, *B_d, *C_d;
    size_t size = n * sizeof(float);

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc((void **)&A_d, size));
    CUDA_CHECK(cudaMalloc((void **)&B_d, size));
    CUDA_CHECK(cudaMalloc((void **)&C_d, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel
    vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main()
{
    int n = 1000;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Call CUDA function
    vecAdd(A, B, C, n);

    // Print first 10 results
    cout << "First 10 results:\n";
    for (int i = 0; i < 10; i++) {
        cout << "C[" << i << "] = " << C[i] << endl;
    }

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
