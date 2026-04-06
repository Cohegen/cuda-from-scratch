#include <iostream>
#include <cuda_runtime.h>

using namespace std;

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
    int size = n * sizeof(float);

    // Allocate memory on device
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // Copy data from host to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch kernel
    vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, n);

    // Copy result back to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    int n = 1000;
    int size = n * sizeof(float);

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
