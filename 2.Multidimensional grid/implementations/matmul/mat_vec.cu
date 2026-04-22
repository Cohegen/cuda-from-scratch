#include <iostream>
#include <cuda_runtime.h>

/*
A: N x N matrix
B: N x 1 vector
C: N x 1 vector
*/

__global__ void matVecKernel(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N)
    {
        float Pvalue = 0.0f;

        for (int k = 0; k < N; k++)
        {
            Pvalue += A[row * N + k] * B[k];
        }

        C[row] = Pvalue;
    }
}

int main()
{
    int N = 4;

    size_t sizeMatrix = N * N * sizeof(float);
    size_t sizeVector = N * sizeof(float);

    // Host memory
    float h_A[N * N];
    float h_B[N];
    float h_C[N];

    // Initialize A and B
    for (int i = 0; i < N * N; i++)
        h_A[i] = 1.0f;   // simple test values

    for (int i = 0; i < N; i++)
        h_B[i] = 2.0f;

    // Device pointers
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeMatrix);
    cudaMalloc(&d_B, sizeVector);
    cudaMalloc(&d_C, sizeVector);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeVector, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matVecKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, sizeVector, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result vector C:\n";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
