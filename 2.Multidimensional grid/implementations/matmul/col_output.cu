/*
This program implements a matrix-matrix multiplication
in which one thread produces one column of the resultant matrix
*/
#include <iostream>
#include <cuda_runtime.h>

#define N 4

__global__ void columnThreadKernel(float *A, float *B, float *C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N)
    {
        for (int row = 0; row < N; row++)
        {
            float Pvalue = 0.0f;

            for (int k = 0; k < N; k++)
            {
                Pvalue += A[row * N + k] * B[k * N + col];
            }

            C[row * N + col] = Pvalue;
        }
    }
}

int main()
{
    size_t size = N * N * sizeof(float);

    float h_A[N * N], h_B[N * N], h_C[N * N];

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    columnThreadKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Column-thread result:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            std::cout << h_C[i * N + j] << " ";
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
