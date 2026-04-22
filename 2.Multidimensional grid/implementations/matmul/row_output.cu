/*
This program intends to implement a GPU kernel in 
which one thread produces one row of the output matrix
*/
#include <iostream>
#include <cuda_runtime.h>

#define N 4   // Square matrix size

/*
Each thread computes one full row of C
*/
__global__ void rowThreadOutKernel(float *A, float *B, float *C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N)
    {
        for (int col = 0; col < N; col++)
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

    // Host matrices
    float h_A[N * N];
    float h_B[N * N];
    float h_C[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device pointers
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel (1D configuration)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    rowThreadOutKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
