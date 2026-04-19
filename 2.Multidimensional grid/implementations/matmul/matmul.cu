/*
This program performs matrix multiplication to
two non-square matrcies
A has dimensions (MxN)
B has dimensions (NxK)
C = A*B
C has dimensions (MxK)
*/

#include <iostream>
#include <cuda_runtime.h>

/*
A: M x N
B: N x K
C: M x K
*/

__global__ void matmulKernel(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K)
    {
        float Pvalue = 0.0f;

        for (int k = 0; k < N; ++k)
        {
            Pvalue += A[row * N + k] * B[k * K + col];
        }

        C[row * K + col] = Pvalue;
    }
}

int main()
{
    // Example dimensions
    int M = 3;   // rows of A and C
    int N = 4;   // cols of A, rows of B
    int K = 2;   // cols of B and C

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Host memory
    float h_A[M * N];
    float h_B[N * K];
    float h_C[M * K];

    // Initialize A and B
    for (int i = 0; i < M * N; i++) h_A[i] = 1.0f;   // simple test
    for (int i = 0; i < N * K; i++) h_B[i] = 2.0f;

    // Device pointers
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Block + grid setup
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch kernel
    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix C (" << M << "x" << K << "):\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            std::cout << h_C[i * K + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
