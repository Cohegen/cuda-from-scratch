#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 4   // Size of square matrix

// Kernel
__global__ void matmulKernel(float *A, float *B, float *C, int Width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width)
    {
        float Pvalue = 0.0f;

        for (int k = 0; k < Width; ++k)
        {
            Pvalue += A[row * Width + k] * B[k * Width + col];
        }

        C[row * Width + col] = Pvalue;
    }
}

int main()
{
    int size = WIDTH * WIDTH * sizeof(float);

    // Host matrices
    float h_A[WIDTH * WIDTH];
    float h_B[WIDTH * WIDTH];
    float h_C[WIDTH * WIDTH];

    // Initialize matrices
    for (int i = 0; i < WIDTH * WIDTH; i++)
    {
        h_A[i] = 1.0f;  // simple values
        h_B[i] = 2.0f;
    }

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + 15) / 16, (WIDTH + 15) / 16);

    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);

    // Wait for GPU
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            std::cout << h_C[i * WIDTH + j] << " ";
        }
        std::cout << "\n";
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
