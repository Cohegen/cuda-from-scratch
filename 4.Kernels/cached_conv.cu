/*
An optimized 2D convolution kernel that uses 
constant memory, caches and same dimension input and output tile
*/

#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define FILTER_RADIUS 1   // define this (example)
#define FILTER_WIDTH (2*FILTER_RADIUS+1)

// constant memory must use __constant__
__constant__ float F[FILTER_WIDTH][FILTER_WIDTH];

__global__ void optimized_2d_conv(float* N, float* P, int width, int height)
{
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // shared memory must use __shared__
    __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

    // loading input tile
    if (row < height && col < width)
    {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;  // fixed typo
    }

    __syncthreads();  // must use __syncthreads()

    // computing output
    if (col < width && row < height)
    {
        float Pvalue = 0.0f;

        for (int fRow = 0; fRow < FILTER_WIDTH; fRow++)
        {
            for (int fCol = 0; fCol < FILTER_WIDTH; fCol++)
            {
                int localRow = threadIdx.y - FILTER_RADIUS + fRow;
                int localCol = threadIdx.x - FILTER_RADIUS + fCol;

                // inside shared memory tile
                if (localRow >= 0 && localRow < TILE_WIDTH &&
                    localCol >= 0 && localCol < TILE_WIDTH)
                {
                    Pvalue += F[fRow][fCol] *
                              N_s[localRow][localCol];
                }
                else
                {
                    int globalRow = row - FILTER_RADIUS + fRow;
                    int globalCol = col - FILTER_RADIUS + fCol;

                    if (globalRow >= 0 && globalRow < height &&
                        globalCol >= 0 && globalCol < width)
                    {
                        Pvalue += F[fRow][fCol] *
                                  N[globalRow * width + globalCol];
                    }
                }
            }
        }

        P[row * width + col] = Pvalue; // moved outside loop
    }
}