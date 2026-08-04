#include <cuda_runtime.h>

#define FILTER_RADIUS 1
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)

#define OUT_TILE_DIM 32
#define IN_TILE_DIM (OUT_TILE_DIM + 2 * FILTER_RADIUS)

__constant__ float F[FILTER_WIDTH][FILTER_WIDTH];

__global__ void tiledConv2D(
    const float* N,
    float* P,
    int width,
    int height)
{
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    // Global coordinates of the input tile
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Load input tile + halo into shared memory
    if (row >= 0 && row < height &&
        col >= 0 && col < width)
    {
        N_s[threadIdx.y][threadIdx.x] =
            N[row * width + col];
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Only inner threads compute outputs
    if (threadIdx.x >= FILTER_RADIUS &&
        threadIdx.x < IN_TILE_DIM - FILTER_RADIUS &&
        threadIdx.y >= FILTER_RADIUS &&
        threadIdx.y < IN_TILE_DIM - FILTER_RADIUS)
    {
        int outCol =
            blockIdx.x * OUT_TILE_DIM +
            (threadIdx.x - FILTER_RADIUS);

        int outRow =
            blockIdx.y * OUT_TILE_DIM +
            (threadIdx.y - FILTER_RADIUS);

        if (outRow < height && outCol < width)
        {
            float Pvalue = 0.0f;

            for (int fRow = 0; fRow < FILTER_WIDTH; fRow++)
            {
                for (int fCol = 0; fCol < FILTER_WIDTH; fCol++)
                {
                    Pvalue +=
                        F[fRow][fCol] *
                        N_s[threadIdx.y - FILTER_RADIUS + fRow]
                           [threadIdx.x - FILTER_RADIUS + fCol];
                }
            }

            P[outRow * width + outCol] = Pvalue;
        }
    }
}