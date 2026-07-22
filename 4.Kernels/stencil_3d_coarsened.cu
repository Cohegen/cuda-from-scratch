/*
A 3D coarsened stencil using thread coarsening along the x-dimension.
*/

#include <cuda_runtime.h>

#define TILE_WIDTH 8
#define COARSE_FACTOR 2

__global__ void stencil_3d_coarsened(const float* input, float* output, int width, int height, int depth)
{
    // Global coordinates
    int x = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int slice = width * height;

    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int currentX = x + c * TILE_WIDTH;

        if (currentX > 0 && currentX < width - 1 &&
            y > 0 && y < height - 1 &&
            z > 0 && z < depth - 1)
        {
            int idx = z * slice + y * width + currentX;

            output[idx] = (
                input[idx] +
                input[idx - 1] +
                input[idx + 1] +
                input[idx - width] +
                input[idx + width] +
                input[idx - slice] +
                input[idx + slice]
            ) / 7.0f;
        }
        else if (currentX < width && y < height && z < depth)
        {
            int idx = z * slice + y * width + currentX;
            output[idx] = input[idx];
        }
    }
}

