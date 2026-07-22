/*
Tiled 3D stencil using shared memory.
A tiled 3D stencil whose goal is to increase operations per byte
by loading an 8x8x8 input tile into shared memory also including halo
cells thus catering for top, left, right, bottom, front and back neighbors
of the center element (input tile size becomes 10x10x10).
*/

#include <cuda_runtime.h>

#define TILE_WIDTH 8

__global__ void stencil_3d_tiled(const float* input, float* output, int width, int height, int depth)
{
    // Defining input tile stored in shared memory
    __shared__ float input_tile[TILE_WIDTH + 2][TILE_WIDTH + 2][TILE_WIDTH + 2];

    // Global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int slice = width * height;
    int idx = z * slice + y * width + x;

    // Room in shared memory for halo cells
    int ty = threadIdx.y + 1;
    int tx = threadIdx.x + 1;
    int tz = threadIdx.z + 1;

    // Loading center element
    if (x < width && y < height && z < depth)
    {
        input_tile[tz][ty][tx] = input[idx];
    }

    // Loading left halo
    if (threadIdx.x == 0 && x > 0 && y < height && z < depth)
    {
        input_tile[tz][ty][0] = input[idx - 1];
    }

    // Loading right halo
    if (threadIdx.x == TILE_WIDTH - 1 && x < width - 1 && y < height && z < depth)
    {
        input_tile[tz][ty][TILE_WIDTH + 1] = input[idx + 1];
    }

    // Loading top halo
    if (threadIdx.y == 0 && y > 0 && x < width && z < depth)
    {
        input_tile[tz][0][tx] = input[idx - width];
    }

    // Loading bottom halo
    if (threadIdx.y == TILE_WIDTH - 1 && y < height - 1 && x < width && z < depth)
    {
        input_tile[tz][TILE_WIDTH + 1][tx] = input[idx + width];
    }

    // Loading front halo
    if (threadIdx.z == 0 && z > 0 && y < height && x < width)
    {
        input_tile[0][ty][tx] = input[idx - slice];
    }

    // Loading back halo
    if (threadIdx.z == TILE_WIDTH - 1 && z < depth - 1 && x < width && y < height)
    {
        input_tile[TILE_WIDTH + 1][ty][tx] = input[idx + slice];
    }

    // Wait for all threads to load shared memory tile
    __syncthreads();

    // Calculating output elements
    if (x > 0 && x < width - 1 &&
        y > 0 && y < height - 1 &&
        z > 0 && z < depth - 1)
    {
        output[idx] = (
            input_tile[tz][ty][tx] +      // center
            input_tile[tz][ty][tx - 1] +  // left
            input_tile[tz][ty][tx + 1] +  // right
            input_tile[tz][ty - 1][tx] +  // top
            input_tile[tz][ty + 1][tx] +  // bottom
            input_tile[tz - 1][ty][tx] +  // front
            input_tile[tz + 1][ty][tx]    // back
        ) / 7.0f;
    }
    else if (x < width && y < height && z < depth)
    {
        output[idx] = input[idx];
    }
}

