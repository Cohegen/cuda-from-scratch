/*
Tiled 3D stencil using shared memory.
*/

#include <cuda_runtime.h>

#define STENCIL_3D_TILE_WIDTH 8
#define STENCIL_3D_TILE_RADIUS 1
#define STENCIL_3D_SHARED_WIDTH (STENCIL_3D_TILE_WIDTH + 2 * STENCIL_3D_TILE_RADIUS)
#define STENCIL_3D_FILTER_WIDTH (2 * STENCIL_3D_TILE_RADIUS + 1)

__global__ void stencil3DTiled(
    const float *input,
    const float *coefficients,
    float *output,
    int width,
    int height,
    int depth
)
{
    __shared__ float tile[STENCIL_3D_SHARED_WIDTH]
                         [STENCIL_3D_SHARED_WIDTH]
                         [STENCIL_3D_SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int outputX = blockIdx.x * STENCIL_3D_TILE_WIDTH + tx;
    int outputY = blockIdx.y * STENCIL_3D_TILE_WIDTH + ty;
    int outputZ = blockIdx.z * STENCIL_3D_TILE_WIDTH + tz;
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int linearThread = (tz * blockDim.y + ty) * blockDim.x + tx;
    int sharedCells = STENCIL_3D_SHARED_WIDTH *
        STENCIL_3D_SHARED_WIDTH * STENCIL_3D_SHARED_WIDTH;

    for (int cell = linearThread; cell < sharedCells; cell += threadsPerBlock)
    {
        int tileX = cell % STENCIL_3D_SHARED_WIDTH;
        int tileY = (cell / STENCIL_3D_SHARED_WIDTH) % STENCIL_3D_SHARED_WIDTH;
        int tileZ = cell / (STENCIL_3D_SHARED_WIDTH * STENCIL_3D_SHARED_WIDTH);
        int inputX = blockIdx.x * STENCIL_3D_TILE_WIDTH + tileX - STENCIL_3D_TILE_RADIUS;
        int inputY = blockIdx.y * STENCIL_3D_TILE_WIDTH + tileY - STENCIL_3D_TILE_RADIUS;
        int inputZ = blockIdx.z * STENCIL_3D_TILE_WIDTH + tileZ - STENCIL_3D_TILE_RADIUS;

        tile[tileZ][tileY][tileX] =
            (inputX >= 0 && inputX < width &&
             inputY >= 0 && inputY < height &&
             inputZ >= 0 && inputZ < depth)
                ? input[(inputZ * height + inputY) * width + inputX]
                : 0.0f;
    }

    __syncthreads();

    if (outputX >= width || outputY >= height || outputZ >= depth)
    {
        return;
    }

    float value = 0.0f;

    for (int filterZ = 0; filterZ < STENCIL_3D_FILTER_WIDTH; ++filterZ)
    {
        for (int filterY = 0; filterY < STENCIL_3D_FILTER_WIDTH; ++filterY)
        {
            for (int filterX = 0; filterX < STENCIL_3D_FILTER_WIDTH; ++filterX)
            {
                int filterIndex =
                    (filterZ * STENCIL_3D_FILTER_WIDTH + filterY) *
                    STENCIL_3D_FILTER_WIDTH + filterX;

                value += tile[tz + filterZ][ty + filterY][tx + filterX] *
                    coefficients[filterIndex];
            }
        }
    }

    output[(outputZ * height + outputY) * width + outputX] = value;
}
