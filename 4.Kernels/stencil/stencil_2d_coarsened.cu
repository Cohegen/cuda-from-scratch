/*
A 2D stencil kernel where instead of an individual thread calculating one output element,
it instead calculates two or more output elements using thread coarsening.
*/

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_WIDTH 16
#define COARSE_FACTOR 2

__global__
void stencil_2d_coarsened(const float* input,
                          float* output,
                          int width,
                          int height)
{
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x;

    int rowOffset = row * width;

    for (int c = 0; c < COARSE_FACTOR; c++)
    {
        int currentCol = col + c * TILE_WIDTH;

        if (row > 0 &&
            row < height - 1 &&
            currentCol > 0 &&
            currentCol < width - 1)
        {
            int idx = rowOffset + currentCol;

            output[idx] =
            (
                input[idx] +
                input[idx - 1] +
                input[idx + 1] +
                input[idx - width] +
                input[idx + width]
            ) / 5.0f;
        }
        else if (row < height && currentCol < width)
        {
            int idx = rowOffset + currentCol;
            output[idx] = input[idx];
        }
    }
}

void stencil_2d_coarsened_cuda(torch::Tensor input, torch::Tensor output, int width, int height)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (width + (TILE_WIDTH * COARSE_FACTOR) - 1) / (TILE_WIDTH * COARSE_FACTOR),
        (height + TILE_WIDTH - 1) / TILE_WIDTH
    );

    stencil_2d_coarsened<<<blocksPerGrid, threadsPerBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width,
        height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

