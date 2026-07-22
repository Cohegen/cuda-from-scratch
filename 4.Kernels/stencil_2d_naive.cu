/*
This is a 2D implementation of a 5-point stencil using naive global memory access.
Each thread calculates one output pixel by averaging itself and its 4 orthogonal neighbors.
*/

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void stencil_2d(const float* input, float* output, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < width - 1 && col > 0 && col < width - 1)
    {
        int idx = row * width + col;
        output[idx] = (
            input[idx] +
            input[idx - 1] +
            input[idx + 1] +
            input[idx - width] +
            input[idx + width]
        ) / 5.0f;
    }
    else if (row < width && col < width)
    {
        // For boundary elements, preserve input value
        output[row * width + col] = input[row * width + col];
    }
}

void stencil_2d_cuda(torch::Tensor input, torch::Tensor output, int width)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (width + 15) / 16);

    stencil_2d<<<blocksPerGrid, threadsPerBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

