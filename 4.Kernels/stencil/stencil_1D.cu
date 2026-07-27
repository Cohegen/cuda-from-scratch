/*
A 1D implementation of Stencil
This stencil averages three neighboring elements i.e on the right and one on the left
*/

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void stencil_1D(const float* input, float* output, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width)
    {
        // Handle boundary conditions: use the element itself if neighbor is out of bounds
        float left = (i > 0) ? input[i - 1] : input[i];
        float center = input[i];
        float right = (i < width - 1) ? input[i + 1] : input[i];

        output[i] = (left + center + right) / 3.0f;
    }
}

void stencil_1d_cuda(torch::Tensor input, torch::Tensor output, int width)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((width + 255) / 256);

    stencil_1D<<<blocksPerGrid, threadsPerBlock, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}