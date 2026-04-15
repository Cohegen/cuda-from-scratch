#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

__global__
void colorToGrayscaleKernel(
    unsigned char* __restrict__ Pout,
    const unsigned char* __restrict__ Pin,
    int width,
    int height,
    int channels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int grayOffset = row * width + col;
        int imageSize = width * height;

        // PyTorch uses CHW layout:
        // Channel 0 = R, Channel 1 = G, Channel 2 = B
        unsigned char r = Pin[0 * imageSize + grayOffset];
        unsigned char g = Pin[1 * imageSize + grayOffset];
        unsigned char b = Pin[2 * imageSize + grayOffset];

        Pout[grayOffset] =
            static_cast<unsigned char>(
                0.21f * r + 0.71f * g + 0.07f * b
            );
    }
}


// Ceiling division helper
inline unsigned int cdiv(unsigned int a, unsigned int b)
{
    return (a + b - 1) / b;
}


// Host function (PyTorch binding)
torch::Tensor colorToGrayscale(torch::Tensor image, int radius)
{
    TORCH_CHECK(image.device().is_cuda(), "Input image must be a CUDA tensor");
    TORCH_CHECK(image.scalar_type() == torch::kByte, "Input image must be of type uint8 (torch.kByte)");
    TORCH_CHECK(image.dim() == 3, "Input image must be 3D (C, H, W), got ", image.dim(), "D");
    TORCH_CHECK(radius > 0, "Radius must be positive, got ", radius);

    // PyTorch tensor shape: [C, H, W]
    const int channels = image.size(0);
    const int height   = image.size(1);
    const int width    = image.size(2);

    TORCH_CHECK(channels == 3, "Input image must have 3 channels (RGB), got ", channels);

    // Output is grayscale: [H, W]
    auto result = torch::empty(
        {height, width},
        image.options()
    );

    dim3 blockDim(16, 16);
    dim3 gridDim(
        cdiv(width, blockDim.x),
        cdiv(height, blockDim.y)
    );

    colorToGrayscaleKernel<<<
        gridDim,
        blockDim,
        0,
        c10::cuda::getCurrentCUDAStream()
    >>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        channels
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
