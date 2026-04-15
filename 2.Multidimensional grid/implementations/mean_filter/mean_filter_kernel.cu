#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

/*
    CUDA Kernel: Mean Filter (Box Blur)

    Each thread computes ONE pixel value for ONE channel.
    We average values in a square window of size (2*radius + 1)^2.
*/
__global__
void mean_filter_kernel(unsigned char *output, unsigned char* input,
                        int width, int height, int channels, int radius)
{
    // Compute pixel coordinates handled by this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x-axis (width)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y-axis (height)

    // Each block in z-dimension handles one channel (R, G, B, etc.)
    int channel = blockIdx.z;

    // Make sure thread is inside valid image bounds
    if (col < width && row < height && channel < channels)
    {
        // Offset to the start of this channel in memory
        // Layout assumed: [C, H, W]
        int baseOffset = channel * height * width;

        int pixVal = 0;   // Sum of neighboring pixel values
        int pixels = 0;   // Number of valid pixels included in average

        /*
            Iterate over the square window centered at (row, col)
            Example: radius = 1 → 3x3 window
        */
        for (int blurRow = -radius; blurRow <= radius; blurRow++)
        {
            for (int blurCol = -radius; blurCol <= radius; blurCol++)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Boundary check (important at edges of image)
                if (curRow >= 0 && curRow < height &&
                    curCol >= 0 && curCol < width)
                {
                    // Accumulate pixel value
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels++;
                }
            }
        }

        // Write averaged value to output image
        output[baseOffset + row * width + col] =
            static_cast<unsigned char>(pixVal / pixels);
    }
}

/*
    Helper: Ceiling Division

    Used to compute how many blocks we need to cover the image.
    Example: 100 pixels with block size 16 → 7 blocks (not 6)
*/
inline unsigned int cdiv(unsigned int a, unsigned int b)
{
    return (a + b - 1) / b;
}

/*
    Host Function (PyTorch binding)

    - Validates input
    - Configures CUDA launch
    - Calls kernel
*/
torch::Tensor mean_filter(torch::Tensor image, int radius)
{
    // Ensure tensor is on GPU
    TORCH_CHECK(image.device().is_cuda(), "Input image must be a CUDA tensor");

    // Ensure data type is unsigned 8-bit (grayscale or RGB image)
    TORCH_CHECK(image.scalar_type() == torch::kByte,
        "Input image must be of type uint8 (torch.kByte)");

    // Ensure tensor is 3D [C, H, W]
    TORCH_CHECK(image.dim() == 3,
        "Input image must be 3D (C, H, W), got ", image.dim(), "D");

    // Radius must be positive
    TORCH_CHECK(radius > 0, "Radius must be positive, got ", radius);

    // Extract tensor dimensions: [C, H, W]
    const int channels = image.size(0);
    const int height   = image.size(1);
    const int width    = image.size(2);

    // Allocate output tensor (same shape/type as input)
    auto result = torch::empty_like(image);

    /*
        Define CUDA execution configuration:

        blockDim:
            - 16x16 threads per block (good balance for most GPUs)
        gridDim:
            - Enough blocks to cover entire image
            - z-dimension = number of channels
    */
    dim3 blockDim(16, 16);
    dim3 gridDim(
        cdiv(width, blockDim.x),   // number of blocks along width
        cdiv(height, blockDim.y),  // number of blocks along height
        channels                   // one block layer per channel
    );

    // Launch kernel on current CUDA stream
    mean_filter_kernel<<<gridDim, blockDim, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(), // output pointer
        image.data_ptr<unsigned char>(),  // input pointer
        width,
        height,
        channels,
        radius
    );

    // Check if kernel launch failed
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
