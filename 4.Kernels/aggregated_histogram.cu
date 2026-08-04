#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int NUM_BINS = 7;

// ----------------------------------------------------------------------------
// CUDA Kernel
// ----------------------------------------------------------------------------
__global__ void histo_private_kernel(
    const char* data,
    unsigned int length,
    unsigned int* histo)
{
    __shared__ unsigned int histo_shared[NUM_BINS];

    // ------------------------------------------------------------------------
    // Initialize shared histogram
    // ------------------------------------------------------------------------
    for (unsigned int bin = threadIdx.x;
         bin < NUM_BINS;
         bin += blockDim.x)
    {
        histo_shared[bin] = 0u;
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Aggregated histogram construction
    // ------------------------------------------------------------------------
    unsigned int accumulator = 0;
    int prevBinIdx = -1;

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < length; i += stride)
    {
        int alphabet_position = data[i] - 'a';

        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            int bin = alphabet_position / 4;

            if (bin == prevBinIdx)
            {
                // Same bin as previous character
                ++accumulator;
            }
            else
            {
                // Flush previous accumulated count
                if (accumulator > 0)
                {
                    atomicAdd(&histo_shared[prevBinIdx], accumulator);
                }

                prevBinIdx = bin;
                accumulator = 1;
            }
        }
    }

    // Flush the final accumulated run
    if (accumulator > 0)
    {
        atomicAdd(&histo_shared[prevBinIdx], accumulator);
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Commit block histogram to global histogram
    // ------------------------------------------------------------------------
    for (unsigned int bin = threadIdx.x;
         bin < NUM_BINS;
         bin += blockDim.x)
    {
        unsigned int value = histo_shared[bin];

        if (value > 0)
        {
            atomicAdd(&histo[bin], value);
        }
    }
}

// ----------------------------------------------------------------------------
// Launcher
// ----------------------------------------------------------------------------
torch::Tensor histogram_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8,
                "input must have dtype torch.uint8");
    TORCH_CHECK(input.is_contiguous(),
                "input must be contiguous");

    auto histogram = torch::zeros(
        {NUM_BINS},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(input.device()));

    constexpr int THREADS = 256;
    constexpr int BLOCKS = 256;

    histo_private_kernel<<<
        BLOCKS,
        THREADS,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            reinterpret_cast<const char*>(input.data_ptr<uint8_t>()),
            static_cast<unsigned int>(input.numel()),
            reinterpret_cast<unsigned int*>(histogram.data_ptr<int>()));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return histogram;
}

// ----------------------------------------------------------------------------
// PyBind
// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "histogram",
        &histogram_cuda,
        "Aggregated Histogram CUDA");
}