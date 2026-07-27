#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int NUM_BINS = 7;
constexpr int CFACTOR = 4;

// ----------------------------------------------------------------------------
// CUDA Kernel
// ----------------------------------------------------------------------------
__global__ void histo_private_kernel(
    const char* data,
    unsigned int length,
    unsigned int* histo)
{
    __shared__ unsigned int histo_shared[NUM_BINS];

    // Initialize shared histogram
    for (unsigned int bin = threadIdx.x;
         bin < NUM_BINS;
         bin += blockDim.x)
    {
        histo_shared[bin] = 0;
    }

    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int start = tid * CFACTOR;
    unsigned int end = min(start + CFACTOR, length);

    for (unsigned int i = start; i < end; ++i)
    {
        int alphabet_position = data[i] - 'a';

        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            atomicAdd(&histo_shared[alphabet_position / 4], 1);
        }
    }

    __syncthreads();

    // Commit block histogram to global histogram
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
                "input must be torch.uint8");

    auto histogram = torch::zeros(
        {NUM_BINS},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(input.device()));

    constexpr int THREADS = 256;

    const unsigned int length = input.numel();

    const unsigned int logical_threads =
        (length + CFACTOR - 1) / CFACTOR;

    const unsigned int BLOCKS =
        (logical_threads + THREADS - 1) / THREADS;

    histo_private_kernel<<<
        BLOCKS,
        THREADS,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            reinterpret_cast<const char*>(input.data_ptr<uint8_t>()),
            length,
            reinterpret_cast<unsigned int*>(histogram.data_ptr<int>()));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return histogram;
}