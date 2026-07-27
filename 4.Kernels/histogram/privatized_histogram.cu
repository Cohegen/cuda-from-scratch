/*
An optimized version of histogram, whereby 
instead of all threads are contending for the same
memory loction here threads in the blocks contend with 
each other
*/
#define NUM_BINS 7   // since 26 letters / 4 ≈ 7 bins

__global__ void histo_private_kernel(
    const char* data,
    unsigned int length,
    unsigned int* histo   // size = gridDim.x * NUM_BINS + NUM_BINS
){
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    // Each block's private histogram starts here
    unsigned int* private_histo = &histo[blockIdx.x * NUM_BINS];

    // Final global histogram is stored at the beginning
    unsigned int* global_histo  = &histo[0];

    
    // Phase 1: Build private histogram
    if (i < length)
    {
        int alphabet_position = data[i] - 'a';

        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            int bin = alphabet_position / 4;

            // Contention only within the block
            atomicAdd(&private_histo[bin], 1);
        }
    }

    // Ensure all updates to private histogram are done
    __syncthreads();

    
    // Phase 2: Merge into global histogram
    for (unsigned int bin = tid; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int binValue = private_histo[bin];

        if (binValue > 0)
        {
            // Now only a few atomics per block
            atomicAdd(&global_histo[bin], binValue);
        }
    }
}