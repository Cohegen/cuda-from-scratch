
__global__ void histo_private_kernel(
    char* data,
    unsigned int length,
    unsigned int* histo
){
    //intialize privatized bins
    __shared__ unsigned int histo_shared[NUM_BINS];
    for(unsigned int bin= threadIdx.x;bin<NUM_BINS;bin+=blockDim.x)
    {
        histo_shared[bin]=0u;
    }
    __syncthreads();

    //histogram
    unsigned int i = blockIdx.x*blockDim.x+threadIdx;
    if(i <length)
    {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >=0 && alphabet_position <26)
        {
            atomicAdd(&histo_shared[alphabet_position/4],1);
        }
    }
    __syncthreads();
    //merge
    for(unsigned int bin=threadIdx.x;bin<NUM_BINS;bin+=blockDim.x)
    {
        unsigned int binValue = histo_shared[bin];
        if(binValue>0)
        {
            atomicAdd(&histo[bin],binValue);
        }
    }
}