#include <iostream>
#include <cuda_runtime.h>

using namespace std;

//kernel function (it runs on the GPU)
__global__ void vecSubKernel(float *A,float*B,float *C,int n)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;

    if(i < n)
    {
        C[i] = B[i] - A[i];
    }
}




void vecSub(float*A,float*B,float *C,int n)
{
    //declaring device memory locations for each vector
    float *A_d,*B_d,*C_d;
    int size = n *sizeof(float);

    //allocating device memory resource
    cudaMalloc((void**)&A_d,size);
    cudaMalloc((void**)&B_d,size);
    cudaMalloc((void**)&C_d,size);

    //data transfer
    cudaMemcpy(A_d,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,size,cudaMemcpyHostToDevice);


    

    //kernel lauch configuration
    int blockSize = 256; // number of threads within the block
    int numBlocks = (n+ blockSize - 1) /blockSize;

    //Launching kernel
    vecSubKernel <<<numBlocks,blockSize >>>(A_d,B_d,C_d,n);

    //copying result back to host
    cudaMemcpy(C,C_d,size,cudaMemcpyDeviceToHost);
    //freeing up  device memory space
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    int n = 1000; //number of elements our respective vectors will have
    int size = n*sizeof(float);

    //Allocating host memory
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];

    //Intializing input vectors
    for(int i=0;i<n;i++)
    {
        A[i] = i*1.0f;
        B[i] = i * 2.0f;
    }

    //Calling CUDA function
    vecSub(A,B,C,n);

    //printing the first 10 results
    cout << "Results of the first 10 elements in our resultant vector : " <<endl;
    for(int i=0;i<10;i++)
    {
        cout << "C[" << i << "] = " << C[i] << endl;

    }

    //Freeing host memory
    delete [] A;
    delete [] B;
    delete [] C;


    return 0;

}
