# CUDA C program Structure
- The structure of a CUDA C program reflects the coexistence of a **host** (CPU) and one or more **devices** (GPUs) in the computer.
- The device code is clearly marked with special CUDA C keywords.
- The device code includes functions, or **kernels**, whose code is executed in a data-parallel mannner.
- The execution of a CUDA program is shown below:
```

```

- The execution starts with host code normally the CPU serial code.
- When a kernel function is called, large number of threads are launched on a device to execute the kernel.
- All threads that are launched by a kernel call are collectively called a **grid**.
- Launching a grid typically generates many threads to exploit data parallelism.

# A vector Addition kernel
- Vector addition program is considered to be the Hello,World of new CUDA programmer.
- Let's have a recap of how a traditional vector addition function appears and how it works in either C or C++.
```
void vecAdd(float *A_H,float * B_h,float * C_h,int n)
{
    for(int i =0; i< n; ++i)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main()
{
    //Memory allocation for arrays A,B and C
    //I/O to read A and B,N elements each
    ...

    vecAdd(A,B,C,N);
}
```
- The program above consists of a main function and a function responsible for vector addition.
- Whenever there is a need to distinguish between host and device data,  we suffix names of variables that are used by the host with **_h** and those used by the device(GPU) with **_d**.
- Since our program above only uses the host(CPU), we only suffix our variable names with **_h**.
- The vector addition function arguments A,B,C are pointers.
- From our C++ or C knowledge we may remember that a pointer is typically a container which holds the memory address of another variable.
- A pointer is declared with the asterisk symbol "*" preceding name of the pointer.
- So if we had a pointer P which stores the memory address of a float variable V, we would declare it as follows:
```
float V = 42; //regular float
float *P;//declaration of a pointer
P = &V; //intialization of the pointer
*P = 43;//changing the value of V
```
- An array in either a C or C++ program can be accessed through a pointer that points to its nth element.
- For example the statement **P=&(A[0])** makes P point to the
first element of the array A.
- P[i] becomes a synonym for A[i].
- vecAdd makes the function's first parameter A_h point to the first element of A.
- As a result, A_h[i] in the function body can be used to access A[i] for the array A in the main function.
- We assume the vector to be  added are stored in arrays **A** and **B** that are allocated and intialized in the main program.
- The output vector  is in array **C**, which is also allocated in the main program.
- However, for simplicity purposes the details of how **A**,**B**,**C** are allocated or intialized in the main function are not provided.
- The pointers to these arrays are passed to the vecAdd function, along with the variable **N** that contains the fixed length of the vectors.
- The **vecAdd** function uses a **for-loop** to iterate through the vector elements.
- In the ith iteration, output element C_h[i] receives the sum of **A_h[i]** and **B_h[i]**.
- The vector length parameter **n** is used to control the loop so that the number of iterations matches the length of the vectors.
- The function reads the elements of vectors **A** and **B** and writes the elements of **C** through the pointer **A_h,B_h** and **C_h** respectively.
- A straightforward way to execute vector addition in parallen is to modify the **vecAdd** function and move its calculations to a device.
- The structure of such a modified vecAdd function is shown below:

**ADD VECADD HOST TO DEVICE IMAGE HERE**

- Part 1 allocates space in device (GPU) memory to hold copies of **A**,**B**, and **C** vectors and copies the **A** and **B** vectors from the host memory to the device memory.
- Part2 calles the actual vector addition kernel to launch a grid of threads on the device.
- Part 3 copies the sum vector **C** from the device memory to the host memory and deallocates the three arrays from the device memory.
- To all this we need to update the vecAdd function as follows:
```
void vecAdd(float*A,float*B,float*C,int n)
{
    int size = n *sizeof(float);
    float *d_A,*d_B,*d_C;

    //Part-1:Allocating device memory for A,B and C
    //Copy A and B to device memory


    //Part 2 Calling kernel -to launch a grid of threads
    //to perform the actual vector addition

    //Part 3:Copy C from the device memory
    //free device vectors

}
```

# Device global memory and data transfer
- In CUDA systems, devices are often hardware cards that come with their own dynamic random-access memory called device **global** memory, or simply global memory.
- In out vector addition kernel,before calling the kernel, we need to first allocate space in the devices's global memory and transfer data from the host memory to the allocated space in the device's global memory.
- And after the device execution we also need to transfer the result data from the device global memory back to the host memory and free up the allocated space in the device global memory that is no longer needed.
- The CUDA runtime system provides an API which helps us to perform the above tasks on our behalf.
- These APIs include the **cudaMalloc** and the **cudaFree** functions.

## cudaMalloc
- **cudaMalloc()** - allocates a piece of device global memory for an object.
- It has two parameters, namely:
     - **Address of a pointer** to the allocated object.
     - **Size** of the allocated object in terms of bytes
- The first paremeter which is the address of the pointer variable should be cst to (void **) because the function expects a generic pointer i.e memory allocation function is a generic function meaning it is not restricted to any particular type of objects.
- This parameter allows the **cudaMalloc** function to write the address of the allocated memory into the provided pointer variable regardless of this.

## cudaFree()
- **cudaFree()** - frees object from device global memory
- It only has one parameter:
     - **Pointer** to freed object.

The code below shows the use of **cudaMalloc** and **cudaFree**:
```
float *A_d;
int size=n*sizeof(float);
cudaMalloc((void**)&A_d,size)
...

cudaFree(A_d);
```
- The first argument passed to **cudaMalloc** is the **address** of pointer **A_d** casted into a void pointer.
- When **cudaMalloc**, returns, **A_d** will point to the device global memory region for the A vector.
- The second argument passed in **cudaMalloc** is the size of the region to be allocated.
- Since we're using floating point variables the size of the array will be n*4. Where **n** is the number of elements in the array and 4 is the number of bytes in terms memory space allocated by today's computers to store floating point values.
- After the computation, cudaFree is called with pointer **A_d** as an argument to free the storage space for the A vector from the device global memory.


- Once the host code has allocated space in the device global memory for the data objects, it can request that data be transferred from host to device.

## cudaMemcpy
- This is accomplished by another CUDA API function called **cudaMemcpy**.
- The **cudaMemcpy** function takes four parameters.
- The first parameter is a pointer to the destination location for the data object to be copid.
- The second parameter points to the source location.
- The third parameter specifies the number of bytes copied.
- The fourth parameter indicates the types of memory involved in the copy: from host to host, from host to device, from device to host and from device to device.
- The **vecAdd** function calls the **cudaMemcpy** function to copy the **A_h** and **B_h** vectors from the host memory to **A_d** and **B_d** in the device memory before adding them and to copy the **C_d** from device memory to **C_h** the host memory after the addition is has been done.
- Assuming that the values of **A_h**, **B_h**, **A_d**, **B_h** and size have already been set, the three **cudaMemcpy** calls are as shown below.
```
cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
cudaMemcpy(B_d,B_h,size,cudaMemcpyHostToDevice);

cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);
```
- The two symbolic constants , **cudaMemcpyHostToDevice** and **cudamMemcpyDeivceToHost** explain the direction of data transfer we wish to take e.g host to device and device to host.
- Now the **vecAdd** function is updates to allocate device global memory memory, to request data transfers and calls the kernel that performs the actual vector addition.
- The updated **vecAdd** function is shown below:
```
void vectAdd(float*A_h,float *B_h,float * C_h,int n)
{
    int size = n*sizeof(float);
    float *A_d, B_d , *C_d;
    
    //allocating memory spaces in device global memory 
    cudaMalloc((void **)&A_d,size);
    cudaMalloc((void **) &B_d,size);
    cudaMalloc((void**)&C_d,size);
    
    //data transfer from host to device
    cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,size,cudaMemcpyHostToDevice);

    //Kernel invocation code to be showed
    // in the next updated vecAdd function
    ...

    //data transfer from device to host memory
    cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);

    //freeing up space in device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

#Kernel Functions and threading
- In CUDA C, a kernel function specifies the code to be executed by all threads during a parallel phase.
- When a program's host code calls a kernel, the CUDA runtime system launches a grid of threads that are organized into two-level hierarchy.
- Each grid is organized as an array of **thread blocks** , which refer to as blocks for simplicity purposes.
- All blocks of a grid are of the same size, each block can contain up to 1024 thread on current systems.
- The total number of threads in each threads is specified by the host code when a kernel is called.
- The same kernel can be called with different numbers of threads at different parts of the host code.
- For a given grid of threads, the number of threads in a block is available in a built-in variable named **blockDim**.
- The **blocDim** variable is a struct with three unsigned integer fields **(x,y,z)** that helps us as programmers to organize the threads to a one-, two-,three- dimensional array.
- For one-dimension organization  the **x** field is used,while **x** and **y** fields  are used for two-dimension organization and **x**,**y** and **z** fields are used for three-dimension organization.
- The choice of dimensions depends on dimesionality of the data.
- In general is recommended that the number of threads in each dimension of thread block be a multiple of 32 for hardware effficiency reasons.

**ADD ONE DIMENSIONAL THREAD BLOCK HERE**

- CUDA threads have access two more built-in variables **threadIdx** and **blockIdx** that allow threads to distinguish themselves from each other and to determine the area of data each thread is to work on.
- The **threadIdx** variable gives each thread a unique coordinate within a block.
- In the figure above since we are using a one-dimensional thread organization, only **threadIdx** is used.
- The **threadIdx** is shown in each thread box.
- The first thread in each block has a value 0 in its **threadIdx.x** variable,the second thread has value 1, the third has a value of 2 and so on.
- The **blockIdx** variable gives all threads in a block a common block coordinate.
- For example if we have many blocks,all threads in the first block have a value of 0  in their **blockIdx.x** variables, the second thread block have value of 1 and so on.
- A unique global index i is calculated as **i=blockIdx.x * blockDim.x + threadIdx.x**.
- For example if our **blockDim** is 256 , the values of i of threads in block 0 range from 0 and 255.
- The **i** values of threads in block 1 range is 256 to 511.
- The **i** values of threads in block 2 range from 512 to 767.
- Since each threads uses **i** to access **A**, **B** and **C**  these threads cover the first 768 iterations of the original loop.
- By lauching a grid  with a larger number of blocks, one can process larger vectors.
- By launching or n or more threads, one can process vectors of length n.
- Our kernel function is describe below.
- There's a CUDA-C specific keyword "__global__" in front of the declaration of the **vecAddKernel** function.
- This keyword indicates that the function is a kernel and it can be called to generate a grid of threads.
```
//compute vector sum C=A+B
//each thread performs one pair-wise addition

__global___
void vecAddKernel(float *A,float*B,float*C,int n)
{
    int i = threadIdx.x + blockDim.x * blockKIdx.x;
    if(i < n)
    {
        C[i] = A[i] + B[i];
    }
}
```

# Calling Kernel functions
- The code below shows that when the host code calls a kernel,it sets the grid and thread block dimensios via **execution configuration parameters**.
```
int vecAdd(float *A, float*B,float* C,int n)
{
    //A_d,B_d,C_d allocations and copies omitted

    //Launch ceil(n/256) blocks of 256 threads each
    vecAddKernel <<<ceil(m/256.0), 256>>>(A_d,B_d,C_d,n);

}

//a complete version of host code in the vecAdd function
void vecAdd(float* A, float * B, float*C,int n)
{
    float *A_d,*B_d,*C_d;
    int size = n*sizeof(float);

    cudaMalloc((void **) &A_d,size);
    cudaMalloc((void **),&B_d,size);
    cudaMalloc((void **),&C_d,size);

    cudaMemcpy(A_d,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,size,cudaMemcpyHostToDevice);

    vecAddKernel<<< ceil(n/256.0),256>>>(A_d,B_d,C_d,n);
    cudaMemcpy(C,C_d,size,cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

- The configuration parameters are given between the **<<<** and **>>>**.
- The first configuration parameter gives the number of blocks in the grid.
- The second specifies the number of threads in each block.
- In the code above we have 256 threads in each block.
- To ensure we have enough threads in the grid to cover all vector elements, we need to set the number of blocks in the grid to the ceiling division (rounding up the quotient to the immediate higher integer value) of the desired number of threads (n in this case) by the thread block size (256 in this case).
- One way is doing this it appy the C ceiling function to **n/256.0**.
- Using the floating-point value 256.0 ensures that we generate a floating value for the division so that the ceiling function can round it up corrrectly.
- For example if we want 1000 threads, we would launch ceil(1000/256.0) = 4 thread blocks.
- As a result, the statement will lauch 4 x 256 = 1024 threads.
- With the if(i < n) statement, the first 1000 threads will perform addition of 1000 vector elements.
- The remain 23 will not.
- If **n** is 750, three thread blocks will be used, if n is 10,000,000 39063 blocks will be used.
- However, GPUs are limited to the amount of thread blocks they may have.
- For instance a small GPU with a small amout of execution resources may execute only one or two of these thread blocks in parallel.

