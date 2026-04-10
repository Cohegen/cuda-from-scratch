# Introduction to the module
- In the last module we looked at how we can lauch a one-dimensional grid of threads by calling a kernel function to operate on elements of a one-dimensional arrays.
- However, what is we're dealing with multidimensional data would we still launch a  one-dimensional grid of threads?
- No that wouldn't even be possible.
- In this module you'll learn how threads are organized and learn how threads and blocks can be used to process multidimensional arrays.

## 1.Multidimensional grid organization
- In CUDA, all threads in a grid execute the same kernel function, and they rely on coordinates i.e thread indices, to distinguish themselves from each other to identify the appropriate portion of the data to process.
- From our previous module we learned that a grid constists of one or more blocks, and each block consists of one or more threads.
- All threads in a block share the same block index, which can be accessed via **blockIdx** variable.
- Each threads within the block has its unique index which can be accessed via the **threadIdx** variable.
- When a thread executes a kernel function, references to the **blockIdx** and **threadIdx** variables return the coordinates of the thread.
- A grid is a 3D array of block, and each blocks is 3D array of threads.
- Now when calling the kernel we need to specifiy the size of the gird and blocks in each dimension.
- Let's have a recap on what we learned in module 1.
- We know that in order for us to specify things like size of the grid and block dimensions, we specify them using the execution configuration parameters.
- The first execution configuration parameter specifies the dimensions of the grid in  number of blocks.
- The second specifies the dimension of each block in the number of threads.
- Each parameter has the type **dim3** which is an integer vector type of three elemetns **x**, **y** and **z**.
- These three elements specify the sizes of the three dimensions.
- In this example, the following code can be used to call the **vecAddkernel()** kernel function and generate a 1D grid that consists of 32 blocks, each of which consists of 128 threads.
- Then this will make the total number of threads in the grid to be 128*32 = 4096.
```
dim3 dimGrid(32,1,1)
dim3 dimBlock(128,1,1)
vecAddkernel <<<dimGrid, dimBlock>>>(...);
```
- The grid and block dimensions can also be calculated from other variables.
```
dim3 dimGrid(ceil(n/256.0),1,1);
dim3 dimBlock(256,1,1);
vecAddkernel <<<dimGrid, dimBlock>>> (...);

```
- This allows the number of blocks to vary with the size of the vectors so that the grid will have enough threads to cover all vector elements.
- In the code above we can see that we have to fix the block size at 256.
- The value of the variable n at the kernel call tme will determine the dimension of the grid i.e the number of blocks within the grid.
- If say n is equal to 1000, the grid will consist of four blocks.
- This would enable the threads to cover a vector of maximum size of 1024 elements.

### 1.1 The **1D shortcut** for kernel launch
- Previously when describing the dimensionality of our grid and threadblocks
we did that as follows:
```
dim3 dimGrid(n,1,1);
dim3 dimBlock(256,1,1);
vecAddkernel<<<dimGrid,dimBlock>>>(...);
```
- where **n** is the number of blocks we wish out grid to have.
- However, CUDA allows a shortcut for 1D cases:
```
kernel <<<dimGrid,dimBlock>>>(..);
```
- Say we write something like this:
```
kernel <<< 16,256>>>();
```
- What CUDA does internally is that it interprets it as:
```
dim3 dimGrid(16,1,1);
dim3 dimBlock(256,1,1);
```
- Within the kernel function, the **x** field of the variables **gridDim** and **blockDim** are preinitalized according the values of the execution configuration parameters.
- If n is equal to 4000, refernces to **gridDim.x** and **dimBlock.x** in the **vecAddkernel** kernel will result in 16 and 256 respectively.

### 1.2 Why does the **1D** shortcut work?
- If you have some basic knowledge of C++ you maybe familiar with structs and constructors.
- CUDA sues a **dim3** struct type as follows:
```
struct dim
{
    unsigned int x,y,z;
}
```
- And it also has a constructor like:
```
dim3(unsigned int x =1, unsigned int y = 1, unsigned int z = 1);
```
- So when intialize **dim3** as follows:
      - **dim3(16)** it becomes **(16,1,1)**.
      - **dim3(8,4)** it becomes **(8,4,1)**.

### 1.3 Builti-in variables inside kernels
- CUDA gives us automatic variables.
      -1. **Grid Size**: **gridDim.x ,gridDim.y, gridDim.z**
      - 2 **Block size** : **blockDim.x, blockDim.y, blockDim.z**
      - 3 **Thread position** : **threadIdx.x,threadIdx.y, threadIdx.z**
      - 4 **Block position** : **blockIdx.x, blockIdx.y, blockIdx.z**

- In CUDA C the allowed values of **gridDim.x** range from 1 to (2^31) -1, and those of **gridDim.y** and **grdiDim.z** range from 1 to (2^16) -1 (65,535).
- All threads in a block share the same **blockIdx.x , blockIdx.y** and **blockIdx.z** values.
- Among blocks, the **blockIdx.x** values ranges from 0 to **gridDim.x-1**, the **blockIdx.y** value ranges from 0 to **gridDim.y-1** and the **blockIdx.z** value ranges from 0 to **gridDim.z-1**.

### 1.4  Block Configuration
- Each block is organized into a 3D array of treads.
- Two-dimensional blocks can be created by setting **blockDim.z** to 1 , as in the **vecAddkernel** example.
- All blocks in a grid have the same dimensions and sizes.
- The number of threads in each dimension of a block is specified by the second execution configuration parameter in the kernel call.
- Within the kernel call this configuration parameter can be accessed as the **x,y** and **z** field of **blockDim**.
- The total size of a block in the current CUDA systems is limited to 1024 threads.
- These threads can be distributed across three dimensions in any way as long as the total number of threads does not exceed 1024.

**ADD Grid DIAGRAM HERE!!**

- A grid and its blocks do not need the same dimensionality.
- A grid can have higher dimensionality that its blcoks and vice versa.
- For example in the above diagram, we see a grid with **gridDim** of (2,2,1) and a **blockDim** of (4,2,2).
- Such a grid would be created as the following host code:
```
dim3 dimGrid(2,2,1);
dim3 dimBlock(4,2,2);
KernelFunction <<<dimGrid, dimBlock>>>(...);
```
- The each  grid consists of four of blcks organized into a 2x2 array.
- Each block is labeled with **(blockIdx.y,blockIdx.x)**.
- For example the block (1,0) has **blockIdx.y** = 1 and **blockIdx.x** = 0.
- Each **threadIdx** also constists of three field: x coordinate **threadIdx.x**, y coordinate **threadIdx.y** and the z coordinate **threadIdx.z**.
- So if we have a grid of shape (2,2,1), it means that there's:
      - 2 blocks in the x-direction
      - 2 blocks in y-direction
      - 1 in z-direction
- So total blocks  = 2 * 2 * 1= 4 blocks
- Also if the had block of dimension (4,2,2),it means that there's:
      - 4 threads in the x-direction
      - 2 in y-direction
      - 2 in z-direction

- The total threads per block is:
```
4 x 2 x 2 = 16 threads
````

## 2 Mapping threads to multidimensional data
- Say we have picture **P** which havs 62 pixels in the vertical or y direction and 76 pixels the horizontal or x axis.
- Assume that we decided to use a 16 x 16 block, with 16 threads in the x direction and 16 threads in the y direction.
- We will need four blocks in the y direction and five blocks in the x direction, which resulrt to 4 x 5 = 20 blocks.
- One Grid  looks like this:
```
            X direction →
        0      1      2      3      4
      +------+------+------+------+------+
Y  0  | B00  | B01  | B02  | B03  | B04  |
   1  | B10  | B11  | B12  | B13  | B14  |
   2  | B20  | B21  | B22  | B23  | B24  |
   3  | B30  | B31  | B32  | B33  | B34  |
      +------+------+------+------+------+
```

- Each block looks like this:
```
        threadIdx.x →
        0 1 2 3 ... 15
      +------------------+
   0  | o o o o ... o    |
   1  | o o o o ... o    |
   2  | o o o o ... o    |
   .  | o o o o ... o    |
  15  | o o o o ... o    |
      +------------------+
threadIdx.y ↓
```

- Each thread is assigned to process a pixel whose y and x coordinates are derived from its **blockIdx**


