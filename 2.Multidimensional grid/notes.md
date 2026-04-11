# Introduction to the Module
In the previous module, we used a 1D grid of threads to process 1D data such as vectors.

That works well for arrays like:
- `[1, 2, 3, 4]`

But many real problems are not 1D. Images, matrices, and volumes are naturally 2D or 3D. In those cases, CUDA lets us organize threads and blocks in multiple dimensions so the thread layout matches the data layout more naturally.

In this module, the main idea is simple:
- A grid is made of blocks.
- A block is made of threads.
- Both grids and blocks can be 1D, 2D, or 3D.
- We use thread and block coordinates to decide which part of the data each thread should process.

## 1. Multidimensional Grid Organization
Every thread in a CUDA grid runs the same kernel function. What makes one thread different from another is its position.

CUDA gives each thread built-in coordinates:
- `blockIdx` tells us which block the thread belongs to.
- `threadIdx` tells us the thread's position inside its block.

CUDA also gives:
- `gridDim` for the size of the grid
- `blockDim` for the size of each block

A useful way to think about this is:
- The grid is an array of blocks.
- Each block is an array of threads.

Both of these arrays can be 1D, 2D, or 3D.

## 1.1 Specifying Grid and Block Size
When launching a kernel, we specify:
- the number of blocks in the grid
- the number of threads in each block

CUDA uses the `dim3` type for this. `dim3` has three fields:
- `x`
- `y`
- `z`

Example:

```cpp
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

This means:
- the grid has `32` blocks
- each block has `128` threads
- total threads launched = `32 * 128 = 4096`

We can also compute the grid size from the input size:

```cpp
dim3 dimGrid(ceil(n / 256.0), 1, 1);
dim3 dimBlock(256, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

Here:
- each block has `256` threads
- the number of blocks depends on `n`
- this ensures there are enough threads to cover all elements

If `n = 1000`, then:
- `ceil(1000 / 256.0) = 4`
- so the grid has `4` blocks
- total threads launched = `4 * 256 = 1024`

Some threads may not be needed, so the kernel usually checks whether the thread index is still inside the valid data range.

## 1.2 The 1D Shortcut
For 1D launches, CUDA allows a shorter syntax:

```cpp
kernel<<<16, 256>>>();
```

CUDA treats this as:

```cpp
dim3 dimGrid(16, 1, 1);
dim3 dimBlock(256, 1, 1);
kernel<<<dimGrid, dimBlock>>>();
```

So:
- `16` means 16 blocks in the grid
- `256` means 256 threads per block

Inside the kernel:
- `gridDim.x` will be `16`
- `blockDim.x` will be `256`

## 1.3 Why the Shortcut Works
The shortcut works because `dim3` fills missing dimensions with `1`.

Examples:
- `dim3(16)` becomes `(16, 1, 1)`
- `dim3(8, 4)` becomes `(8, 4, 1)`

That is why writing `<<<16, 256>>>` is valid for a 1D launch.

## 1.4 Built-In Variables Inside Kernels
These are the most important built-in variables:

- Grid size: `gridDim.x`, `gridDim.y`, `gridDim.z`
- Block size: `blockDim.x`, `blockDim.y`, `blockDim.z`
- Thread position in a block: `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
- Block position in the grid: `blockIdx.x`, `blockIdx.y`, `blockIdx.z`

Important facts:
- All threads in the same block have the same `blockIdx`.
- Threads in the same block have different `threadIdx` values.
- Blocks are indexed from `0` up to `gridDim - 1` in each dimension.

## 1.5 Block Configuration
Each block can be arranged as a 1D, 2D, or 3D set of threads.

Example:

```cpp
dim3 dimGrid(2, 2, 1);
dim3 dimBlock(4, 2, 2);
kernelFunction<<<dimGrid, dimBlock>>>(...);
```

This means:
- grid size = `(2, 2, 1)`
- block size = `(4, 2, 2)`

So the grid has:
- 2 blocks in the `x` direction
- 2 blocks in the `y` direction
- 1 block in the `z` direction

Total number of blocks:

```cpp
2 * 2 * 1 = 4 blocks
```

Each block has:
- 4 threads in `x`
- 2 threads in `y`
- 2 threads in `z`

Total threads per block:

```cpp
4 * 2 * 2 = 16 threads
```

Also note:
- all blocks in a grid have the same shape
- the total number of threads in one block cannot exceed `1024` on current CUDA systems

## 2. Mapping Threads to Multidimensional Data
Multidimensional grids are especially useful for images.

Suppose we have an image:
- height = `62` pixels
- width = `76` pixels

Suppose we choose a block size of `16 x 16`.

That means:
- each block has `16` threads in `x`
- each block has `16` threads in `y`

To cover the whole image, we need:
- `ceil(76 / 16.0) = 5` blocks in the `x` direction
- `ceil(62 / 16.0) = 4` blocks in the `y` direction

So total blocks:

```cpp
5 * 4 = 20 blocks
```

We can visualize the grid like this:

```text
            x direction ->
        0      1      2      3      4
      +------+------+------+------+------+
y  0  | B00  | B01  | B02  | B03  | B04  |
   1  | B10  | B11  | B12  | B13  | B14  |
   2  | B20  | B21  | B22  | B23  | B24  |
   3  | B30  | B31  | B32  | B33  | B34  |
      +------+------+------+------+------+
```

Each block contains a `16 x 16` arrangement of threads.

## 2.1 How a Thread Finds Its Pixel
For 2D image processing, each thread is usually mapped to one pixel.

The global row and column handled by a thread are:

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

This formula is very important.

It says:
- `blockIdx` tells us which block we are in
- `threadIdx` tells us where we are inside that block
- multiplying by `blockDim` shifts us to the correct global position

Example:
- thread `(0, 0)` inside block `(1, 0)`
- block size = `(16, 16)`

Then:

```cpp
row = 1 * 16 + 0 = 16
col = 0 * 16 + 0 = 0
```

So that thread processes pixel:

```cpp
Pin(16, 0)
```

## 2.2 Why Boundary Checks Are Needed
Sometimes the grid launches more threads than the data actually needs.

For example:
- the image width is `76`
- but `5` blocks of width `16` give us `80` thread positions in `x`

That means some threads fall outside the image.

So the kernel must check bounds before reading or writing:

```cpp
if (row < n && col < m) {
    // process valid pixel
}
```

Here:
- `n` is the image height
- `m` is the image width

Without this check, threads outside the valid range may access invalid memory.

## 2.3 Launching a 2D Kernel for an Image
Assume:
- `Pin_d` points to the input image in device memory
- `Pout_d` points to the output image in device memory
- `m` is the width
- `n` is the height

The host code can launch a 2D kernel like this:

```cpp
dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
dim3 dimBlock(16, 16, 1);
colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, m, n);
```

In this launch:
- each block contains `16 x 16 = 256` threads
- grid width depends on the image width
- grid height depends on the image height

If the image size is `1500 x 2000`, then:
- blocks in `x` = `ceil(2000 / 16.0) = 125`
- blocks in `y` = `ceil(1500 / 16.0) = 94`
- total blocks = `125 * 94 = 11750`

Inside the kernel:
- `gridDim.x = 125`
- `gridDim.y = 94`
- `blockDim.x = 16`
- `blockDim.y = 16`

## 3. Main Takeaways
- CUDA grids and blocks can be 1D, 2D, or 3D.
- `blockIdx` identifies the block.
- `threadIdx` identifies the thread inside the block.
- `gridDim` and `blockDim` tell us the size of the launch configuration.
- For 2D data such as images, 2D blocks and 2D grids make indexing much easier.
- The global coordinates of a thread are computed from `blockIdx`, `blockDim`, and `threadIdx`.
- Boundary checks are necessary when the total number of launched threads is larger than the data size.
