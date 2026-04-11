from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("colorToGrayScale.cu").read_text()

    cpp_source = r"""
    #include <torch/extension.h>

    torch::Tensor colorToGrayscale(torch::Tensor image, int radius);
    """

    rgb_to_grayscale_extension = load_inline(
        name="colorToGrayscale_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["colorToGrayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )

    return rgb_to_grayscale_extension


def main():
    """
    Compile CUDA extension, run grayscale kernel, save output image.
    """

    ext = compile_extension()

    # Read image: shape [C, H, W]
    x = read_image("sky.jpg").contiguous().cuda()

    assert x.dtype == torch.uint8
    print("Input image:", x.shape, x.dtype)

    # Call CUDA extension
    y = ext.colorToGrayscale(x, 8)

    print("Output image:", y.shape, y.dtype)

    # torchvision expects CPU tensor
    # Add a channel dimension to the 2D grayscale output [H, W] -> [1, H, W]
    write_png(y.cpu().unsqueeze(0), "output.png")


if __name__ == "__main__":
    main()
