from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


EXTENSION_NAME = "stencil_2d_coarsened_extension"
CUDA_FILE = "stencil_2d_coarsened.cu"


def _source_dir():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def _check_cuda_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab, select Runtime > Change runtime type > GPU."
        )


def compile_extension(verbose=False):
    _check_cuda_runtime()

    cuda_source = (_source_dir() / CUDA_FILE).read_text()
    cpp_source = """
    #include <torch/extension.h>

    void stencil_2d_coarsened_cuda(torch::Tensor input, torch::Tensor output, int width, int height);
    """

    return load_inline(
        name=EXTENSION_NAME,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["stencil_2d_coarsened_cuda"],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=verbose,
    )


def stencil_2d_coarsened(input_tensor, ext=None):
    ext = ext or compile_extension()
    height, width = input_tensor.shape
    output = torch.empty_like(input_tensor)
    ext.stencil_2d_coarsened_cuda(input_tensor.contiguous(), output, width, height)
    return output


def main():
    ext = compile_extension(verbose=True)
    # Further implementation to test the kernel can be added here.


if __name__ == "__main__":
    main()