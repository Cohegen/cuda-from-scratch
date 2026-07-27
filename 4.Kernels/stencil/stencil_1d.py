from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


EXTENSION_NAME = "stencil_1d_extension"
CUDA_FILE = "stencil_1D.cu"


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

    void stencil_1d_cuda(torch::Tensor input, torch::Tensor output, int width);
    """

    return load_inline(
        name=EXTENSION_NAME,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["stencil_1d_cuda"],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=verbose,
    )


def stencil_1d(input_tensor, ext=None):
    ext = ext or compile_extension()
    width = input_tensor.size(0)
    output = torch.empty_like(input_tensor)
    ext.stencil_1d_cuda(input_tensor.contiguous(), output, width)
    return output


def main():
    ext = compile_extension(verbose=True)
    # Further implementation to test the kernel can be added here.


if __name__ == "__main__":
    main()