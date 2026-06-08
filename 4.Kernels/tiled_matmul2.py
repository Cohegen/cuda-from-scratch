from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


EXTENSION_NAME = "tiled_matmul2_extension"
CUDA_FILE = "tiled_matmul2.cu"


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

    torch::Tensor tiled_matmul2(torch::Tensor M, torch::Tensor N);
    """

    return load_inline(
        name=EXTENSION_NAME,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["tiled_matmul2"],
        with_cuda=True,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=verbose,
    )


def tiled_matmul2(m, n, ext=None):
    ext = ext or compile_extension()
    return ext.tiled_matmul2(m.contiguous(), n.contiguous())


def main():
    ext = compile_extension(verbose=True)

    width = 50
    m = torch.ones((width, width), device="cuda", dtype=torch.float32)
    n = torch.full((width, width), 2.0, device="cuda", dtype=torch.float32)

    result = tiled_matmul2(m, n, ext)
    expected = torch.matmul(m, n)

    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Output:", result.shape, result.dtype, result.device)
    print("Matches torch.matmul:", torch.allclose(result, expected))
    print("Top-left 4x4 result:")
    print(result[:4, :4].cpu())


if __name__ == "__main__":
    main()
