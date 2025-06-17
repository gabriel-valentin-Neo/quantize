from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="neoQuant",
    ext_modules=[
        CUDAExtension(
            name="neoQuant",
            sources=["calculando_scales.cu"],
            include_dirs=[
                "/usr/local/lib/python3.12/dist-packages/pybind11/include"  # Caminho para pybind11 headers
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17", "-O3", "-arch=sm_100a",  "-lcuda"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
