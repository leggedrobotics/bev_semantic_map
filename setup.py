import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

INSTALL_REQUIRES = [
    # generic
    "numpy",
    "tqdm",
    "pip",
    "torchvision",
    "torch@https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl",
    "torchmetrics",
    "matplotlib",
    "efficientnet-pytorch",
    "opencv-python>=4.6",
    "wandb",
    "icecream",
    "torchshow",
 #   "seaborn",
 #   "pandas",
 #   "scipy",
 #   "scikit-image",
 #   "scikit-learn",
    "pytictac",
]

def make_cuda_ext(name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="bevnet",
        author="Jonas Frey",
        version="1.0.0",
        description="Hot package",
        author_email="jonfrey@ethz.ch",
        packages=find_packages(),
        include_package_data=True,
        package_data={"bevnet": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: NaN",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        license="TBD",
        ext_modules=[
            make_cuda_ext(
                name="bev_pool_ext", module="bevnet.ops.bev_pool", sources=["src/bev_pool.cpp", "src/bev_pool_cuda.cu"]
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
        install_requires=[INSTALL_REQUIRES],
    )
