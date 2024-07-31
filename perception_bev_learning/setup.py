import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

INSTALL_REQUIRES = [
    "numpy",
    "tqdm",
    "pip",
    "torchmetrics",
    "matplotlib",
    "seaborn",
    "pandas",
    "pytictac",
    "neptune",
    "prettytable",
    "termcolor",
    "wget",
    "black",
    "open3d",
    "h5py",
    "moviepy",
    "ninja",
    "lightning",
    "simple_parsing",
    "scikit-image",
    "efficientnet_pytorch",
    "opencv-python",
    "chardet",
]


def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):
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
        name="perception_bev_learning",
        author="Manthan Patel",
        version="2.0.0",
        description="Hot package for BEV learning",
        install_requires=[INSTALL_REQUIRES],
        author_email="patelm@ethz.ch",
        packages=find_packages(),
        include_package_data=True,
        package_data={"perception_bev_learning": ["*/*.so"]},
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
                name="sparse_conv_ext",
                module="perception_bev_learning.ops.spconv",
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(
                            *"perception_bev_learning.ops.spconv".split("."), "include/"
                        )
                    )
                ],
                sources=[
                    "src/all.cc",
                    "src/reordering.cc",
                    "src/reordering_cuda.cu",
                    "src/indice.cc",
                    "src/indice_cuda.cu",
                    "src/maxpool.cc",
                    "src/maxpool_cuda.cu",
                ],
                extra_args=["-w", "-std=c++17"],
            ),
            make_cuda_ext(
                name="bev_pool_ext",
                module="perception_bev_learning.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="iou3d_cuda",
                module="perception_bev_learning.ops.iou3d",
                sources=[
                    "src/iou3d.cpp",
                    "src/iou3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="perception_bev_learning.ops.voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/scatter_points_cuda.cu",
                    "src/voxelization_cpu.cpp",
                    "src/voxelization_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_ext",
                module="perception_bev_learning.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/points_in_boxes_cpu.cpp",
                ],
                sources_cuda=[
                    "src/roiaware_pool3d_kernel.cu",
                    "src/points_in_boxes_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="ball_query_ext",
                module="perception_bev_learning.ops.ball_query",
                sources=["src/ball_query.cpp"],
                sources_cuda=["src/ball_query_cuda.cu"],
            ),
            make_cuda_ext(
                name="knn_ext",
                module="perception_bev_learning.ops.knn",
                sources=["src/knn.cpp"],
                sources_cuda=["src/knn_cuda.cu"],
            ),
            make_cuda_ext(
                name="assign_score_withk_ext",
                module="perception_bev_learning.ops.paconv",
                sources=["src/assign_score_withk.cpp"],
                sources_cuda=["src/assign_score_withk_cuda.cu"],
            ),
            make_cuda_ext(
                name="group_points_ext",
                module="perception_bev_learning.ops.group_points",
                sources=["src/group_points.cpp"],
                sources_cuda=["src/group_points_cuda.cu"],
            ),
            make_cuda_ext(
                name="interpolate_ext",
                module="perception_bev_learning.ops.interpolate",
                sources=["src/interpolate.cpp"],
                sources_cuda=["src/three_interpolate_cuda.cu", "src/three_nn_cuda.cu"],
            ),
            make_cuda_ext(
                name="furthest_point_sample_ext",
                module="perception_bev_learning.ops.furthest_point_sample",
                sources=["src/furthest_point_sample.cpp"],
                sources_cuda=["src/furthest_point_sample_cuda.cu"],
            ),
            make_cuda_ext(
                name="gather_points_ext",
                module="perception_bev_learning.ops.gather_points",
                sources=["src/gather_points.cpp"],
                sources_cuda=["src/gather_points_cuda.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
        zip_safe=False,
    )
