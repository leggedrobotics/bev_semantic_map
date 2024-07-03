from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["perception_bev_learning_ros"],
    package_dir={"": "src"},
)

setup(**d)
