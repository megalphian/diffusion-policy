from setuptools import setup, find_packages
import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def read_requirements():
    """Read requirements from the main project requirements.txt"""
    req_file = os.path.join(os.path.dirname(BASE_DIR), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="diffusion_policy",
    version="0.1.0",
    description="Diffusion Policy for BenchNPIN",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where='.'),  # Look in current directory
    package_dir={'': '.'},  # Packages are in current directory
    include_package_data=True,
    package_data={
        "diffusion_policy": ["**/*.yaml", "**/*.yml", "**/*.json"],
    },
    install_requires=[
        "torch>=1.12.0",
        "torchvision",
        "numpy>=1.20.0",
        "scipy",
        "matplotlib",
        "hydra-core",
        "einops",
        "tqdm",
        "dill",
        "termcolor",
        "psutil",
        "click",
        "imageio",
        "zarr",
        "numcodecs",
        "threadpoolctl",
        "diffusers",
        "accelerate",
        "wandb",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)