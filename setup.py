from setuptools import setup, find_packages

setup(
    name="synapsefs",
    version="1.0.0",
    description="Permutation-Aware Cryptographic Version Control System & Virtual Filesystem for Neural Network Checkpoints",
    author="Shashwat",
    author_email="shashwat060207@gmail.com",
    url="https://github.com/shashwat-cloud-gpu/SynapseFS",
    packages=find_packages(),

    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "zstandard>=0.18.0",
        "safetensors>=0.3.0",
    ],
    extras_require={
        "fuse": ["fusepy>=3.0.1"],
        "dev": ["pytest>=7.0.0", "torch>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "synapsefs=synapsefs.cli:main",
        ],
    },
)
