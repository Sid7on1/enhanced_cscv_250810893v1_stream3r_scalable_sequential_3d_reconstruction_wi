import setuptools
from setuptools import setup, find_packages
from pathlib import Path

# Get the contents of the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="STREAM_3R",
    version="1.0.0",
    author="Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen",
    author_email="contact@example.com",
    description="STREAM 3R: Scalable Sequential 3D Reconstruction Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://nirvanalan.github.io/projects/stream3r",
    packages=find_packages(),
    install_requires=["torch", "numpy", "pandas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)