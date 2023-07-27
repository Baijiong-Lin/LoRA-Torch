import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loratorch",
    version="0.1.0",
    author="Baijiong Lin",
    author_email="bj.lin.email@gmail.com",
    description="PyTorch reimplementation of low-rank adaptation (LoRA).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Baijiong-Lin/LoRA-Torch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)