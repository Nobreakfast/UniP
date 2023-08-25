import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
exec(open("unip/version.py").read())

setuptools.setup(
    name="UniP",
    version=__version__,
    author="Haocheng Zhao",
    author_email="Haocheng.Zhao@hotmail.com",
    description="A unified framework for Pruning in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nobreakfast/UniP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["torch", "torchvision", "numpy", "thop", "tqdm"],
    python_requires=">=3.6",
)
