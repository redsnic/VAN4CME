from setuptools import setup, find_packages

setup(
    name="van4cme",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy"
    ],
    author="Nicolo Rossi",
    author_email="nicolo.rossi@bsse.ethz.ch",
    description="Gumbel Softmax Variational Autoencoder for Chemical Master Equations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/redsnic/van4cme",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)