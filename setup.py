from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="distribution_matching",
    version='0.01',
    description="Experiments for studying distribution matching.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/sambklein/distribution_matching",
    author="Sam Klein",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
        "scipy",
        "pandas",
        "nflows",
        "xgboost",
        "geomloss"
    ],
    dependency_links=[],
)
