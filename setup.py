from setuptools import setup, find_packages

setup(
    name="flusim",
    version="0.0.1",
    description="A package for simulating and computing statistics of a small-scale flu-spread problem.",
    package_dir={"": "app"},
    author="Samuel Wang",
    author_email="swang3068@gatech.edu",
    url="https://github.com/ZebraAlgebra/flu-sim",
    license="MIT",
    packages=find_packages(where="app"),
)

with open("app/README.md", "r") as f:
    long_description = f.read()
