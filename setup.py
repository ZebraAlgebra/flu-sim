from setuptools import setup, find_packages

with open("app/README.md", "r") as f:
    long_description = f.read()

description = """
Simulation, computation, visualization of some statistics, probabilities
for a small-scale flu-spread problem.
"""

setup(
    name="flusim",
    version="0.0.2",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "app"},
    author="Samuel Wang",
    author_email="swang3068@gatech.edu",
    url="https://github.com/ZebraAlgebra/flu-sim",
    license="MIT",
    packages=find_packages(where="app"),
)
