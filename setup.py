from setuptools import setup, find_packages


setup(
    name="torchph",
    version="0.0.0",
    packages=setuptools.find_packages(exclude=('tests*',))
)
