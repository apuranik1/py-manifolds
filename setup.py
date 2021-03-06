from setuptools import setup, find_packages

NAME = "manifolds"
DESCRIPTION = "Operations on manifolds"
EMAIL = "alokpuranik1@gmail.com"
AUTHOR = "Alok Puranik"
REQUIRES_PYTHON = ">= 3.7.0"
VERSION = "0.0.1"

setup(
    name="manifolds",
    version="0.0.1",
    description="Operations on manifolds",
    author="Alok Puranik",
    author_email="alokpuranik1@gmail.com",
    python_requires=">= 3.7.0",
    packages=find_packages(exclude=["tests"]),
)
