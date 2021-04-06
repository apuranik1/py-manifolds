from setuptools import setup, find_packages

setup(
    name="manifolds",
    version="0.0.1",
    description="Operations on manifolds",
    author="Alok Puranik",
    author_email="alokpuranik1@gmail.com",
    python_requires=">= 3.7.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "jax == 0.2.10",
        "numpy ~= 1.20",
        "pytest ~= 6.0",
    ],
)
