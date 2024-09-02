# %%
from setuptools import setup, find_packages

setup(
    name="coLamB",
    version="0.1.2",
    description="A package for simulating binary coherent phase field with elasticity.",
    author="Furukawa Tan",
    packages=find_packages(),
    install_requires=[
        # "cupy",
        "numpy",
        "matplotlib",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="MIT",
)

# pip install -e .

# %%
