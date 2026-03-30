"""Setup script for HIM+HER package."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="him_her",
    version="0.1.0",
    description="Hindsight Inconsistency Mitigation + Hindsight Experience Replay for Multi-Agent RL",
    author="HIM+HER Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": ["black", "flake8", "mypy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
