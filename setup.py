#!/usr/bin/env python3
"""
Setup script for CleaveRNA - RNA cleavage site prediction tool
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "CleaveRNA - RNA cleavage site prediction tool using machine learning"

setup(
    name="CleaveRNA",
    version="1.0.0",
    author="reytakop",
    author_email="your.email@example.com",  # Replace with your actual email
    description="RNA cleavage site prediction tool using machine learning and structural analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reytakop/CleaveRNA",
    packages=find_packages(),
    license="MIT License with Attribution Requirements",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "argparse",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "cleaverna=CleaveRNA.CleaveRNA:main",
            "cleaverna-feature=CleaveRNA.Feature:main",
        ],
    },
    include_package_data=True,
    package_data={
        "CleaveRNA": ["*.md", "*.txt"],
    },
    keywords="RNA, cleavage, prediction, machine learning, bioinformatics, structural biology",
    project_urls={
        "Bug Reports": "https://github.com/reytakop/CleaveRNA/issues",
        "Source": "https://github.com/reytakop/CleaveRNA",
        "Documentation": "https://github.com/reytakop/CleaveRNA/wiki",
    },
)