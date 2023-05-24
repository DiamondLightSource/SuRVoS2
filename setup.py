#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
from pathlib import Path

# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "survos2/_version.py"}

base_path = os.path.abspath(os.path.dirname(__file__))


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
root = Path(__file__).parent
if platform.system() == "Windows":
    filename = str(root / "requirements_windows.txt")
elif platform.system() == "Linux":
    filename = str(root / "requirements.txt")


with open(filename) as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)


if __name__ == "__main__":
    setup(
        # name="SuRVoS2",
        version="2.2",
        # author='DLS',
        # author_email='',
        # license='CC-BY-NC-4.0',
        # url='',
        description="Volumetric Image Segmentation",
        # packages=find_packages(),
        python_requires=">=3.8",
        install_requires=requirements,
        use_scm_version=use_scm,
        setup_requires=["setuptools_scm"],
        classifiers=[
            "Intended Audience :: Developers",
            "Framework :: napari",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: OS Independent",
            "Framework :: snapari",
        ],
        entry_points={
            "napari.plugin": [
                "SuRVoS2 = survos2",
            ],
        },
    )
