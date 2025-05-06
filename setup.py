from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="autofhr",
    version="0.1.0",
    description="Automated Fetal Heart Rate Localization using GenAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cliffordlab/AutoFHR-Leveraging-Neural-Sequential-Modeling-for-Ultrasound-Analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "autofhr-train=src.train:main",
            "autofhr-predict=src.predict:main",
        ],
    },
) 