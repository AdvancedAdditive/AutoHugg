from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()
    setup(
        name="AutoHugg",
        version="0.0.1",
        author="Felix Schelling",
        author_email="felix.schelling@advanced-additive.de",
        description="proof of concept of ai based slicing",
        long_description=long_description,
        url="",
        packages=find_packages(),
        install_requires=[requirements],
        python_requires=">=3.10",
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "Operating System :: OS Independent",
        ],
    )
