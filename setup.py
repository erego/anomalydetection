from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyande",
    version="0.0.1",
    author="Eladio Rego",
    author_email="eladiorego@yahoo.es",
    description="Anomaly Detection package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erego/anomalydetection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)