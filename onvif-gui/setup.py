from setuptools import setup, find_packages
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="onvif-gui",
    version="1.0.3",
    author="Stephen Rhodes",
    author_email="sr99622@gmail.com",
    description="GUI program for onvif",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'gui_scripts': [
            'onvif-gui=gui.main:run'
        ]
    }
)