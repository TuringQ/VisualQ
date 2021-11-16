from __future__ import print_function
from setuptools import setup, find_packages
import sys

setup(
    name="deepquantum",
    version="1.4.3",
    author="Turing Quantum",  #作者名字
    author_email="",
    description="a Quantum computation Framework.",
    license="MIT",
    url="",  #github地址或其他地址
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Environment :: Web Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft',
        'Operating System :: Unix',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
            'torch>=1.9.0',  #所需要包的版本号
            'numpy>=1.14.0'   #所需要包的版本号
    ],
    python_requires=">=3.6",
)
