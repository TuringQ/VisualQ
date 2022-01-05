from __future__ import print_function
from setuptools import setup, find_packages
import sys
filepath = 'README.md'

setup(
    name="deepquantum",
    version="0.0.2",
    author="TuringQ",  #作者名字
    author_email="",
    description="IR for quantum computation.",
    long_description=open(filepath, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
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
            'torch>=1.10.0',  #所需要包的版本号
            'numpy>=1.14.0',   #所需要包的版本号
            'scipy>=1.7.3',
            'typing>=3.5.0',
    ],
    python_requires=">=3.6",
)
