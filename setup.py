#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="house_prices_m05_pkg",
    version="1.0.1",
    description="house prices prediction tool",
    url="https://github.com/CherubiniArt/mini_project_m05",
    license="BSD",
    author="Arthur Cherubini & Michelle Meguep",
    author_email="cherubini.art@gmail.com",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["run-house-prices-pred = house_prices_m05_pkg.toolchain_all_params:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
