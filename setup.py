# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BetaBoost",
    version="0.0.4",
    author="Tyler Blume",
    url="https://github.com/tblume1992/BetaBoost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "BetaBoosting: gradient boosting with a beta function.",
    author_email = 'tblume@mail.USF.edu', 
    keywords = ['xgboost', 'gradient boosting', 'data science', 'machine learning'],
      install_requires=[           
                        'xgboost',
                        'pandas',
                        'numpy',
                        'scipy',
                        'matplotlib',
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


