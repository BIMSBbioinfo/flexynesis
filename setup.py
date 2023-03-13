import os
from setuptools import setup, find_packages

setup(
    name='flexynesis',
    version='0.1',
    author="Bora Uyar",
    author_email="bora.uyar@mdc-berlin.de",
    packages=find_packages(),
    install_requires=[
        'ray',
	'torch',
	'torchvision',
	'tqdm',
	'pytorch-lightning'
    ],
    entry_points={
        'console_scripts': [
            'flexynesis=flexynesis.__main__:main'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
    ],
)

