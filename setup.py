#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Zhu Liang",
    author_email='zhul9311@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A package tool to analyze simulations on Erbium-DEHP system in bulk phase and interface",
    entry_points={
        'console_scripts': [
            'erbium_interface=erbium_interface.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='erbium_interface',
    name='erbium_interface',
    packages=find_packages(include=['erbium_interface', 'erbium_interface.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zhul9311/erbium_interface',
    version='0.1.0',
    zip_safe=False,
)
