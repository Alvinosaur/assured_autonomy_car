"""
setup
"""

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension


setup(
    name="astarlib",
    description="A* search algorithm implemented in Cython",
    python_requires=">=2.7",
    install_requires=[
        "Cython>=0.29.12"
    ],
    packages=find_packages(exclude=["docs", "tests"]),
    ext_modules=[Extension("astarlib", sources=["astarlib.pyx"])],
    include_package_data=True,
    zip_safe=False,
    classifiers=[  # https://pypi.org/classifiers
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
)
