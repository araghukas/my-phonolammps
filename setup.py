from setuptools import setup, find_packages
from my_phonolammps import __version__

setup(
    name="my_phonolammps",
    version=__version__,
    description="My personal version of phonolammps and Python lammps.",
    author="Ara Ghukasyan",
    author_email="ghukasa@mcmaster.ca",
    url="https://github.com/araghukas/my-phonolammps.git",
    license="MIT",
    install_requires=["lammps", "numpy", "phonopy", "seekpath"],
    packages=find_packages()
)
