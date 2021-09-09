from setuptools import setup, find_packages
from my_phonolammps import __version__

setup(
    name="my_phonolammps",
    version=__version__,
    description="A package that uses LAMMPS to compute phonon quantities.",
    author="Ara Ghukasyan",
    install_requires=["numpy", "phonopy", "seekpath"],
    packages=find_packages()
)
