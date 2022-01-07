"""Modified version of phonolammps for my personal use in nanowire simulations"""
from os.path import expanduser, abspath
from my_phonolammps._lammps import MyLammps
from my_phonolammps._phonolammps import MyPhonolammps
from my_phonolammps.phonon_runner import PhononRunner


def set_lammps_modpath(modpath: str) -> None:
    """
    Set the LAMMPS wrapper's `modpath` attribute to direct the
    Python wrapper to liblammps.so or liblammps.dylib.
    """
    MyLammps.MODPATH = abspath(expanduser(modpath))


__version__ = "2021.5.4_debug7"
