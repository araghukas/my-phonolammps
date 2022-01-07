"""A wrapper for running generic lammps scripts with injected variables"""
import os
from typing import Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from my_phonolammps._phonopy_overrides import (write_hdf5_band_structure,
                                               my_phonopy_load,
                                               MyPhonopy)
from my_phonolammps._phonolammps import MyPhonolammps

from my_phonolammps.util import _print
from my_phonolammps.phonon_outputs import PhononOutputFiles
from my_phonolammps.lammps_runner import LammpsRunner, LammpsVarLine, VarType

_ID_MATRIX = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]


class PhononRunner:
    """
    Runs phonolammps and phonopy calculations on a given nanowire structure
    and calculates the Γ->Χ phonon dispersion along the wire axis by default.
    Adjust `phonon_path` and `phonon_mesh` accordingly to make changes.

    The main output is an HDF5 file (default "band.hdf5").
    With the default settings, this can easily be several GB in size for a
    large (100-1000 atom) unit cell.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, wire_datafile: str, potential_file: str,
                 output_dir='.',
                 generic_relax='./generic_relax.lammps',
                 generic_in='./generic_in.lammps',
                 marker=None,
                 clean_up=False,
                 trash_counter_max=50):

        self.clean_up = clean_up
        self._trash_counter_max = trash_counter_max

        # nothing works if these don't exist
        generic_relax = os.path.expanduser(generic_relax)
        if not os.path.isfile(generic_relax):
            raise FileNotFoundError(f"file not found '{generic_relax}'")
        self.generic_relax = generic_relax

        generic_in = os.path.expanduser(generic_in)
        if not os.path.isfile(generic_in):
            raise FileNotFoundError(f"file not found '{generic_in}'")
        self.generic_in = generic_in

        potential_file = os.path.expanduser(potential_file)
        if not os.path.isfile(potential_file):
            raise FileNotFoundError(f"file not found '{potential_file}'")
        self.potential_file = potential_file

        # validate paths before assigning
        wire_datafile = os.path.expanduser(wire_datafile)
        if os.path.isfile(wire_datafile):
            self.wire_datafile = wire_datafile
        else:
            raise FileNotFoundError(f"file not found '{wire_datafile}'")
        self.outputs = PhononOutputFiles(output_dir, marker=marker)

        self._phonon_path = [[[0., 0., 0.], [0., 0., .5]]]
        self._phonon_mesh = [1, 1, 20]
        self._nqpoints = 51
        self._phonon = None
        self._phl = None

    def __repr__(self):
        return f"""\
PhononRunner(wire_datafile='{self.wire_datafile}',
             potential_file='{self.potential_file}',
             output_dir='{self.outputs.output_dir}',
             generic_relax='{self.generic_relax}',
             generic_in='{self.generic_in}',
             marker='{self.outputs.marker}',
             clean_up={self.clean_up},
             trash_counter_max={self._trash_counter_max})
             
             self.phonon_path={self.phonon_path if self.phonon_path else None}
             self.phonon_mesh={self.phonon_mesh if self.phonon_mesh else None}
             self.nqpoints={self.nqpoints if self.nqpoints else None}
             self.phl={self._phl if self._phl else None}
             self.outputs={self.outputs}
            """

    def __del__(self):
        if self.clean_up:
            self.outputs.clean_up()

    @property
    def phonon(self) -> MyPhonopy:
        if self._phonon is None:
            _print("loading phonon")
            load_kwargs = dict(supercell_matrix=_ID_MATRIX,
                               unitcell_filename=self.outputs.unitcell_filename,
                               force_constants_filename=self.outputs.force_constants_filename)
            self._phonon = my_phonopy_load(**load_kwargs)
        return self._phonon

    @phonon.setter
    def phonon(self, ph: Phonopy):
        if isinstance(ph, Phonopy):
            self._phonon = ph

    @property
    def phonon_path(self):
        return self._phonon_path

    @phonon_path.setter
    def phonon_path(self, _path):
        self._phonon_path = _path

    @property
    def phonon_mesh(self):
        return self._phonon_mesh

    @phonon_mesh.setter
    def phonon_mesh(self, _mesh):
        if len(_mesh) != 3:
            raise ValueError("mesh should be a 3-vector")
        self._phonon_mesh = [int(n) for n in _mesh]

    @property
    def nqpoints(self):
        return self._nqpoints

    @nqpoints.setter
    def nqpoints(self, _npoints):
        self._nqpoints = int(_npoints)

    @property
    def phl(self) -> MyPhonolammps:
        return self._phl

    def run(self,
            omit_zeros: bool = True,
            omit_zeros_thresh: float = 1e-6,
            with_eigenvectors: bool = True,
            with_group_velocities: bool = True) -> None:
        """relax structure, compute forces, and compute phonons"""
        self.relax_structure()
        self.compute_forces(omit_zeros=omit_zeros, omit_zeros_thresh=omit_zeros_thresh)
        self.compute_band_structure(with_eigenvectors=with_eigenvectors,
                                    with_group_velocities=with_group_velocities)

    def relax_structure(self, dummy: bool = False) -> None:
        """
        use LAMMPS to relax input structure
        (minimize potential energy at 0 pressure)
        """

        # aliasing for neatness
        wire = self.wire_datafile
        wire_relaxed = self.outputs.relaxed_unitcell_filename  # this file is written
        potential_file = self.potential_file

        required_vars = [
            LammpsVarLine("data_file", VarType.STRING, wire),
            LammpsVarLine("output_file", VarType.STRING, wire_relaxed),
            LammpsVarLine("potential_file", VarType.STRING, potential_file)
        ]
        lmp_runner = LammpsRunner(self.generic_relax, required_vars)
        append_commands = [
            'reset_timestep 0',
            f'dump RELAXED all atom 1 {self.outputs.relaxed_unitcell_filename}',
            'dump_modify RELAXED sort id '
            'format line "%d %d %20.15g %20.15g %20.15g"',  # IMPORTANT
            'run 0',
            'undump RELAXED'
        ]
        if dummy:
            lmp_runner.run(append_commands=append_commands,
                           skip_keywords=["minimize",
                                          "compute",
                                          "thermo_style"])
        else:
            lmp_runner.run(append_commands=append_commands)

    def compute_forces(self,
                       relaxed: bool = True,
                       omit_zeros: bool = True,
                       omit_zeros_thresh: float = 1e-6) -> None:
        """use phonolammps to compute and write the force constants"""

        # aliasing for neatness
        wire_datafile = self.wire_datafile
        relaxed_datafile = self.outputs.relaxed_unitcell_filename
        unitcell = self.outputs.unitcell_filename
        force_constants = self.outputs.force_constants_filename
        potential_file = self.potential_file

        if not relaxed:
            # re-write 'relaxed' data-file without doing minimization
            self.relax_structure(dummy=True)

        required_vars = [
            # TODO: fix these terrible names
            LammpsVarLine("data_file", VarType.STRING, wire_datafile),
            LammpsVarLine("output_file", VarType.STRING, relaxed_datafile),
            LammpsVarLine("potential_file", VarType.STRING, potential_file)
        ]

        input_file = LammpsRunner(self.generic_in,
                                  required_vars).write_final_script()

        if self._phl is None:
            self._phl = MyPhonolammps(input_file,
                                      supercell_matrix=_ID_MATRIX,
                                      show_progress=True,
                                      trash_counter_max=self._trash_counter_max)

        self._phl.write_unitcell_POSCAR(unitcell)
        self._phl.write_force_constants(force_constants,
                                        omit_zeros=omit_zeros,
                                        omit_zeros_thresh=omit_zeros_thresh)

        # delete substituted LAMMPS input file
        os.remove(input_file)

    def compute_harmonic_constants(self,
                                   relaxed: bool = True,
                                   omit_zeros: bool = True,
                                   omit_zeros_thresh: float = 1e-6) -> None:
        """use phonolammps to compute and write the force constants"""

        # aliasing for neatness
        wire_datafile = self.wire_datafile
        relaxed_datafile = self.outputs.relaxed_unitcell_filename
        hc_filename = self.outputs.harmonic_constants_filename
        potential_file = self.potential_file

        if not relaxed:
            # re-write 'relaxed' data-file without doing minimization
            self.relax_structure(dummy=True)

        required_vars = [
            LammpsVarLine("data_file", VarType.STRING, wire_datafile),
            LammpsVarLine("output_file", VarType.STRING, relaxed_datafile),
            LammpsVarLine("potential_file", VarType.STRING, potential_file)
        ]

        input_file = LammpsRunner(self.generic_in,
                                  required_vars).write_final_script()

        if self._phl is None:
            self._phl = MyPhonolammps(input_file,
                                      supercell_matrix=_ID_MATRIX,
                                      show_progress=True,
                                      trash_counter_max=self._trash_counter_max)

        self._phl.write_harmonic_constants(hc_filename,
                                           omit_zeros=omit_zeros,
                                           omit_zeros_thresh=omit_zeros_thresh)

        # delete substituted LAMMPS input file
        os.remove(input_file)

    def run_mesh(self,
                 with_eigenvectors: bool = True,
                 with_group_velocities: bool = True) -> None:
        """run the mesh on the phonon attribute"""
        _print("running mesh")
        self.phonon.run_mesh(self._phonon_mesh,
                             with_eigenvectors=with_eigenvectors,
                             with_group_velocities=with_group_velocities)

    def compute_band_structure(self,
                               with_eigenvectors: bool = True,
                               with_group_velocities: bool = True,
                               use_C_library: bool = True) -> None:
        """use phonopy to compute band structure"""
        if self.phonon.mesh is None:
            self.run_mesh(with_eigenvectors, with_group_velocities)

        _print("getting qpoints and connections")
        qs, cons = get_band_qpoints_and_path_connections(band_paths=self._phonon_path,
                                                         npoints=self._nqpoints)
        _print("running band structure")
        self.phonon.run_band_structure(qs,
                                       path_connections=cons,
                                       with_eigenvectors=with_eigenvectors,
                                       with_group_velocities=with_group_velocities,
                                       use_C_library=use_C_library)

        # use alternate function defined below
        _print("writing band structure")
        write_hdf5_band_structure(self.phonon, paths=qs, filename=self.outputs.band)

    def compute_total_dos(self,
                          with_eigenvectors: bool = True,
                          with_group_velocities: bool = True):
        """use phonopy to compute the total DOS"""
        if self.phonon.mesh is None:
            self.run_mesh(with_eigenvectors, with_group_velocities)

        _print("running total dos")
        self.phonon.run_total_dos()

        _print("writing total dos")
        self.phonon.write_total_dos(filename=self.outputs.dos)

    def plot_bands(self) -> Tuple[Figure, Axes]:
        """run the default phonopy `plot_band_structure` method"""
        plt = self.phonon.plot_band_structure()
        return plt.gcf(), plt.gca()
