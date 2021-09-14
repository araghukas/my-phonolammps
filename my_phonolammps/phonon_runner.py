"""A wrapper for running generic lammps scripts with injected variables"""
import os
import re
from typing import List, Set, Tuple
from enum import Enum
from dataclasses import dataclass

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import lammps
from phonopy import Phonopy, load as phonopy_load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from my_phonolammps import MyPhonolammps

_ID_MATRIX = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]


class VarType(Enum):
    """enum of LAMMPS variable declaration types"""
    STRING = "string"
    EQUAL = "equal"


class MissingVarError(Exception):
    """raised when insufficient arguments passed for undefined LAMMPS input variables"""


@dataclass(frozen=True)
class LammpsVarLine:
    """
    Represents a LAMMPS command line of the sort:
        variable varname equal 123.321 or variable string file_name "file.txt"

    Convert instances to `str` in order to access these lines.
    """

    name: str
    typ: VarType
    value: str

    def __str__(self):
        """return the complete LAMMPS line"""
        if self.typ is VarType.STRING:
            return f"variable {self.name} {self.typ.value} \"{self.value}\"\n"
        return f"variable {self.name} {self.typ.value} {self.value}\n"


class LammpsRunner:
    """runs generic LAMMPS scripts after making the variable injections"""
    generic_lmp: str
    required_vars: List[LammpsVarLine]

    def __init__(self, generic_lmp, required_vars):
        generic_lmp = os.path.expanduser(generic_lmp)
        if os.path.isfile(generic_lmp):
            self.generic_lmp = generic_lmp
        else:
            raise FileNotFoundError(f"'{generic_lmp}' does not exist")
        self.required_vars = required_vars
        self._validate_provided_vars()
        self._commands_list = None

    @property
    def commands_list(self):
        if self._commands_list is None:
            self._commands_list = self._generate_commands_list()
        return self._commands_list

    def run(self,
            append_commands: List[str] = None,
            skip_lines: List[int] = None,
            skip_keywords: List[str] = None) -> None:
        """inject variables and run the resulting script"""
        if append_commands is None:
            append_commands = []

        lmp = lammps.lammps()
        commands_list = self._get_skipped_commands_list(skip_lines, skip_keywords)
        commands_list += append_commands

        self._commands_list = commands_list
        lmp.commands_list(self._commands_list)
        lmp.close()

    def write_final_script(self, filename: str = None) -> str:
        """write copy of the final script"""
        if filename is None:
            filename = f"in_{id(self)}.lammps"

        with open(filename, 'w') as _file:
            _file.write("".join(self.commands_list))

        return filename

    def _generate_commands_list(self) -> List[str]:
        """generate list of commands to pass to `lammps.lammps.commands_list`"""
        commands_list = [str(v_line) for v_line in self.required_vars]
        with open(self.generic_lmp) as _file:
            commands_list += _file.readlines()
        return commands_list

    def _validate_provided_vars(self) -> None:
        """check that required vars contains all undefined variables"""
        provided_var_names = {v.name for v in self.required_vars}
        undefined_var_names = self._find_undefined_vars()
        var_set_diff = undefined_var_names - provided_var_names
        if len(var_set_diff) > 0:
            msg = "missing variables: "
            for var_name in var_set_diff:
                msg += f"'{var_name}', "
            msg += f"for input file '{self.generic_lmp}'"
            raise MissingVarError(msg)

    def _find_undefined_vars(self) -> Set[str]:
        """find undefined variables in the generic lammps script"""
        with open(self.generic_lmp) as _file:
            ss = "".join(_file.readlines())

        var_decs = re.findall("^variable.*", ss, flags=re.MULTILINE)
        declared = {}
        for var_dec in var_decs:
            name = var_dec.split()[1].strip()
            declared[name] = None

        var_evaluations = []
        for s in ss:
            if not s.startswith('#'):
                var_evals = re.findall(r"(\${\w+}|\$\w+)", s)
                for var_eval in var_evals:
                    var_evaluations.append(var_eval.strip("${}"))

        return {v for v in var_evaluations if v not in declared}

    def _get_skipped_commands_list(self,
                                   skip_lines: List[int],
                                   skip_keywords: List[str]) -> List[str]:
        """selectively delete certain commands from the input file"""
        # first, exclude any lines indexed in `skip_lines`
        raw_commands_list = self.commands_list
        commands_list = []
        skip_lines = [] if skip_lines is None else skip_lines
        for i, command in enumerate(raw_commands_list):
            if i not in skip_lines:
                commands_list.append(command)

        # exclude any lines starting with `skip_keywords` items
        raw_commands_list = commands_list
        commands_list = []
        skip_keywords = [] if skip_keywords is None else skip_keywords
        skip_keywords = [kw + " " for kw in skip_keywords
                         if not (kw.endswith(' ') or kw.endswith('\n'))]

        for command in raw_commands_list:
            match = False
            for kw in skip_keywords:
                if command.startswith(kw):
                    match = True
                    break
            if not match:
                commands_list.append(command)

        return commands_list


class PhononOutputFiles:
    """manages output files of phonon calculations"""
    output_dir: str
    marker: str

    def __init__(self, output_dir, marker: str):

        output_dir = os.path.expanduser(output_dir)
        if os.path.isdir(output_dir):
            self.output_dir = output_dir
        else:
            raise FileNotFoundError(f"invalid output directory '{output_dir}'")

        self.band = self.get_path(f"band_{marker}.hdf5")
        self.dos = self.get_path(f"dos_{marker}.dat")
        self.force_constants_filename = self.get_path(f"FC_{marker}")
        self.harmonic_constants_filename = self.get_path(f"HC_{marker}")
        self.unitcell_filename = self.get_path(f"UC_{marker}")
        self.relaxed_unitcell_filename = self.get_path(f"RELAXED_{marker}.data")
        self.marker = marker

    def clean_up(self) -> None:
        """delete temporary files"""
        os.remove(self.relaxed_unitcell_filename)
        os.remove(self.force_constants_filename)
        os.remove(self.unitcell_filename)

    def get_path(self, filename: str) -> str:
        """create and return a path in the output directory"""
        return os.path.join(self.output_dir, filename)


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

        self.clean_up = clean_up
        self._phonon_path = [[[0., 0., 0.], [0., 0., .5]]]
        self._phonon_mesh = [1, 1, 20]
        self._phonon_npoints = 51
        self._phonon = None
        self._phl = None
        self._trash_counter_max = trash_counter_max

    def __del__(self):
        if self.clean_up:
            self.outputs.clean_up()

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
    def phonon_npoints(self):
        return self._phonon_npoints

    @phonon_npoints.setter
    def phonon_npoints(self, _npoints):
        self._phonon_npoints = int(_npoints)

    @property
    def phonon(self) -> Phonopy:
        return self._phonon

    @property
    def phl(self) -> MyPhonolammps:
        return self._phl

    def run(self) -> None:
        """relax structure, compute forces, and compute phonons"""
        self.relax_structure()
        self.compute_forces()
        self.compute_band_structure()

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
            # TODO: fix these terrible names
            LammpsVarLine("data_file", VarType.STRING, wire_datafile),
            LammpsVarLine("output_file", VarType.STRING, relaxed_datafile),
            LammpsVarLine("potential_file", VarType.STRING, potential_file)
        ]

        input_file = LammpsRunner(self.generic_in,
                                  required_vars).write_final_script()

        self._phl = MyPhonolammps(input_file,
                                  supercell_matrix=_ID_MATRIX,
                                  show_progress=True,
                                  trash_counter_max=self._trash_counter_max)

        self._phl.write_harmonic_constants(hc_filename,
                                           omit_zeros=omit_zeros,
                                           omit_zeros_thresh=omit_zeros_thresh)

        # delete substituted LAMMPS input file
        os.remove(input_file)

    def compute_band_structure(self) -> None:
        """use phonopy to compute band structure and total DOS"""

        # aliasing for neatness
        unitcell = self.outputs.unitcell_filename
        force_constants = self.outputs.force_constants_filename
        band = self.outputs.band
        dos = self.outputs.dos

        self._phonon = phonopy_load(supercell_matrix=_ID_MATRIX,
                                    unitcell_filename=unitcell,
                                    force_constants_filename=force_constants)
        self._phonon.run_mesh(self._phonon_mesh)

        qs, cons = get_band_qpoints_and_path_connections(band_paths=self._phonon_path,
                                                         npoints=self._phonon_npoints)
        self._phonon.run_band_structure(qs,
                                        path_connections=cons,
                                        with_eigenvectors=True)
        self._phonon.write_hdf5_band_structure(filename=band)

        self._phonon.run_total_dos()
        self._phonon.write_total_dos(filename=dos)

    def plot_bands(self) -> Tuple[Figure, Axes]:
        """run the default phonopy `plot_band_structure` method"""
        plt = self._phonon.plot_band_structure()
        return plt.gcf(), plt.gca()
