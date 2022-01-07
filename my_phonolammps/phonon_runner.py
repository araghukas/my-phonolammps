"""A wrapper for running generic lammps scripts with injected variables"""
import os
import re
import h5py
import numpy as np
from typing import List, Set, Tuple
from enum import Enum
from dataclasses import dataclass
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import phonopy
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections, BandStructure

from my_phonolammps._lammps import MyLammps
from my_phonolammps._phonolammps import MyPhonolammps

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

        lmp = MyLammps()
        commands_list = self._get_skipped_commands_list(skip_lines, skip_keywords)
        commands_list += append_commands

        self._commands_list = commands_list
        lmp.commands_list(self._commands_list)
        lmp.close()

    def write_final_script(self, filename: str = None) -> str:
        """write copy of the final script"""
        if filename is None:
            filename = f"in_{MyPhonolammps.get_random_id_string(12)}.lammps"

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

    def __repr__(self):
        return f"""\
PhononOutputFiles(output_dir='{self.output_dir}',
                  marker='{self.marker}')
                  
                  self.band='{self.band}'
                  self.dos='{self.dos}'
                  self.force_constants_filename='{self.force_constants_filename}'
                  self.harmonic_constants_filename='{self.harmonic_constants_filename}'
                  self.unitcell_filename='{self.unitcell_filename}'
                  self.relaxed_unitcell_filename='{self.relaxed_unitcell_filename}'
                """

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
    def phonon(self) -> Phonopy:
        return self._phonon

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

    def compute_band_structure(self,
                               with_eigenvectors: bool = True,
                               with_group_velocities: bool = True) -> None:
        """use phonopy to compute band structure and total DOS"""

        # aliasing for neatness
        unitcell = self.outputs.unitcell_filename
        force_constants = self.outputs.force_constants_filename
        band = self.outputs.band
        dos = self.outputs.dos
        # ---------------------

        _print("loading phonon")
        load_kwargs = dict(supercell_matrix=_ID_MATRIX,
                           unitcell_filename=unitcell,
                           force_constants_filename=force_constants)
        self._phonon = phonopy_load(**load_kwargs)
        _print("running mesh")
        self._phonon.run_mesh(self._phonon_mesh)

        _print("getting qpoints and connections")
        qs, cons = get_band_qpoints_and_path_connections(band_paths=self._phonon_path,
                                                         npoints=self._nqpoints)
        _print("running band structure")
        self._phonon.run_band_structure(qs,
                                        path_connections=cons,
                                        with_eigenvectors=with_eigenvectors,
                                        with_group_velocities=with_group_velocities)

        # use alternate function defined below
        _print("writing band structure")
        write_hdf5_band_structure(self._phonon, paths=qs, filename=band)

        _print("running total dos")
        self._phonon.run_total_dos()

        _print("writing total dos")
        self._phonon.write_total_dos(filename=dos)

    def plot_bands(self) -> Tuple[Figure, Axes]:
        """run the default phonopy `plot_band_structure` method"""
        plt = self._phonon.plot_band_structure()
        return plt.gcf(), plt.gca()


def phonopy_load(phonopy_yaml=None,
                 supercell_matrix=None,
                 primitive_matrix=None,
                 is_nac=False,  # used to be True
                 calculator=None,
                 unitcell=None,
                 supercell=None,
                 nac_params=None,
                 unitcell_filename=None,
                 supercell_filename=None,
                 born_filename=None,
                 force_sets_filename=None,
                 force_constants_filename=None,
                 fc_calculator=None,
                 fc_calculator_options=None,
                 factor=None,
                 frequency_scale_factor=None,
                 produce_fc=True,
                 is_symmetry=True,
                 symmetrize_fc=True,
                 is_compact_fc=True,
                 symprec=1e-5,
                 log_level=0) -> Phonopy:
    """
    alias for phonopy.cui.load.load, where I've changed some defaults

    ORIGINAL DOCSTRING
    ------------------------------------------------------------------------------------------------
    Create Phonopy instance from parameters and/or input files.

    "phonopy_yaml"-like file is parsed unless crystal structure information
    is given by unitcell_filename, supercell_filename, unitcell
    (PhonopyAtoms-like), or supercell (PhonopyAtoms-like).
    Even when "phonopy_yaml"-like file is parse, parameters except for
    crystal structure can be overwritten.

    Phonopy default files of 'FORCE_SETS' and 'BORN' are parsed when they
    are found in current directory and those data are not yet provided by
    other means.

    Crystal structure
    -----------------
    Means to provide crystal structure(s) and their priority:
        1. unitcell_filename (with supercell_matrix)
        2. supercell_filename
        3. unitcell (with supercell_matrix)
        4. supercell.
        5. phonopy_yaml

    Force sets or force constants
    -----------------------------
    Optional. Means to provide information to generate force constants
    and their priority:
        1. force_constants_filename
        2. force_sets_filename
        3. phonopy_yaml if force constants are found in phonopy_yaml.
        4. phonopy_yaml if forces are found in phonopy_yaml.dataset.
        5. 'FORCE_CONSTANTS' is searched in current directory.
        6. 'force_constants.hdf5' is searched in current directory.
        7. 'FORCE_SETS' is searched in current directory.
    When both of 3 and 4 are satisfied but not others, force constants and
    dataset are stored in Phonopy instance, but force constants are not
    produced from dataset.

    Parameters for non-analytical term correction (NAC)
    ----------------------------------------------------
    Optional. Means to provide NAC parameters and their priority:
        1. born_filename
        2. nac_params
        3. phonopy_yaml.nac_params if existed and is_nac=True.
        4. 'BORN' is searched in current directory when is_nac=True.

    Parameters
    ----------
    phonopy_yaml : str, optional
        Filename of "phonopy.yaml"-like file. If this is given, the data
        in the file are parsed. Default is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to input cell basis vectors.
        shape=(3, ) or (3, 3), where the former is considered a diagonal
        matrix. Default is the unit matrix.
        dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is
        None, which is equivalent to 'auto'.
        For array_like, shape=(3, 3), dtype=float.
        When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix,
        the primitive matrix for the character found at
        https://spglib.github.io/spglib/definition.html
        is used.
    is_nac : bool, optional
        If True, look for 'BORN' file. If False, NAS is turned off.
        Default is True.
    calculator : str, optional.
        Calculator used for computing forces. This is used to switch the set
        of physical units. Default is None, which is equivalent to "vasp".
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None.
    supercell : PhonopyAtoms, optional
        Input supercell. With given, default value of primitive_matrix is set
        to 'auto' (can be overwritten). supercell_matrix is ignored. Default is
        None.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default is
        None.
        {'born': Born effective charges
                 (array_like, shape=(primitive cell atoms, 3, 3), dtype=float),
         'dielectric': Dielectric constant matrix
                       (array_like, shape=(3, 3), dtype=float),
         'factor': unit conversion factor (float)}
    unitcell_filename : str, optional
        Input unit cell filename. Default is None.
    supercell_filename : str, optional
        Input supercell filename. When this is specified, supercell_matrix is
        ignored. Default is None.
    born_filename : str, optional
        Filename corresponding to 'BORN', a file contains non-analytical term
        correction parameters.
    force_sets_filename : str, optional
        Filename of a file corresponding to 'FORCE_SETS', a file contains sets
        of forces and displacements. Default is None.
    force_constants_filename : str, optional
        Filename of a file corresponding to 'FORCE_CONSTANTS' or
        'force_constants.hdf5', a file contains force constants. Default is
        None.
    fc_calculator : str, optional
        Force constants calculator. Currently only 'alm'. Default is None.
    fc_calculator_options : str, optional
        Optional parameters that are passed to the external fc-calculator.
        This is given as one text string. How to parse this depends on the
        fc-calculator. For alm, each parameter is splitted by comma ',',
        and each set of key and value pair is written in 'key = value'.
    factor : float, optional
        Phonon frequency unit conversion factor. Unless specified, default
        unit conversion factor for each calculator is used.
    frequency_scale_factor : float, optional
        Factor multiplied to calculated phonon frequency. Default is None,
        i.e., effectively 1.
    produce_fc : bool, optional
        Setting False, force constants are not calculated from displacements
        and forces. Default is True.
    is_symmetry : bool, optional
        Setting False, crystal symmetry except for lattice translation is not
        considered. Default is True.
    symmetrize_fc : bool, optional
        Setting False, force constants are not symmetrized when creating
        force constants from displacements and forces. Default is True.
    is_compact_fc : bool
        Force constants are produced in the array whose shape is
            True: (primitive, supercell, 3, 3)
            False: (supercell, supercell, 3, 3)
        where 'supercell' and 'primitive' indicate number of atoms in these
        cells. Default is True.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    log_level : int, optional
        Verbosity control. Default is 0.

    """
    return phonopy.load(phonopy_yaml=phonopy_yaml,  # phonopy.yaml-like must be the first argument.
                        supercell_matrix=supercell_matrix,
                        primitive_matrix=primitive_matrix,
                        is_nac=is_nac,
                        calculator=calculator,
                        unitcell=unitcell,
                        supercell=supercell,
                        nac_params=nac_params,
                        unitcell_filename=unitcell_filename,
                        supercell_filename=supercell_filename,
                        born_filename=born_filename,
                        force_sets_filename=force_sets_filename,
                        force_constants_filename=force_constants_filename,
                        fc_calculator=fc_calculator,
                        fc_calculator_options=fc_calculator_options,
                        factor=factor,
                        frequency_scale_factor=frequency_scale_factor,
                        produce_fc=produce_fc,
                        is_symmetry=is_symmetry,
                        symmetrize_fc=symmetrize_fc,
                        is_compact_fc=is_compact_fc,
                        symprec=symprec,
                        log_level=log_level)


def write_hdf5_band_structure(phonon: Phonopy,
                              paths: list,
                              filename: str,
                              comment: dict = None) -> None:
    """
    This is meant to replace the BandStructure method of the same name,
    because the original crashes with a segfault on my larger cells.

    Below is the method that does the heavy lifting:

        `phonopy.band_structure.BandStructure.write_hdf5()`

    --------------------------------------------------------------------------------------------
    \"\"\"Write band structure in hdf5 format.\"\"\"
    import h5py
    with h5py.File(filename, 'w') as w:
        w.create_dataset('path', data=self._paths)
        w.create_dataset('distance', data=self._distances)
        w.create_dataset('frequency', data=self._frequencies)
        if self._eigenvectors is not None:
            w.create_dataset('eigenvector', data=self._eigenvectors)
        if self._group_velocities is not None:
            w.create_dataset('group_velocity', data=self._group_velocities)
        if comment:
            for key in comment:
                if key not in ('path',
                               'distance',
                               'frequency',
                               'eigenvector',
                               'group_velocity'):
                    w.create_dataset(key, data=np.string_(comment[key]))

        path_labels = []
        if self._labels:
            if self._is_legacy_plot:
                for i in range(len(self._paths)):
                    path_labels.append([np.string_(self._labels[i]),
                                        np.string_(self._labels[i + 1])])
            else:
                i = 0
                for c in self._path_connections:
                    path_labels.append([np.string_(self._labels[i]),
                                        np.string_(self._labels[i + 1])])
                    if c:
                        i += 1
                    else:
                        i += 2
        w.create_dataset('label', data=path_labels)

        nq_paths = []
        for qpoints in self._paths:
            nq_paths.append(len(qpoints))
        w.create_dataset('nqpoint', data=[np.sum(nq_paths)])
        w.create_dataset('segment_nqpoint', data=nq_paths)
    --------------------------------------------------------------------------------------------

    [X] maybe importing h5py globally will solve the issue, so we can keep it basically the same
    [?] some ridiculous debugging printouts
    """
    np.set_printoptions(precision=1, linewidth=300)

    bs: BandStructure = phonon.band_structure

    _print("paths: {}".format(paths))
    _print("distance: {}".format(bs.distances))
    _print("frequencies: {}".format(bs.frequencies))
    _print("eigvs: {}".format(bs.eigenvectors))
    _print("velocities: {}".format(bs.group_velocities))

    _print("opening file")
    with h5py.File(filename, 'w') as w:

        _print("writing paths")
        w.create_dataset('path', data=paths)

        _print("writing distances")
        w.create_dataset('distance', data=bs.distances)

        _print("writing frequencies")
        w.create_dataset('frequency', data=bs.frequencies)
        if bs.eigenvectors is not None:
            _print("writing eigenvectors")
            w.create_dataset('eigenvector', data=bs.eigenvectors)
        if bs.group_velocities is not None:
            _print("writing velocities")
            w.create_dataset('group_velocity', data=bs.group_velocities)
        if comment:
            for key in comment:
                if key not in ('path',
                               'distance',
                               'frequency',
                               'eigenvector',
                               'group_velocity'):
                    w.create_dataset(key, data=np.string_(comment[key]))

        path_labels = []
        if bs.labels:
            if bs.is_legacy_plot:
                for i in range(len(paths)):
                    path_labels.append([np.string_(bs.labels[i]),
                                        np.string_(bs.labels[i + 1])])
            else:
                i = 0
                for c in bs.path_connections:
                    path_labels.append([np.string_(bs.labels[i]),
                                        np.string_(bs.labels[i + 1])])
                    if c:
                        i += 1
                    else:
                        i += 2
        _print("writing labels")
        w.create_dataset('label', data=path_labels)

        nq_paths = []
        for qpoints in paths:
            nq_paths.append(len(qpoints))

        _print("writing qpoints")
        w.create_dataset('nqpoint', data=[np.sum(nq_paths)])

        _print("writing segment_nqpoint")
        w.create_dataset('segment_nqpoint', data=nq_paths)


def _print(s: str) -> None:
    s_ = "\n\n---> my_phonolammps-debug\n"
    print(s_ + s + "\n\n")
