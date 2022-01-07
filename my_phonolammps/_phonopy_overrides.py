"""modified objects from phonopy"""
import h5py
import numpy as np
from typing import Tuple

from phonopy import Phonopy
from phonopy.phonon.band_structure import BandStructure
from phonopy.cui.load import load_helper
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.cells import get_primitive_matrix, Primitive
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import get_default_physical_units
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.units import VaspToTHz

from my_phonolammps.util import _print


class MyDynamicalMatrix(DynamicalMatrix):
    """
    Extend the phonopy DynamicalMatrix class to implement ad-hoc fixes to my problems.

    Added some debugging features, but the functionality remains the same.
    Rewrote _set_force_constants method
    """

    _PRINT = False

    def __init__(self,
                 supercell: PhonopyAtoms,
                 primitive: Primitive,
                 force_constants: np.ndarray,
                 decimals: int = None):
        super().__init__(supercell=supercell,
                         primitive=primitive,
                         force_constants=force_constants,
                         decimals=decimals)

        self._debug_printout_counter = 0

    def __repr__(self):
        return f"""\
MyDynamicalMatrix (debug call {self._debug_printout_counter})
[
self._force_constants.shape={self._force_constants.shape}
self._svecs.shape={self._svecs.shape}
self._multi.shape={self._multi.shape}
len(self._pcell)={len(self._pcell)}
len(self._scell)={len(self._scell)}
len(self._pcell.masses)={len(self._pcell.masses)}
self._pcell.p2s_map.shape={self._pcell.p2s_map.shape}
self._pcell.s2p_map.shape={self._pcell.s2p_map.shape}
]
            """

    def _print_debug(self):
        if not MyDynamicalMatrix._PRINT:
            return
        print(self)
        self._debug_printout_counter += 1

    def _set_force_constants(self, fc: np.ndarray):
        """overriding this to just force a contiguous array every time"""
        self._force_constants = np.ascontiguousarray(fc, dtype=np.double)

    def _run_py_dynamical_matrix(self, q):
        """
        overriding this to try and fix the numpy segfaults issue

        ORIGINAL DOCSTRING
        --------------------------------------------------------------------------------------------
        Python implementation of building dynamical matrix.

        This is not used in production.
        This works only with full-fc.
        """
        self._print_debug()

        fc = self._force_constants
        svecs = self._svecs
        multi = self._multi
        num_atom = len(self._pcell)
        # dm = np.zeros((3 * num_atom, 3 * num_atom), dtype=self._dtype_complex)
        dm = np.zeros((3 * num_atom, 3 * num_atom), dtype=np.complex)
        mass = self._pcell.masses
        if fc.shape[0] == fc.shape[1]:
            is_compact_fc = False
        else:
            is_compact_fc = True

        for i, s_i in enumerate(self._pcell.p2s_map):
            if is_compact_fc:
                fc_elem = fc[i]
            else:
                fc_elem = fc[s_i]
            for j, s_j in enumerate(self._pcell.p2s_map):
                sqrt_mm = np.sqrt(mass[i] * mass[j])
                # dm_local = np.zeros((3, 3), dtype=self._dtype_complex)
                dm_local = np.zeros((3, 3), dtype=np.complex)

                # Sum in lattice points
                for k in range(len(self._scell)):
                    if s_j == self._s2p_map[k]:
                        m, adrs = multi[k][i]
                        svecs_at = svecs[adrs: adrs + m]
                        phase = []
                        for ll in range(m):
                            vec = svecs_at[ll]
                            phase.append(np.vdot(vec, q) * 2j * np.pi)
                        phase_factor = np.exp(phase).sum()
                        dm_local += fc_elem[k] * phase_factor / sqrt_mm / m

                # dm[(i * 3): (i * 3 + 3), (j * 3): (j * 3 + 3)] += dm_local
                dm[(i * 3): (i * 3 + 3), (j * 3): (j * 3 + 3)] = dm_local

        # Impose Hermitian condition
        self._dynamical_matrix = (dm + dm.conj().transpose()) / 2

    def _run_c_dynamical_matrix(self, q):
        self._print_debug()
        super()._run_c_dynamical_matrix(q)

    @staticmethod
    def set_debug_print(b: bool) -> None:
        MyDynamicalMatrix._PRINT = bool(b)


class MyBandStructure(BandStructure):
    """
    Extend the phonopy BandStructure class to implement ad-hoc fixes to my problems.

    This subclass does essentially the same thing, except it has a `use_C_library`
    attribute that tells it to use either the C-extension of pure Python when
    running the dynamical matrix at a given q-point.

    Also, NAC is disabled here.
    """

    def __init__(self,
                 paths,
                 dynamical_matrix: DynamicalMatrix,
                 with_eigenvectors: bool = False,
                 is_band_connection: bool = False,
                 group_velocity: np.ndarray = None,
                 path_connections: list = None,
                 labels: list = None,
                 is_legacy_plot: bool = False,
                 factor: float = VaspToTHz,
                 use_C_library: bool = True):

        self._use_C_library = use_C_library
        super().__init__(
            paths=paths,
            dynamical_matrix=dynamical_matrix,
            with_eigenvectors=with_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=group_velocity,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=is_legacy_plot,
            factor=factor,
        )

    def _solve_dm_on_path(self,
                          path: list) -> Tuple[list, list, list, list]:
        # is_nac = self._dynamical_matrix.is_nac()
        distances_on_path = []
        eigvals_on_path = []
        eigvecs_on_path = []
        gv_on_path = []
        # prev_eigvecs = None

        self._dynamical_matrix: DynamicalMatrix

        if self._group_velocity is not None:
            self._group_velocity.run(path)
            gv = self._group_velocity.group_velocities
        else:
            gv = None

        # this is the main modification here (see use below)
        lang = "C" if self._use_C_library else ""
        # --------------------------------------------------

        for i, q in enumerate(path):
            self._shift_point(q)
            distances_on_path.append(self._distance)

            # if is_nac:
            #     q_direction = None
            #     if (np.abs(q) < 0.0001).all():  # For Gamma point
            #         q_direction = path[0] - path[-1]
            #     self._dynamical_matrix.run(q, q_direction=q_direction)
            # else:
            #     self._dynamical_matrix.run(q)

            self._dynamical_matrix.run(q, lang)
            dm = self._dynamical_matrix.dynamical_matrix

            if self._with_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(dm)
                eigvals = eigvals.real

                eigvals_on_path.append(eigvals)
                eigvecs_on_path.append(eigvecs)
            else:
                eigvals = np.linalg.eigvalsh(dm).real
                eigvals_on_path.append(eigvals)

            # if self._is_band_connection:
            #     if i == 0:
            #         band_order = range(len(eigvals))
            #     else:
            #         band_order = estimate_band_connection(
            #             prev_eigvecs, eigvecs, band_order
            #         )
            #     eigvals_on_path.append(eigvals[band_order])
            #     eigvecs_on_path.append((eigvecs.T)[band_order].T)
            #
            #     if self._group_velocity is not None:
            #         gv_on_path.append(gv[i][band_order])
            #     prev_eigvecs = eigvecs
            # else:
            #     eigvals_on_path.append(eigvals)
            #     if self._with_eigenvectors:
            #         eigvecs_on_path.append(eigvecs)
            #     if self._group_velocity is not None:
            #         gv_on_path.append(gv[i])
            if gv is not None:
                gv_on_path.append(gv[i])

        return distances_on_path, eigvals_on_path, eigvecs_on_path, gv_on_path


class MyPhonopy(Phonopy):
    """
    Extend the Phonopy class to implement ad-hoc fixes to my problems.

    Modified to pass `use_C_library` to MyBandStructure.
    Modified to assign a MyDynamicalMatrix instance in `_set_dynamical_matrix()`
    """

    def run_band_structure(self,
                           paths,
                           with_eigenvectors=False,
                           with_group_velocities=False,
                           path_connections=None,
                           labels=None,
                           is_legacy_plot=False,
                           use_C_library: bool = True):
        """
        Override run band structure to add option for choosing C or Python routine
        for running the dynamical matrix at whatever 'q'.

        ORIGINAL DOCSTRING
        --------------------------------------------------------------------------------------------
        Run phonon band structure calculation.

        Parameters
        ----------
        paths : List of array_like
            Sets of qpoints that can be passed to phonopy.set_band_structure().
            Numbers of qpoints can be different.
            shape of each array_like : (qpoints, 3)
        with_eigenvectors : bool, optional
            Flag whether eigenvectors are calculated or not. Default is False.
        with_group_velocities : bool, optional
            Flag whether group velocities are calculated or not. Default is
            False.
        (DISABLED)is_band_connection : bool, optional
            Flag whether each band is connected or not. This is achieved by
            comparing similarity of eigenvectors of neighboring points. Sometimes
            this fails. Default is False.
        path_connections : List of bool, optional
            This is only used in graphical plot of band structure and gives
            whether each path is connected to the next path or not,
            i.e., if False, there is a jump of q-points. Number of elements is
            the same at that of paths. Default is None.
        labels : List of str, optional
            This is only used in graphical plot of band structure and gives
            labels of end points of each path. The number of labels is equal
            to (2 - np.array(path_connections)).sum().
        is_legacy_plot: bool, optional
            This makes the old style band structure plot. Default is False.
        --------------------------------------------------------------------------------------------
        use_C_library: bool, optional
            The DynamicalMatrix.run() method will be called with lang="C" if this is true,
            otherwise the Python solver will be used.
        """

        # imposing this to simplify things, I don't use it anyway
        is_band_connection = False

        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        self._band_structure = MyBandStructure(
            paths,
            self._dynamical_matrix,
            with_eigenvectors=with_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=group_velocity,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=is_legacy_plot,
            factor=self._factor,
            use_C_library=use_C_library
        )

    def _set_dynamical_matrix(self):
        self._dynamical_matrix = None

        # if self._is_symmetry and self._nac_params is not None:
        #     borns, epsilon = symmetrize_borns_and_epsilon(
        #         self._nac_params["born"],
        #         self._nac_params["dielectric"],
        #         self._primitive,
        #         symprec=self._symprec,
        #     )
        #     nac_params = self._nac_params.copy()
        #     nac_params.update({"born": borns, "dielectric": epsilon})
        # else:
        #     nac_params = self._nac_params

        if self._supercell is None or self._primitive is None:
            raise RuntimeError("Supercell or primitive is not created.")
        if self._force_constants is None:
            raise RuntimeError("Force constants are not prepared.")
        if self._primitive.masses is None:
            raise RuntimeError("Atomic masses are not correctly set.")

        if self._frequency_scale_factor is None:
            force_constants = self._force_constants
        else:
            force_constants = self._force_constants * self._frequency_scale_factor**2

        self._dynamical_matrix = MyDynamicalMatrix(self._supercell,
                                                   self._primitive,
                                                   force_constants,
                                                   decimals=None)

        # DynamicalMatrix instance transforms force constants in correct
        # type of numpy array.
        self._force_constants = self._dynamical_matrix.force_constants

        if self._group_velocity is not None:
            self._set_group_velocity()


def write_hdf5_band_structure(phonon: MyPhonopy,
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


def my_phonopy_load(phonopy_yaml=None,  # phonopy.yaml-like must be the first argument.
                    supercell_matrix=None,
                    primitive_matrix=None,
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
                    is_symmetry=False,  # use to be True
                    symmetrize_fc=False,  # use to be True
                    is_compact_fc=True,
                    store_dense_svecs=False,
                    symprec=1e-5,
                    log_level=0, ) -> MyPhonopy:
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
    (DISABLED)is_nac : bool, optional
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
    is_compact_fc : bool, optional
        Force constants are produced in the array whose shape is
            True: (primitive, supercell, 3, 3)
            False: (supercell, supercell, 3, 3)
        where 'supercell' and 'primitive' indicate number of atoms in these
        cells. Default is True.
    store_dense_svecs : bool, optional
        This is for the test use. Do not set True.
        Default is False.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    log_level : int, optional
        Verbosity control. Default is 0.
    """

    # imposing this to simplify things, I don't use it anyway
    is_nac = False

    if (
            supercell is not None
            or supercell_filename is not None
            or unitcell is not None
            or unitcell_filename is not None
    ):  # noqa E129
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=calculator,
            symprec=symprec,
            log_level=log_level,
        )
        _calculator = calculator
        _nac_params = nac_params
        _dataset = None
        _fc = None
    elif phonopy_yaml is not None:
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(phonopy_yaml)
        cell = phpy_yaml.unitcell
        smat = phpy_yaml.supercell_matrix
        if smat is None:
            smat = np.eye(3, dtype="intc", order="C")
        if primitive_matrix is not None:
            pmat = get_primitive_matrix(primitive_matrix, symprec=symprec)
        else:
            pmat = phpy_yaml.primitive_matrix
        if nac_params is not None:
            _nac_params = nac_params
        elif is_nac:
            _nac_params = phpy_yaml.nac_params
        else:
            _nac_params = None
        _dataset = phpy_yaml.dataset
        _fc = phpy_yaml.force_constants
        if calculator is None:
            _calculator = phpy_yaml.calculator
        else:
            _calculator = calculator
    else:
        msg = "Cell information could not found. " "Phonopy instance loading failed."
        raise RuntimeError(msg)

    if log_level and _calculator is not None:
        print('Set "%s" mode.' % _calculator)

    # units keywords: factor, nac_factor, distance_to_A
    units = get_default_physical_units(_calculator)
    if factor is None:
        _factor = units["factor"]
    else:
        _factor = factor

    phonon = MyPhonopy(  # dropping in my own class here
        cell,
        smat,
        primitive_matrix=pmat,
        factor=_factor,
        frequency_scale_factor=frequency_scale_factor,
        symprec=symprec,
        is_symmetry=is_symmetry,
        store_dense_svecs=store_dense_svecs,
        calculator=_calculator,
        log_level=log_level,
    )

    # NAC params
    if born_filename is not None or _nac_params is not None or is_nac:
        ret_nac_params = load_helper.get_nac_params(
            primitive=phonon.primitive,
            nac_params=_nac_params,
            born_filename=born_filename,
            is_nac=is_nac,
            nac_factor=units["nac_factor"],
            log_level=log_level,
        )
        if ret_nac_params is not None:
            phonon.nac_params = ret_nac_params

    # Displacements, forces, and force constants
    load_helper.set_dataset_and_force_constants(
        phonon,
        _dataset,
        _fc,
        force_constants_filename=force_constants_filename,
        force_sets_filename=force_sets_filename,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
        produce_fc=produce_fc,
        symmetrize_fc=symmetrize_fc,
        is_compact_fc=is_compact_fc,
        log_level=log_level,
    )

    return phonon
