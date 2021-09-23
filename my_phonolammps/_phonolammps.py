"""
Module of objects connecting core phonopy functionality
"""
import os
import warnings
import numpy as np
from abc import ABC, abstractmethod
from math import sqrt

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import parse_BORN
from phonopy.file_IO import (write_FORCE_CONSTANTS,
                             write_force_constants_to_hdf5,
                             write_FORCE_SETS)

from my_phonolammps._lammps import MyLammps

# define the force unit conversion factors to LAMMPS metal style (eV/Angstrom)
unit_factors = {
    'real': 4.336410389526464e-2,
    'metal': 1.0,
    'si': 624150636.3094,
    'tinker': 0.043  # kcal/mol to eV
}


def mass_to_symbol(mass, tolerance=5e-1):
    from phonopy.structure.atoms import atom_data

    for element in atom_data:
        if element[3] is not None and abs(mass - element[3]) < tolerance:
            return element[1]

    return 'H'  # in case no match found use H as wildcard


def get_phonon(structure,
               NAC=False,
               setup_forces=True,
               super_cell_phonon=np.identity(3),
               primitive_matrix=np.identity(3),
               symmetrize=True):
    """
    Return a phonopy phonon object (instance of the class Phonon)

    :param structure: unit cell matrix (lattice vectors in rows)
    :param NAC: (Bool) activate/deactivate Non-analytic corrections
    :param setup_forces: (Bool) decide if pre-calculate harmonic forces in phonon object
    :param super_cell_phonon: 3x3 array containing the supercell to be used to calculate the force constants
    :param primitive_matrix: 3x3 array containing the primitive axis (in rows) which define the primitive cell
    :param symmetrize: decide if symmetrize the force constants
    :return: phonopy phonon object
    """

    phonon = Phonopy(structure, super_cell_phonon,
                     symprec=1e-5,
                     primitive_matrix=primitive_matrix,
                     is_symmetry=symmetrize)

    # Non Analytical Corrections (NAC) from Phonopy [Frequencies only, eigenvectors no affected by this option]

    if setup_forces:
        if structure.get_force_constants() is not None:
            phonon.force_constants = structure.get_force_constants().get_array()
            phonon.force_constants = structure.get_force_constants().get_array()
        elif structure.get_force_sets() is not None:
            phonon.dataset = structure.get_force_sets().get_dict()
            phonon.produce_force_constants()
            structure.set_force_constants(ForceConstants(phonon.force_constants,
                                                         supercell=structure.get_force_sets().get_supercell()))
        else:
            print('No force sets/constants available!')
            exit()

    if NAC:
        print("Using non-analytical corrections")
        primitive = phonon.primitive
        try:
            nac_params = parse_BORN(primitive)
            phonon.nac_params = nac_params
        except OSError:
            print('Required BORN file not found!')
            exit()

    return phonon


def obtain_phonon_dispersion_bands(structure, bands_ranges, force_constants, supercell,
                                   NAC=False, band_resolution=30, band_connection=False,
                                   primitive_matrix=np.identity(3)):
    """
    Get the phonon dispersion bands in phonopy format

    :param structure: unit cell matrix (lattice vectors in rows)
    :param bands_ranges: define the path in the reciprocal space (phonopy format)
    :param force_constants: force constants array ( in phonopy format)
    :param supercell: 3x3 array containing the supercell to be used to calculate the force constants
    :param NAC: (Bool) activate/deactivate Non-analytic corrections
    :param band_resolution: define number of points in path in the reciprocal space
    :param band_connection: decide if bands will be all connected or in segments
    :param primitive_matrix: 3x3 array containing the primitive axis (in rows) which define the primitive cell
    :return:
    """
    phonon = get_phonon(structure, NAC=NAC, setup_forces=False,
                        super_cell_phonon=supercell,
                        primitive_matrix=primitive_matrix)

    phonon.force_constants = force_constants

    bands = []
    for q_start, q_end in bands_ranges:
        band = []
        for i in range(band_resolution + 1):
            band.append(
                np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    try:
        phonon.run_band_structure(bands, is_band_connection=band_connection, with_eigenvectors=True)
        bands_dict = phonon.get_band_structure_dict()
        bands_phonopy = (bands_dict['qpoints'],
                         bands_dict['distances'],
                         bands_dict['frequencies'],
                         bands_dict['eigenvectors'])

    except AttributeError:
        # phonopy 1.9.x+ support
        phonon.run_band_structure(bands, is_band_connection=band_connection, with_eigenvectors=True)
        bands_phonopy = phonon.get_band_structure_dict()

    return bands_phonopy


def generate_VASP_structure(structure, scaled=True):
    cell = structure.get_cell()
    types = structure.get_chemical_symbols()

    elements = [types[0]]
    elements_count = [1]

    for t in types[1:]:
        if t == elements[-1]:
            elements_count[-1] += 1
        else:
            elements.append(t)
            elements_count.append(1)

    # atom_type_unique = np.unique(types, return_index=True)

    # To use unique without sorting
    # sort_index = np.argsort(atom_type_unique[1])
    # elements = np.array(atom_type_unique[0])[sort_index]

    # elements_count= np.diff(np.append(np.array(atom_type_unique[1])[sort_index], [len(types)]))
    # print(elements_count)

    vasp_POSCAR = 'Generated using phonoLAMMPS\n'
    vasp_POSCAR += '1.0\n'
    for row in cell:
        vasp_POSCAR += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*row)
    vasp_POSCAR += ' '.join(elements)
    vasp_POSCAR += ' \n'
    vasp_POSCAR += ' '.join([str(i) for i in elements_count])

    if scaled:
        scaled_positions = structure.get_scaled_positions()
        vasp_POSCAR += '\nDirect\n'
        for row in scaled_positions:
            vasp_POSCAR += '{0:15.15f}   {1:15.15f}   {2:15.15f}\n'.format(*row)

    else:
        positions = structure.get_positions()
        vasp_POSCAR += '\nCartesian\n'
        for row in positions:
            vasp_POSCAR += '{0:20.10f} {1:20.10f} {2:20.10f}\n'.format(*row)

    return vasp_POSCAR


def read_structure_params_from_dump(dump_file: str, na: int):
    ids, masses, symbols = [], [], []
    positions = np.zeros((na, 3), dtype='double')
    with open(dump_file) as f:
        line = f.readline()
        while line and not line.startswith("ITEM: ATOMS"):
            line = f.readline()
        if not line:
            raise ValueError("could not find 'ITEM: ATOMS' section in ", dump_file)

        line = f.readline()
        i = 0
        while i < na:
            line = line.strip()
            nums = line.split()
            atom_id = int(nums[0])
            mass = float(nums[1])
            symbol = mass_to_symbol(mass)
            position = [float(s) for s in nums[2:5]]

            ids.append(atom_id)
            masses.append(mass)
            symbols.append(symbol)
            positions[i] = position
            line = f.readline()
            i += 1

    os.remove(dump_file)
    return ids, masses, symbols, positions


def read_forces_from_dump(dump_file: str):
    with open(dump_file) as f:
        line = f.readline()
        while line and not line.startswith("ITEM: NUMBER OF ATOMS"):
            line = f.readline()
        if not line:
            raise ValueError("could not find 'ITEM: NUMBER OF ATOMS' section in ", dump_file)

        n_atoms = int(f.readline().strip())
        forces = np.zeros((n_atoms, 3), dtype='double')

        line = f.readline()
        while line and not line.startswith("ITEM: ATOMS"):
            line = f.readline()
        if not line:
            raise ValueError("could not find 'ITEM: ATOMS' section in ", dump_file)

        line = f.readline()
        i = 0
        while i < n_atoms:
            forces[i] = [float(s) for s in line.strip().split()]
            line = f.readline()
            i += 1

    return forces


class MyPhonoBase(ABC):
    """
    [My version of the..]
    Base class for PhonoLAMMPS
    This class is not designed to be called directly.
    To use it make a subclass and implement the following methods:

    * get_forces()

    """

    def __init__(self):
        self._structure = None
        self._supercell_matrix = None
        self._primitive_matrix = None
        self._NAC = None
        self._symmetrize = None
        self._displacement_distance = None
        self._show_progress = False
        self._force_constants = None
        self._data_set = None

    @abstractmethod
    def get_forces(self, cell_with_dist):
        raise NotImplementedError

    def get_path_using_seek_path(self):

        """ Obtain the path in reciprocal space to plot the phonon band structure

        :return: dictionary with list of q-points and labels of high symmetry points
        """

        try:
            import seekpath

            cell = self._structure.get_cell()
            positions = self._structure.get_scaled_positions()
            numbers = np.unique(self._structure.get_chemical_symbols(), return_inverse=True)[1]

            path_data = seekpath.get_path((cell, positions, numbers))

            labels = path_data['point_coords']

            band_ranges = []
            for _set in path_data['path']:
                band_ranges.append([labels[_set[0]], labels[_set[1]]])

            return {'ranges': band_ranges,
                    'labels': path_data['path']}
        except ImportError:
            print('Seekpath not installed. Autopath is deactivated')
            band_ranges = ([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]])
            return {'ranges': band_ranges,
                    'labels': [['GAMMA', '1/2 0 1/2']]}

    def get_force_constants(self, include_data_set=False):
        """
        calculate the force constants with phonopy using lammps to calculate forces

        :return: ForceConstants type object containing force constants
        """

        if self._force_constants is None:
            phonon = get_phonon(self._structure,
                                setup_forces=False,
                                super_cell_phonon=self._supercell_matrix,
                                primitive_matrix=self._primitive_matrix,
                                NAC=self._NAC,
                                symmetrize=self._symmetrize)
            phonon.generate_displacements(distance=self._displacement_distance)
            cells_with_disp = phonon.supercells_with_displacements
            data_set = phonon.dataset

            # Check forces for non displaced supercell
            forces_supercell = self.get_forces(phonon.supercell)
            if np.max(forces_supercell) > 1e-1:
                warnings.warn('Large atomic forces found for non displaced structure: '
                              '{}. Make sure your unit cell is properly optimized'
                              .format(np.max(forces_supercell)))

            # Get forces from lammps
            for i, cell in enumerate(cells_with_disp):
                if self._show_progress:
                    print('displacement {} / {}'.format(i + 1, len(cells_with_disp)))
                forces = self.get_forces(cell)
                data_set['first_atoms'][i]['forces'] = forces

            phonon.dataset = data_set
            phonon.produce_force_constants()
            self._force_constants = phonon.force_constants
            self._data_set = data_set

        if include_data_set:
            return [self._force_constants, self._data_set]
        else:
            return self._force_constants

    def plot_phonon_dispersion_bands(self):
        """
        Plot phonon band structure using seekpath automatic k-path
        Warning: The labels may be wrong if the structure is not standardized

        """
        import matplotlib.pyplot as plt

        def replace_list(text_string):
            substitutions = {'GAMMA': u'$\Gamma$'}

            for item in substitutions.items():
                text_string = text_string.replace(item[0], item[1])
            return text_string

        force_constants = self.get_force_constants()
        bands_and_labels = self.get_path_using_seek_path()

        _bands = obtain_phonon_dispersion_bands(self._structure,
                                                bands_and_labels['ranges'],
                                                force_constants,
                                                self._supercell_matrix,
                                                primitive_matrix=self._primitive_matrix)

        for i, freq in enumerate(_bands[1]):
            plt.plot(_bands[1][i], _bands[2][i], color='r')

            # plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_xaxis().set_ticks([])

        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, _bands[1][-1][-1]])
        plt.axhline(color='k', ls='dashed')
        plt.suptitle('Phonon dispersion')

        if 'labels' in bands_and_labels:
            plt.rcParams.update({'mathtext.default': 'regular'})

            labels = bands_and_labels['labels']

            labels_e = []
            x_labels = []
            for i, freq in enumerate(_bands[1]):
                if labels[i][0] == labels[i - 1][1]:
                    labels_e.append(replace_list(labels[i][0]))
                else:
                    labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                x_labels.append(_bands[1][i][0])
            x_labels.append(_bands[1][-1][-1])
            labels_e.append(replace_list(labels[-1][1]))
            labels_e[0] = replace_list(labels[0][0])

            plt.xticks(x_labels, labels_e, rotation='horizontal')

        plt.show()

    def write_force_constants(self, filename='FORCE_CONSTANTS', hdf5=False):
        """
        Write the force constants in a file in phonopy plain text format

        :param filename: Force constants filename
        :param hdf5: save as HDF5
        """

        force_constants = self.get_force_constants()
        if hdf5:
            write_force_constants_to_hdf5(force_constants, filename=filename)
        else:
            write_FORCE_CONSTANTS(force_constants, filename=filename)

    def write_force_sets(self, filename='FORCE_SETS'):
        """
        Write the force sets in a file in phonopy plain text format

        :param filename: Force sets filename
        """

        data_set = self.get_force_constants(include_data_set=True)[1]

        write_FORCE_SETS(data_set, filename=filename)

    def get_unitcell(self):
        """
        Get unit cell structure

        :return unitcell: unit cell 3x3 matrix (lattice vectors in rows)
        """
        return self._structure

    def get_supercell_matrix(self):
        """
        Get the supercell matrix

        :return supercell: the supercell 3x3 matrix (list of lists)
        """
        return self._supercell_matrix

    def get_primitive_matrix(self):
        return self._primitive_matrix

    def get_seekpath_bands(self, band_resolution=30):
        ranges = self.get_path_using_seek_path()['ranges']
        bands = []
        for q_start, q_end in ranges:
            band = []
            for i in range(band_resolution + 1):
                band.append(
                    np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
            bands.append(band)

        return bands

    def write_unitcell_POSCAR(self, filename='POSCAR'):
        """
        Write unit cell in VASP POSCAR type file

        :param filename: POSCAR file name (Default: POSCAR)
        """
        poscar_txt = generate_VASP_structure(self._structure)

        with open(filename, mode='w') as f:
            f.write(poscar_txt)

    def get_phonopy_phonon(self):
        """
        Return phonopy phonon object with unitcell, primitive cell and
        the force constants set.

        :return:
        """

        phonon = get_phonon(self._structure,
                            setup_forces=False,
                            super_cell_phonon=self._supercell_matrix,
                            primitive_matrix=self._primitive_matrix,
                            NAC=self._NAC,
                            symmetrize=self._symmetrize)

        phonon.force_constants = self.get_force_constants()

        return phonon


class ForceConstants:
    """
    Define Force constants object
    """

    def __init__(self, force_constants, supercell=np.identity(3)):
        """
        Initialize force constants

        :param force_constants: array matrix containing the force constants (phonopy format)
        :param supercell: 3x3 array (or list of lists) containing the supercell definition
        """

        self._force_constants = np.array(force_constants)
        self._supercell = np.array(supercell)

    def get_array(self):
        """
        get the force constants array in phonopy format

        :return: force constants array
        """
        return self._force_constants

    def get_supercell(self):
        """
        get the supercell (respect to the unit cell) in which the force constants are defined

        :return: 3x3 array containing the supercell
        """
        return self._supercell


class MyPhonopyAtoms(PhonopyAtoms):
    def __init__(self, original_positions, masses, symbols, cell, ids):
        super().__init__(positions=original_positions,
                         masses=masses,
                         symbols=symbols,
                         cell=cell)

        self._ids = np.asarray(ids)

    @property
    def ids(self):
        return self._ids

    def write_as_lammps_data(self, file_name):
        """for debugging: writes current positions with original atom ID order"""
        positions = self.positions
        numbers = self.numbers
        cell = self.cell

        xlo = ylo = zlo = 0.0
        xhi = cell[0][0]
        yhi = cell[1][1]
        zhi = cell[2][2]
        xy = cell[1][0]
        xz = cell[2][0]
        yz = cell[2][1]

        typ_dict = {num: i + 1 for i, num in enumerate(np.unique(self.numbers))}
        n_atom_types = len(typ_dict)

        with open(file_name, 'w') as f:
            f.write("structure data file generated by MyPhonolammps\n")
            f.write("\n")
            f.write("{:d} atoms\n".format(len(numbers)))
            f.write("{:d} atom types\n".format(n_atom_types))
            f.write("\n")
            f.write("{:f} {:f} xlo xhi\n".format(xlo, xhi))
            f.write("{:f} {:f} ylo yhi\n".format(ylo, yhi))
            f.write("{:f} {:f} zlo zhi\n".format(zlo, zhi))
            f.write("{:f} {:f} {:f} xy xz yz\n".format(xy, xz, yz))
            f.write("\nAtoms # atomic\n")
            f.write("\n")

            i_atom = 0
            for pos, num in zip(positions, numbers):
                f.write("{:d} {:d} {:f} {:f} {:f}\n"
                        .format(self.ids[i_atom], typ_dict[num], *pos))
                i_atom += 1
            f.write("\n")


class MyPhonolammps(MyPhonoBase):
    """
    A rewrite of the original, because it seems to shuffle the atom order
    in my structure when extracting properties from LAMMPS.

    I will avoid extracting atom properties, instead (stupidly) writing
    a SORTED dump file and reading it every time.
    """

    def get_forces(self, cell_with_disp):
        """
        I have rewritten this to use dump files and MyPhonopyAtoms.
        The point is to avoid shuffling atoms in the structure.

        ORIGINAL DOCSTRING:

        Calculate the forces of a supercell using lammps

        :param cell_with_disp: supercell from which determine the forces
        :return: numpy array matrix with forces of atoms [N_atoms x 3]
        """
        cmd_list = ['-log', 'none']
        if not self._show_log:
            cmd_list += ['-echo', 'none', '-screen', 'none']

        lmp = MyLammps(cmdargs=cmd_list)
        lmp.commands_list(self._lammps_commands_list)
        lmp.command('replicate {} {} {}'
                    .format(*np.diag(self._supercell_matrix).astype(int)))
        lmp.command('run 0')

        coordinates = cell_with_disp.get_positions()
        for i, atom_id in enumerate(self._structure.ids):
            lmp.command("set atom {:d} x {:f} y {:f} z {:f}"
                        .format(atom_id, *coordinates[i]))
        lmp.command("run 0")

        # tell LAMMPS to dump the relevant info
        _temp_file_ = self._next_temp_file_name("_FORCES_DUMP_")
        lmp.commands_list([
            "dump _GetForcesDump_ all custom 1 {} fx fy fz".format(_temp_file_),
            "dump_modify _GetForcesDump_ sort id"
            " format 1 %20.15g format 2 %20.15g format 3 %20.15g",
            "run 0",
            "undump _GetForcesDump_"
        ])
        lmp.close()

        forces = read_forces_from_dump(_temp_file_) * unit_factors[self.units]

        return forces

    def write_force_constants(self,
                              filename: str = 'FORCE_CONSTANTS',
                              hdf5: bool = False,
                              omit_zeros: bool = True,
                              omit_zeros_thresh: float = 1e-6) -> None:
        """
        Write the force constants in a file in phonopy plain text format

        :param filename: force constants filename
        :param hdf5: write to HDF5 file
        :param omit_zeros: omit writing all-zero entries in the output file
        :param omit_zeros_thresh: omit force constant matrices with norm below this value
        """
        if hdf5:
            # deferring HDF5 writing to super class for now
            super().write_force_constants(filename, hdf5)
            return

        force_constants = self.get_force_constants()
        fc_shape = force_constants.shape
        dim = fc_shape[-1]
        indices = np.arange(fc_shape[0], dtype='intc')

        with open(filename, 'w') as fc_file:

            # write total number of interactions (including 0-matrices)
            fc_file.write("%4d %4d\n" % fc_shape[:2])

            if omit_zeros:
                # only write when interaction matrix norm isn't too small
                for i, s_i in enumerate(indices):
                    for j in range(fc_shape[1]):
                        norm = 0.0
                        for vec in force_constants[i][j]:
                            norm += np.linalg.norm(vec)
                        if norm > omit_zeros_thresh:
                            fc_file.write("%d %d\n" % (s_i + 1, j + 1))
                            for vec in force_constants[i][j]:
                                fc_file.write(("%22.15f" * dim + "\n") % tuple(vec))
            else:
                # write every line
                for i, s_i in enumerate(indices):
                    for j in range(fc_shape[1]):
                        fc_file.write("%d %d\n" % (s_i + 1, j + 1))
                        for vec in force_constants[i][j]:
                            fc_file.write(("%22.15f" * dim + "\n") % tuple(vec))

    def write_harmonic_constants(self,
                                 filename: str = "HARMONIC_CONSTANTS",
                                 omit_zeros: bool = True,
                                 omit_zeros_thresh: float = 1e-6) -> None:
        """
        Write the harmonic constants (force_constants[i][j] / sqrt(mass[i] * mass[j]))
        in the same format as force constants.

        :param filename: force constants filename
        :param omit_zeros: omit writing all-zero entries in the output file
        :param omit_zeros_thresh: omit harmonic constant matrices with norm below this value
        """

        force_constants = self.get_force_constants()
        masses = self._structure.masses
        fc_shape = force_constants.shape
        dim = fc_shape[-1]
        indices = np.arange(fc_shape[0], dtype='intc')

        with open(filename, 'w') as hc_file:

            # write total number of interactions (including 0-matrices)
            hc_file.write("%4d %4d\n" % fc_shape[:2])

            if omit_zeros:
                # only write when interaction matrix norm isn't too small
                for i, s_i in enumerate(indices):
                    for j in range(fc_shape[1]):
                        m_i = masses[i]
                        m_j = masses[j]

                        norm = 0.0
                        for vec in force_constants[i][j]:
                            norm += np.linalg.norm(vec)
                        if norm > omit_zeros_thresh:
                            hc_file.write("%d %d\n" % (s_i + 1, j + 1))
                            for vec in force_constants[i][j]:
                                hc_file.write(("%22.15f" * dim + "\n")
                                              % tuple(vec / sqrt(m_i * m_j)))
            else:
                # write every line
                for i, s_i in enumerate(indices):
                    for j in range(fc_shape[1]):
                        m_i = masses[i]
                        m_j = masses[j]

                        # write interaction sub-matrix [i][j]
                        hc_file.write("%d %d\n" % (s_i + 1, j + 1))
                        for vec in force_constants[i][j]:
                            hc_file.write(("%22.15f" * dim + "\n")
                                          % tuple(vec / sqrt(m_i * m_j)))

    def __init__(self,
                 lammps_input,
                 supercell_matrix=np.identity(3),
                 primitive_matrix=np.identity(3),
                 displacement_distance=0.01,
                 trash_counter_max=50,
                 show_log=False,
                 show_progress=False,
                 use_NAC=False,
                 symmetrize=True,
                 recenter_atoms=True):
        """
        [a modified version of the] Main PhonoLAMMPS class

        :param lammps_input: LAMMPS input file name or list of commands
        :param supercell_matrix:  3x3 matrix supercell
        :param primitive_matrix:  3x3 matrix primitive cell
        :param displacement_distance: displacement distance in Angstroms
        :param trash_counter_max: maximum number of temp files written before overwriting
        :param show_log: Set true to display lammps log info
        :param show_progress: Set true to display progress of calculation
        """
        super().__init__()

        # Check if input is file or list of commands
        if type(lammps_input) is str:
            # read from file name
            self._lammps_input_file = lammps_input
            self._lammps_commands_list = open(lammps_input).read().split('\n')
        else:
            # read from commands
            self._lammps_commands_list = lammps_input

        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        self._displacement_distance = displacement_distance
        self._trash_counter_max = trash_counter_max
        self._show_log = show_log
        self._show_progress = show_progress
        self._symmetrize = symmetrize
        self._NAC = use_NAC

        # replaced this with my own method
        self._structure = self._get_structure_from_lammps(recenter_atoms)

        self._force_constants = None
        self._data_set = None

        # to avoid permission errors when deleting large files after reading
        self._trash_dir = "./__phl_trash__"
        if not os.path.isdir(self._trash_dir):
            os.mkdir(self._trash_dir)
        self._temp_name_counter = 1

        self.units = self.get_units(self._lammps_commands_list)

        if self.units not in unit_factors.keys():
            print('Units style not supported, use: {}'.format(unit_factors.keys()))
            exit()

    def __del__(self):
        """clean up temporary files"""
        try:
            os.system("rm -rf %s" % self._trash_dir)
        except AttributeError:
            pass

    def _next_temp_file_name(self, basename):
        """assign name for temporary kinds with same basename"""
        if self._temp_name_counter > self._trash_counter_max:
            # overwrite oldest files
            self._temp_name_counter = 1
        else:
            self._temp_name_counter += 1
        _file_name = "/".join([self._trash_dir, basename + str(self._temp_name_counter)])
        return _file_name

    def _get_structure_from_lammps(self, recenter_atoms=True):
        """my version that (hopefully) won't shuffle atoms"""

        cmd_list = ['-log', 'none']
        if not self._show_log:
            cmd_list += ['-echo', 'none', '-screen', 'none']

        lmp = MyLammps(cmdargs=cmd_list)
        lmp.commands_list(self._lammps_commands_list)

        # extract box as before
        try:
            xlo = lmp.extract_global("boxxlo", 1)
            xhi = lmp.extract_global("boxxhi", 1)
            ylo = lmp.extract_global("boxylo", 1)
            yhi = lmp.extract_global("boxyhi", 1)
            zlo = lmp.extract_global("boxzlo", 1)
            zhi = lmp.extract_global("boxzhi", 1)
            xy = lmp.extract_global("xy", 1)
            yz = lmp.extract_global("yz", 1)
            xz = lmp.extract_global("xz", 1)
        except UnboundLocalError:
            boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
            xlo, ylo, zlo = boxlo
            xhi, yhi, zhi = boxhi

        # note: box is mapped to the first quadrant here
        unitcell = np.array([[xhi - xlo, xy, xz],
                             [0, yhi - ylo, yz],
                             [0, 0, zhi - zlo]]).T

        # tell LAMMPS to dump the relevant info
        _temp_file_ = "_STRUCTURE_DUMP_"
        lmp.commands_list([
            "dump _GetStructureDump_ all custom 1 {} id mass x y z".format(_temp_file_),
            "dump_modify _GetStructureDump_ sort id"
            " format 3 %20.15g format 4 %20.15g format 5 %20.15g",  # IMPORTANT
            "run 0",
            "undump _GetStructureDump_"
        ])
        na = lmp.get_natoms()
        lmp.close()

        ids, masses, symbols, positions = read_structure_params_from_dump(_temp_file_, na)

        if recenter_atoms:
            # keep atoms positions same relative to the box
            positions = np.asarray(positions) - np.array([xlo, ylo, zlo])

        return MyPhonopyAtoms(original_positions=positions,
                              masses=masses,
                              symbols=symbols,
                              cell=unitcell,
                              ids=ids)

    def write_structure(self, file_name):
        self._structure.write_as_lammps_data(file_name)

    @staticmethod
    def get_units(commands_list):
        """
        Get the units label for LAMMPS "units" command from a list of LAMMPS input commands

        :param commands_list: list of LAMMPS input commands (strings)
        :return units: string containing the units
        """
        for line in commands_list:
            if line.startswith('units'):
                return line.split()[1]
        return 'lj'

    @staticmethod
    def get_force_constants_lines(matrix: np.ndarray,
                                  omit_zeros: bool,
                                  omit_zeros_thresh: float):
        indices = np.arange(matrix.shape[0], dtype='intc')
        lines = []
        _shape = matrix.shape
        lines.append("%4d %4d" % _shape[:2])
        dim = _shape[3]

        if omit_zeros:
            # check interaction matrix norm isn't too small
            for i, s_i in enumerate(indices):
                for j in range(_shape[1]):
                    norm = 0.0
                    for vec in matrix[i][j]:
                        norm += np.linalg.norm(vec)
                    if norm > omit_zeros_thresh:
                        lines.append("%d %d" % (s_i + 1, j + 1))
                        for vec in matrix[i][j]:
                            lines.append(("%22.15f" * dim) % tuple(vec))
        else:
            # write every line
            for i, s_i in enumerate(indices):
                for j in range(_shape[1]):
                    lines.append("%d %d" % (s_i + 1, j + 1))
                    for vec in matrix[i][j]:
                        lines.append(("%22.15f" * dim) % tuple(vec))

        return lines
