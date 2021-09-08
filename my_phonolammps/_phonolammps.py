"""
Module of objects connecting core phonopy functionality
"""
from abc import ABC, abstractmethod
import warnings

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.file_IO import parse_BORN
from phonopy.file_IO import (write_FORCE_CONSTANTS,
                             write_force_constants_to_hdf5,
                             write_FORCE_SETS)

# define the force unit conversion factors to LAMMPS metal style (eV/Angstrom)
unit_factors = {
    'real': 4.336410389526464e-2,
    'metal': 1.0,
    'si': 624150636.3094,
    'tinker': 0.043  # kcal/mol to eV
}


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
            for set in path_data['path']:
                band_ranges.append([labels[set[0]], labels[set[1]]])

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

            phonon.get_displacement_dataset()
            phonon.generate_displacements(distance=self._displacement_distance)
            cells_with_disp = phonon.get_supercells_with_displacements()
            data_set = phonon.get_displacement_dataset()

            # Check forces for non displaced supercell
            forces_supercell = self.get_forces(phonon.get_supercell())
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

            phonon.set_displacement_dataset(data_set)
            phonon.produce_force_constants()
            self._force_constants = phonon.get_force_constants()
            self._data_set = data_set

        if include_data_set:
            return [self._force_constants, self._data_set]
        else:
            return self._force_constants

    def plot_phonon_dispersion_bands(self):
        """
        Plot phonon band structure using seekpath automatic k-path
        Warning: The labels may be wrong if the structure is not standarized

        """
        import matplotlib.pyplot as plt

        def replace_list(text_string):
            substitutions = {'GAMMA': u'$\Gamma$',
                             }

            for item in substitutions.items():
                text_string = text_string.replace(item[0], item[1])
            return text_string

        force_constants = self.get_force_constants()
        bands_and_labels = self.get_path_using_seek_path()

        _bands = obtain_phonon_dispersion_bands(self._structure,
                                                bands_and_labels['ranges'],
                                                force_constants,
                                                self._supercell_matrix,
                                                primitive_matrix=self._primitive_matrix,
                                                band_resolution=30)

        for i, freq in enumerate(_bands[1]):
            plt.plot(_bands[1][i], _bands[2][i], color='r')

            # plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_xaxis().set_ticks([])

        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, _bands[1][-1][-1]])
        plt.axhline(y=0, color='k', ls='dashed')
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

    def get_primitve_matrix(self):
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

        phonon.set_force_constants(self.get_force_constants())

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
                     primitive_matrix=primitive_matrix,
                     is_symmetry=symmetrize)

    # Non Analytical Corrections (NAC) from Phonopy [Frequencies only, eigenvectors no affected by this option]

    if setup_forces:
        if structure.get_force_constants() is not None:
            phonon.force_constants = structure.get_force_constants().get_array()
            phonon.set_force_constants(structure.get_force_constants().get_array())
        elif structure.get_force_sets() is not None:
            phonon.set_displacement_dataset(structure.get_force_sets().get_dict())
            phonon.produce_force_constants()
            structure.set_force_constants(ForceConstants(phonon.get_force_constants(),
                                                         supercell=structure.get_force_sets().get_supercell()))
        else:
            print('No force sets/constants available!')
            exit()

    if NAC:
        print("Using non-analytical corrections")
        primitive = phonon.get_primitive()
        try:
            nac_params = parse_BORN(primitive)
            phonon.set_nac_params(nac_params=nac_params)
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

    phonon.set_force_constants(force_constants)

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
        phonon.set_band_structure(bands, is_band_connection=band_connection, is_eigenvectors=True)
        bands_phonopy = phonon.get_band_structure()

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
