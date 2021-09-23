"""Here the original lammps class is wrapped to allow specifying the `modpath`"""
from lammps import *
from lammps import lammps as original_lammps


class _lammps(original_lammps):
    """new version of the lammps class"""

    # path to the directory containing liblammps.so or liblammps.dylib
    modpath: str = None

    def __init__(self, name="", cmdargs=None, ptr=None, comm=None):
        if self.modpath is None:
            # instantiate the original class as usual
            original_lammps.__init__(self, name, cmdargs, ptr, comm)
        else:
            # instantiate with a specific `modpath`
            modpath = os.path.abspath(_lammps.modpath)
            try:
                self._create_instance(modpath)
            except OSError:
                raise FileNotFoundError(
                    f"LAMMPS shared library not found in directory {modpath}"
                )

    def _create_instance(self, modpath: str,
                         name="", cmdargs=None, ptr=None, comm=None) -> None:
        """
        Run the original instance creation code, except `modpath` is now arbitrary.
        Specifying `modpath` avoids the issue of not finding the LAMMPS shared library,
        even though it may exist elsewhere.

        :param modpath: path to the directory containing "liblammps.so" or "liblammps.dylib"
        """
        self.comm = comm
        self.opened = 0

        # determine module location

        self.lib = None
        self.lmp = None

        # if a pointer to a LAMMPS object is handed in,
        # all symbols should already be available

        try:
            if ptr: self.lib = CDLL("", RTLD_GLOBAL)
        except:
            self.lib = None

        # load liblammps.so unless name is given
        #   if name = "g++", load liblammps_g++.so
        # try loading the LAMMPS shared object from the location
        #   of my_lammps.py with an absolute path,
        #   so that LD_LIBRARY_PATH does not need to be set for regular install
        # fall back to loading with a relative path,
        #   typically requires LD_LIBRARY_PATH to be set appropriately

        if any([f.startswith('liblammps') and f.endswith('.dylib') for f in os.listdir(modpath)]):
            lib_ext = ".dylib"
        else:
            lib_ext = ".so"

        if not self.lib:
            try:
                if not name:
                    self.lib = CDLL(join(modpath, "liblammps" + lib_ext), RTLD_GLOBAL)
                else:
                    self.lib = CDLL(join(modpath, "liblammps_%s" % name + lib_ext),
                                    RTLD_GLOBAL)
            except:
                if not name:
                    self.lib = CDLL("liblammps" + lib_ext, RTLD_GLOBAL)
                else:
                    self.lib = CDLL("liblammps_%s" % name + lib_ext, RTLD_GLOBAL)

        # define ctypes API for each library method
        # NOTE: should add one of these for each lib function

        self.lib.lammps_extract_box.argtypes = \
            [c_void_p, POINTER(c_double), POINTER(c_double),
             POINTER(c_double), POINTER(c_double), POINTER(c_double),
             POINTER(c_int), POINTER(c_int)]
        self.lib.lammps_extract_box.restype = None

        self.lib.lammps_reset_box.argtypes = \
            [c_void_p, POINTER(c_double), POINTER(c_double), c_double, c_double, c_double]
        self.lib.lammps_reset_box.restype = None

        self.lib.lammps_gather_atoms.argtypes = \
            [c_void_p, c_char_p, c_int, c_int, c_void_p]
        self.lib.lammps_gather_atoms.restype = None

        self.lib.lammps_gather_atoms_concat.argtypes = \
            [c_void_p, c_char_p, c_int, c_int, c_void_p]
        self.lib.lammps_gather_atoms_concat.restype = None

        self.lib.lammps_gather_atoms_subset.argtypes = \
            [c_void_p, c_char_p, c_int, c_int, c_int, POINTER(c_int), c_void_p]
        self.lib.lammps_gather_atoms_subset.restype = None

        self.lib.lammps_scatter_atoms.argtypes = \
            [c_void_p, c_char_p, c_int, c_int, c_void_p]
        self.lib.lammps_scatter_atoms.restype = None

        self.lib.lammps_scatter_atoms_subset.argtypes = \
            [c_void_p, c_char_p, c_int, c_int, c_int, POINTER(c_int), c_void_p]
        self.lib.lammps_scatter_atoms_subset.restype = None

        self.lib.lammps_find_pair_neighlist.argtypes = [c_void_p, c_char_p, c_int, c_int, c_int]
        self.lib.lammps_find_pair_neighlist.restype = c_int

        self.lib.lammps_find_fix_neighlist.argtypes = [c_void_p, c_char_p, c_int]
        self.lib.lammps_find_fix_neighlist.restype = c_int

        self.lib.lammps_find_compute_neighlist.argtypes = [c_void_p, c_char_p, c_int]
        self.lib.lammps_find_compute_neighlist.restype = c_int

        self.lib.lammps_neighlist_num_elements.argtypes = [c_void_p, c_int]
        self.lib.lammps_neighlist_num_elements.restype = c_int

        self.lib.lammps_neighlist_element_neighbors.argtypes = [c_void_p, c_int, c_int,
                                                                POINTER(c_int), POINTER(c_int),
                                                                POINTER(POINTER(c_int))]
        self.lib.lammps_neighlist_element_neighbors.restype = None

        # if no ptr provided, create an instance of LAMMPS
        #   don't know how to pass an MPI communicator from PyPar
        #   but we can pass an MPI communicator from mpi4py v2.0.0 and later
        #   no_mpi call lets LAMMPS use MPI_COMM_WORLD
        #   cargs = array of C strings from args
        # if ptr, then are embedding Python in LAMMPS input script
        #   ptr is the desired instance of LAMMPS
        #   just convert it to ctypes ptr and store in self.lmp

        if not ptr:

            # with mpi4py v2, can pass MPI communicator to LAMMPS
            # need to adjust for type of MPI communicator object
            # allow for int (like MPICH) or void* (like OpenMPI)

            if comm:
                if not _lammps.has_mpi4py:
                    raise Exception('Python mpi4py version is not 2 or 3')
                if _lammps.MPI._sizeof(_lammps.MPI.Comm) == sizeof(c_int):
                    MPI_Comm = c_int
                else:
                    MPI_Comm = c_void_p

                narg = 0
                cargs = 0
                if cmdargs:
                    cmdargs.insert(0, "my_lammps.py")
                    narg = len(cmdargs)
                    for i in range(narg):
                        if type(cmdargs[i]) is str:
                            cmdargs[i] = cmdargs[i].encode()
                    cargs = (c_char_p * narg)(*cmdargs)
                    self.lib.lammps_open.argtypes = [c_int, c_char_p * narg, \
                                                     MPI_Comm, c_void_p()]
                else:
                    self.lib.lammps_open.argtypes = [c_int, c_int, \
                                                     MPI_Comm, c_void_p()]

                self.lib.lammps_open.restype = None
                self.opened = 1
                self.lmp = c_void_p()
                comm_ptr = _lammps.MPI._addressof(comm)
                comm_val = MPI_Comm.from_address(comm_ptr)
                self.lib.lammps_open(narg, cargs, comm_val, byref(self.lmp))

            else:
                if _lammps.has_mpi4py:
                    from mpi4py import MPI
                    self.comm = MPI.COMM_WORLD
                self.opened = 1
                if cmdargs:
                    cmdargs.insert(0, "my_lammps.py")
                    narg = len(cmdargs)
                    for i in range(narg):
                        if type(cmdargs[i]) is str:
                            cmdargs[i] = cmdargs[i].encode()
                    cargs = (c_char_p * narg)(*cmdargs)
                    self.lmp = c_void_p()
                    self.lib.lammps_open_no_mpi(narg, cargs, byref(self.lmp))
                else:
                    self.lmp = c_void_p()
                    self.lib.lammps_open_no_mpi(0, None, byref(self.lmp))
                    # could use just this if LAMMPS lib interface supported it
                    # self.lmp = self.lib.lammps_open_no_mpi(0,None)

        else:
            # magic to convert ptr to ctypes ptr
            if sys.version_info >= (3, 0):
                # Python 3 (uses PyCapsule API)
                pythonapi.PyCapsule_GetPointer.restype = c_void_p
                pythonapi.PyCapsule_GetPointer.argtypes = [py_object, c_char_p]
                self.lmp = c_void_p(pythonapi.PyCapsule_GetPointer(ptr, None))
            else:
                # Python 2 (uses PyCObject API)
                pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
                pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]
                self.lmp = c_void_p(pythonapi.PyCObject_AsVoidPtr(ptr))

        # optional numpy support (lazy loading)
        self._numpy = None

        # set default types
        self.c_bigint = get_ctypes_int(self.extract_setting("bigint"))
        self.c_tagint = get_ctypes_int(self.extract_setting("tagint"))
        self.c_imageint = get_ctypes_int(self.extract_setting("imageint"))
        self._installed_packages = None

        # add way to insert Python callback for fix external
        self.callback = {}
        self.FIX_EXTERNAL_CALLBACK_FUNC = CFUNCTYPE(None, py_object, self.c_bigint, c_int,
                                                    POINTER(self.c_tagint),
                                                    POINTER(POINTER(c_double)),
                                                    POINTER(POINTER(c_double)))
        self.lib.lammps_set_fix_external_callback.argtypes = [c_void_p, c_char_p,
                                                              self.FIX_EXTERNAL_CALLBACK_FUNC,
                                                              py_object]
        self.lib.lammps_set_fix_external_callback.restype = None
