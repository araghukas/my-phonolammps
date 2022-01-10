"""handling filenames of PhononRunner outputs"""
import os


class PhononOutputFiles:
    """manages output files of phonon calculations"""
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
