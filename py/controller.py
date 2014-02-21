import sys
import os
import tarfile
import viscosaur as vc
import datetime
import ctypes

class Controller(object):
    def __init__(self, p):
        # This line is required to prevent a weird openmpi bug
        # that causes mpi init to fail with "undefined symbol: mca_base_param_reg_int"
        # Found the fix here: http://fishercat.sr.unh.edu/trac/mrc-v3/ticket/5
        mpi = ctypes.CDLL('libmpi.so.0', ctypes.RTLD_GLOBAL)
        # Initialize the viscosaur system, including deal.ii, PETSc (or Trilinos),
        # and MPI.
        self.instance = vc.Vc(sys.argv, p)
        self.params = p
        self.mpi_rank = self.instance.get_rank()
        self._prep_data_dir()

    def kill(self):
        self._finalize_data_dir()

    def proc0_out(self, info):
        if self.mpi_rank is 0:
            print(info)

    # Clear the data directory if asked. The user is trusted to set the parameter
    # appropriately and not delete precious data
    def _prep_data_dir(self):
        if self.params['clear_data_dir'] and (self.mpi_rank is 0):
            folder = self.params['data_dir']
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception, e:
                    print e

    def _finalize_data_dir(self):
        if self.params['compress_data_dir']:
            archive_name = self.params['data_dir'].replace('/', '.')
            archive_name += datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name += '.tar.gz'
            tar = tarfile.open(archive_name, "w:gz")
            tar.add(self.params['data_dir'])
            tar.close()
