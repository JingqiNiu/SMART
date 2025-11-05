
import os
import sys
import glob2
import shutil


class Logger(object):
    def __init__(self, filename="Default.log"):
        """
        write terminal log information to file
        """
        self.terminal = sys.stdout
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def copy_file_or_dir(names, in_dir, out_dir):
    """
    copy file or directory to destination

    Args:
        names (list): like this [ '*.py' , 'loss_lib/']
        in_dir ( str): 
        out_dir (str): destination
    """
    if isinstance(names, str):
        names = [names]

    for name in names:
        name_lists = glob2.glob(name)
        for item in name_lists:
            in_name = os.path.join(in_dir, item)
            out_name = os.path.join(out_dir, item)

            if os.path.isdir(in_name):
                if os.path.exists(out_name):
                    shutil.rmtree(out_name)
                shutil.copytree(in_name, out_name)

            if os.path.isfile(in_name):
                os.makedirs(os.path.dirname(out_name), exist_ok=True)
                shutil.copyfile(in_name, out_name)
