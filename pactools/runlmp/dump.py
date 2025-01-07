#!/bin/env python3

import os
import sys
import subprocess
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

########################################################################
########################################################################
########################################################################

class Dump:

    name:       str 
    group:      str
    type:       str
    every:      int
    fileext:    str
    args:       str
    sortby:     str

    def __init__(self,name: str, group: str, type: str, every: int, fileext=None, args=None,sortby='id'):
        self.name = name
        self.group = group
        self.type = type
        self.every = every
        if fileext is None:
            self.fileext = '.'+type
        else:
            self.fileext = fileext
        self.args = args
        self.sortby = sortby

    
    def get_outfn(self,baseout_filename: str):
        return baseout_filename + self.fileext
    
    def dump_lines(self,baseout_filename: str) -> str:
        fn = baseout_filename + self.fileext
        dl = f'dump {self.name} {self.group} {self.type} {self.every} {fn}'
        if self.args is not None:
            dl += f' {self.args}'
        dl += '\n'
        dls = list()
        dls.append(dl)

        modifies = list()
        if self.sortby is not None and isinstance(self.sortby,str) and self.sortby.lower() != 'none':
            modifies.append(f'sort {self.sortby}')

        # if self.sortby is not None and isinstance(self.sortby,str) and self.sortby.lower() != 'none':
        #     sortline = f'dump_modify {self.name} sort {self.sortby}\n'
        #     dls.append(sortline)

        # modifies.append(f'precision 1000')

        if modifies != '':
            modify_line = f'dump_modify {self.name}'
            for modify in modifies:
                modify_line += ' ' + modify
            modify_line += '\n'
            dls.append(modify_line) 

        return dls


