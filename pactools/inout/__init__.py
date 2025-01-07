"""
inout
=====

Input and output methods

"""

# conf file methods
from .lmpconf   import write_conf

# xyz file methods
from .custom    import load_custom
from .custom    import read_custom
from .custom    import read_specs
from .xyz       import load_xyz
from .xyz       import load_pos_of_type
from .xyz       import read_xyz
from .xyz       import read_xyz_atomtypes
from .xyz       import write_xyz
