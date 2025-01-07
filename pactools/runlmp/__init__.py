"""
lmprun
=====

Methods to generate LAMMPS configurations and to execute simulations with python

"""

from .types     import TypeSetup
from .types     import TypeNotDefined
from .dump      import Dump
from .sim       import LMPSim
from .elements  import Elements
from .polymer   import add_polymer
from .polymer   import shift2com



