import sys
import os
import numpy as np

from ase import io as ase_io
from ase import spacegroup
from ase import units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal


atoms = ase_io.read(sys.argv[1])
space_group = spacegroup.get_spacegroup(atoms)    
print(space_group.get_symop())   
