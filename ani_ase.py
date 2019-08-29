""" tools for running ani_ase """
import sys
import os
import time

from ase_interface import ANIENS
from ase_interface import aniensloader

import  ase
from ase import units
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from utils import models


def ani_ase_calculator(ani_model):
    """
    Return a calculator from a model.
    choices are %s
    """ % " ".join(models.keys())
    
    return ANIENS(aniensloader(models[ani_model]))

# Calculate energy
#ei = mol.get_potential_energy()
#print("Initial Energy: ",ei)

# Optimize molecule
#print("Optimizing...")
#start_time = time.time()
#dyn = LBFGS(mol)
#dyn.run(fmax=0.001)
#print('[ANI Optimization - Total time:', time.time() - start_time, 'seconds]')

# Calculate energy
#ef = mol.get_potential_energy()
#print("Final Energy: ",ef)

