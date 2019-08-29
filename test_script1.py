import sys
import math
import random
import os
import numpy as np

from ase import io as ase_io
from ase import spacegroup
from ase import units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal
from atoms2st import atom_belong_to_mol1, ase_atoms_to_structure, molecule_lists
from schrodinger.structutils.analyze import center_of_mass

def random_draw(mu, sigma):
    """
    choose a random number from a gaussian distribution
    """
    
    s = np.random.normal(mu, sigma, 1000)

    return random.choice(s)

def get_com(atoms, molecule_lists):
    """
    return a (n_molecules, 3) for center of mass
    """

    ret_arr = []
    st = ase_atoms_to_structure(atoms)
    for molecule_list in molecule_lists:
        com = center_of_mass(st, molecule_list)
        ret_arr.append(com)

    return np.array(ret_arr)

def rigid_body_movement(atoms):

    ret_molecule_lists = molecule_lists(atoms)
    molecule1 = ret_molecule_lists[0]
    atom1_index = molecule1[0]
    scaled_position = atoms.get_scaled_positions(wrap=True)

    print("atom1_index: ", atom1_index)
#    print("atoms scaled position: ", scaled_position)
    print("atoms1 scaled position: ", scaled_position[atom1_index - 1])
     
    atom2_index = molecule1[1]   
    print("atoms2 scaled position: ", scaled_position[atom2_index - 1])
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    




    return atoms

if __name__ == "__main__": 

    print("runing main function for debugging...")
    atoms = ase_io.read(sys.argv[1])
    space_group = spacegroup.get_spacegroup(atoms)
    print("after perturbation: ", space_group)
    ret_atoms = rigid_body_movement(atoms)
    space_group = spacegroup.get_spacegroup(ret_atoms)
    print("after perturbation: ", space_group)
    outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed.cif"
    ase_io.write(outfile, ret_atoms)




