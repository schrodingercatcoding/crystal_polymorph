import sys
import math
import random
import os
import numpy as np

from ase import io as ase_io
from ase import Atoms
from ase import spacegroup
from ase import units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal
from atoms2st import atom_belong_to_mol1, ase_atoms_to_structure, molecule_lists
from schrodinger.structutils.analyze import center_of_mass

def get_name(arr):
    
    ret = ''
    atomic_dict = { '1':'H', '6':'C', '7':'N', '8':'O'}

    for ele in arr:
        ret += atomic_dict[str(ele)]

    return ret

def get_position_for_molecule(molecule_vectors, new_a_vector, new_b_vector, new_c_vector):
    """
    a function to return scaled_positions for molecule in new cell
    
    type  molecule_vectors: np.array
    param molecule_vectors: 3*n array containing molecule positions

    type  new_a_vector, new_b_vector, new_c_vector: np.array
    param new_a_vector, new_b_vector, new_c_vector: 1*3 cell vectors 

    return 3*n array containing molecule scaled positions in new cell
    """
    new_vectors = []
    for v in molecule_vectors:
        Va = np.dot(np.cross(new_b_vector, new_c_vector) / np.dot(new_a_vector, np.cross(new_b_vector, new_c_vector)), v )
        Vb = np.dot(np.cross(new_c_vector, new_a_vector) / np.dot(new_b_vector, np.cross(new_c_vector, new_a_vector)), v )
        Vc = np.dot(np.cross(new_a_vector, new_b_vector) / np.dot(new_c_vector, np.cross(new_a_vector, new_b_vector)), v )
        new_vectors.append([Va, Vb, Vc])
   
    print("np.array(new_vectors): ", np.array(new_vectors))
    return np.array(new_vectors)

def change_cell_length(atoms, space_group):

    name = get_name(atoms.get_atomic_numbers())
    molecule1, molecule2, molecule3, molecule4 = molecule_lists(atoms)[0], molecule_lists(atoms)[1], molecule_lists(atoms)[2], molecule_lists(atoms)[3]  # [1,5,9,13,17,21,etc]
    scaled_positions = atoms.get_scaled_positions()
    molecule1_scaled_positions = np.array([scaled_positions[i] for i in molecule1])
    molecule2_scaled_positions = np.array([scaled_positions[i] for i in molecule2])
    molecule3_scaled_positions = np.array([scaled_positions[i] for i in molecule3])
    molecule4_scaled_positions = np.array([scaled_positions[i] for i in molecule4])
    cell_params = atoms.get_cell()
    a_vector, b_vector, c_vector = cell_params[0], cell_params[1], cell_params[2]
    molecule1_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule1_scaled_positions])
    molecule2_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule2_scaled_positions])
    molecule3_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule3_scaled_positions])
    molecule4_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule4_scaled_positions])
    new_atoms = atoms.copy()
    new_atoms.set_cell(cell_params + np.array([[3.0, 0, 0],[0, 3.0, 0],[0.0, 0, 3.0]]), scale_atoms=False)
    new_cell_params = new_atoms.get_cell()
    new_a_vector, new_b_vector, new_c_vector = new_cell_params[0], new_cell_params[1], new_cell_params[2]
    new_scaled_positions_molecule1 = get_position_for_molecule(molecule1_vectors, new_a_vector, new_b_vector, new_c_vector)
    print("new_scaled_positions_molecule1: ", new_scaled_positions_molecule1)
    exit()
     
if __name__ == "__main__": 

    atoms = ase_io.read(sys.argv[1])
    space_group = spacegroup.get_spacegroup(atoms)
    change_cell_length(atoms, space_group)



