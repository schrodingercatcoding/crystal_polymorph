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
    x_com = get_com(atoms, ret_molecule_lists)
    print("x_com: ", x_com)
    F = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    delta_a, delta_b, delta_c = random_draw(0.1,0.1), random_draw(0.1,0.1), random_draw(0.1,0.1)
    new_F = F + np.matrix([[delta_a, 0, 0],[0, delta_b, 0],[0, 0, delta_c]])
    X = atoms.get_positions()
    x_com_new = np.matmul(x_com, new_F)
    L = atoms.get_cell()
    L_new = np.matmul(L, new_F)
    atoms.set_cell(L_new, scale_atoms=False)

#imolec, molec:  0 [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
#imolec, molec:  1 [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82]
#imolec, molec:  2 [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83]
#imolec, molec:  3 [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84]

    for imolec, molec in enumerate(ret_molecule_lists):
        print("imolec, molec: ", imolec, molec)
        for j in molec:
            for i in [0, 1, 2]:
                X[j-1, i] += (x_com_new[imolec, i] - x_com[imolec, i]) 
                #X[j-1, i] += x_com_new[imolec, i]

    atoms.set_positions(X)

    return atoms

def get_equivalent_sites(scaled_pos, space_group):
    ret = []
    for row in scaled_pos:
        four_rows, _ = space_group.equivalent_sites(row)
        for r in four_rows:
            ret.append(r)

    return np.array(ret)


def move_molecule1(atoms, space_group):
    
    print("atoms.get_distance(1,5): ", atoms.get_distance(1,5, mic=False, vector=True))
    print("atoms.get_distance(1,21): ", atoms.get_distance(1,21, mic=False, vector=True))
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    X = atoms.get_positions()
    old_scaled_position = atoms.get_scaled_positions()
    print("a, b, c, alpha, beta, gamma: ", a, b, c, alpha, beta, gamma)
    #print("X: ", X)
    #print("old_scaled_position: ", old_scaled_position)
    molecule1 = molecule_lists(atoms)[0]
    st = ase_atoms_to_structure(atoms)
    x_com1 = center_of_mass(st, molecule1)
    X_molecule1 = np.array([X[x-1] for x in molecule1])
    #print("X_molecule1: ", X_molecule1)

    F = np.array([[1,0,0],[0,1,0],[0,0,1]])
    delta_a, delta_b, delta_c = random_draw(0.1,0.1), random_draw(0.1,0.1), random_draw(0.1,0.1)
    new_F = F + np.array([[delta_a, 0, 0],[0, delta_b, 0],[0, 0, delta_c]])
    x_com1_new = np.dot(x_com1, new_F)
 
    diff_com = x_com1_new - x_com1
    #print("diff_com: ", diff_com)

    new_X = []
    
    for i, row in enumerate(X):
        if i+1 in molecule1:
            new_X.append(row + diff_com)
        else:
            new_X.append(row)
    
    L = atoms.get_cell()
    new_L = np.dot(L, new_F)

    new_atoms = atoms.copy()
    new_atoms.set_positions(new_X)
    new_atoms.set_cell(new_L, scale_atoms=False)

    scaled_positions = new_atoms.get_scaled_positions() 
    molecule1_scaled_positions = [scaled_positions[i - 1] for i in molecule1]
    final_scaled_positions = get_equivalent_sites(molecule1_scaled_positions, space_group)
    #print("final_scaled_positions: ", final_scaled_positions)
    new_atoms.set_scaled_positions(final_scaled_positions)
    print("new_atoms.get_distance(1,5): ", new_atoms.get_distance(1,5, mic=False, vector=True))
    print("new_atoms.get_distance(1,21): ", new_atoms.get_distance(1,21, mic=False, vector=True))
    
    return new_atoms

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
        #print("b x c: ", np.cross(new_b_vector, new_c_vector))
        #print("a . (b x c): ", np.dot(new_a_vector, np.cross(new_b_vector, new_c_vector)))
        #print("(b x c)/(a. (bxc)): ", np.cross(new_b_vector, new_c_vector) / np.dot(new_a_vector, np.cross(new_b_vector, new_c_vector)))    
        Va = np.dot(np.cross(new_b_vector, new_c_vector) / np.dot(new_a_vector, np.cross(new_b_vector, new_c_vector)), v )
        #print("Va: ", Va)
        Vb = np.dot(np.cross(new_c_vector, new_a_vector) / np.dot(new_b_vector, np.cross(new_c_vector, new_a_vector)), v )
        Vc = np.dot(np.cross(new_a_vector, new_b_vector) / np.dot(new_c_vector, np.cross(new_a_vector, new_b_vector)), v )
        new_vectors.append([Va, Vb, Vc])
   
    #print("np.array(new_vectors): ", np.array(new_vectors))
    return np.array(new_vectors)

def change_cell_length(atoms, space_group):

    #print(atoms.get_positions())
    #print(atoms.get_atomic_numbers())
    name = get_name(atoms.get_atomic_numbers())
    #print(name)
    molecule1, molecule2, molecule3, molecule4 = molecule_lists(atoms)[0], molecule_lists(atoms)[1], molecule_lists(atoms)[2], molecule_lists(atoms)[3]  # [1,5,9,13,17,21,etc]
    #print("molecule1: ", molecule1)
    #print(molecule2)
    #print(molecule3)
    #print(molecule4)
    scaled_positions = atoms.get_scaled_positions()
    molecule1_scaled_positions = np.array([scaled_positions[i] for i in molecule1])
    #print("molecule1_scaled_positions: ", molecule1_scaled_positions)
    molecule2_scaled_positions = np.array([scaled_positions[i] for i in molecule2])
    molecule3_scaled_positions = np.array([scaled_positions[i] for i in molecule3])
    molecule4_scaled_positions = np.array([scaled_positions[i] for i in molecule4])
    cell_params = atoms.get_cell()
    #print(atoms.get_cell())
    a_vector, b_vector, c_vector = cell_params[0], cell_params[1], cell_params[2]
    molecule1_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule1_scaled_positions])
    #print("molecule1_vectors: ", molecule1_vectors)
    molecule2_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule2_scaled_positions])
    molecule3_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule3_scaled_positions])
    molecule4_vectors = np.array([faction[0] * a_vector + faction[1] * b_vector + faction[2] * c_vector for faction in molecule4_scaled_positions])
    new_atoms = atoms.copy()
    new_atoms.set_cell(cell_params + np.array([[3.0, 0, 0],[0, 3.0, 0],[0.0, 0, 3.0]]), scale_atoms=True)
    new_cell_params = new_atoms.get_cell()
    print("first atoms: ", new_atoms.get_positions()[0])
    #print("new_cell_params: ", new_cell_params)
    new_a_vector, new_b_vector, new_c_vector = new_cell_params[0], new_cell_params[1], new_cell_params[2]
    new_scaled_positions_molecule1 = get_position_for_molecule(molecule1_vectors, new_a_vector, new_b_vector, new_c_vector)
    #print("new_scaled_positions_molecule1: ", new_scaled_positions_molecule1)
    new_scaled_positions_molecule2 = get_position_for_molecule(molecule2_vectors, new_a_vector, new_b_vector, new_c_vector)
    new_scaled_positions_molecule3 = get_position_for_molecule(molecule3_vectors, new_a_vector, new_b_vector, new_c_vector)
    new_scaled_positions_molecule4 = get_position_for_molecule(molecule4_vectors, new_a_vector, new_b_vector, new_c_vector)

    #print("get_symop(): ", space_group.get_symop())
    sites = []
    for r1, r2, r3, r4 in zip(new_scaled_positions_molecule1, new_scaled_positions_molecule2, new_scaled_positions_molecule3, new_scaled_positions_molecule4):
        sites.append(r1)
        sites.append(r2)
        sites.append(r3)
        sites.append(r4)

    new_positions = np.array([faction[0] * new_a_vector + faction[1] * new_b_vector + faction[2] * new_c_vector for faction in sites])
    #print(new_positions - atoms.get_positions())
    #print(new_positions)
    writer =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    #for i in molecule3:
    #    print(writer.get_distances(i, molecule3) - atoms.get_distances(i, molecule3))
    #print(writer.get_cell())

    return writer
     
if __name__ == "__main__": 

    print("runing main function for debugging...")
    atoms = ase_io.read(sys.argv[1])
    space_group = spacegroup.get_spacegroup(atoms)
    #print("before perturbation: ", space_group)
    #ret_atoms = move_molecule1(atoms, space_group)
    ret_atoms = change_cell_length(atoms, space_group)
    space_group = spacegroup.get_spacegroup(ret_atoms)
    #print("after perturbation: ", space_group)
    outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed1.cif"
    ase_io.write(outfile, ret_atoms)
    #change_cell_length(atoms, space_group)



