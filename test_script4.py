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

def get_equivalent_sites(scaled_pos, space_group):

    ret = []
    for row in scaled_pos:
        four_rows, _ = space_group.equivalent_sites(row)
        for r in four_rows:
            ret.append(r)

    return np.array(ret)

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
   
    return np.array(new_vectors)

def get_xyz_after_move(atom1_old_xyz, atom1_new_xyz, molecule1_vectors):

    new_vectors = []

    x1, y1, z1 = atom1_old_xyz[:]
    x1_new, y1_new, z1_new = atom1_new_xyz[:]

    for xyz in molecule1_vectors:
        xi, yi, zi = xyz[:]
        xi_new = xi - x1 + x1_new
        yi_new = yi - y1 + y1_new
        zi_new = zi - z1 + z1_new
        new_vectors.append([xi_new, yi_new, zi_new])

    return new_vectors

def change_cell_length(atoms, space_group):

    name = get_name(atoms.get_atomic_numbers())
    #molecule1, molecule2, molecule3, molecule4 = molecule_lists(atoms)[0], molecule_lists(atoms)[1], molecule_lists(atoms)[2], molecule_lists(atoms)[3]  # [1,5,9,13,17,21,etc]
    st = ase_atoms_to_structure(atoms)
    #print("myxyz: ", st.getXYZ())
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = []
    for molecule in st.molecule:
        mole_list = []
        for atom in molecule.atom:
            mole_list.append(atom.index - 1)
        molecules.append(mole_list)

    old_positions = my_atoms.get_positions()
    #get xyz coordinate for original molecules
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    cell_params = my_atoms.get_cell()
    new_atoms = my_atoms.copy()
    new_atoms.set_cell(cell_params + np.array([[3.0, 0, 0],[0, 3.0, 0],[0.0, 0, 3.0]]), scale_atoms=True)
    new_cell_params = new_atoms.get_cell()
    new_a_vector, new_b_vector, new_c_vector = new_cell_params[0], new_cell_params[1], new_cell_params[2]
    new_molecule1_vectors = get_xyz_after_move(my_atoms.get_positions()[0], new_atoms.get_positions()[0], molecule1_vectors)

    sites = []
    for pos in new_molecule1_vectors:
        for rot, trans in space_group.get_symop():
            trans_a, trans_b, trans_c = trans[:]
            site = np.dot(rot, pos) + trans_a * new_a_vector + trans_b * new_b_vector + trans_c * new_c_vector
            sites.append(site)
    new_positions = np.array(sites)

    writer =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return writer
     
if __name__ == "__main__": 

    print("runing main function for debugging...")
    atoms = ase_io.read(sys.argv[1])
    space_group = spacegroup.get_spacegroup(atoms)
    print("before perturbation: ", space_group)
    ret_atoms = change_cell_length(atoms, space_group)
    space_group = spacegroup.get_spacegroup(ret_atoms)
    print("after perturbation: ", space_group)
    outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed1.cif"
    ase_io.write(outfile, ret_atoms)

