import sys
import math
import os
import numpy as np

from ase import io as ase_io
from ase import spacegroup
from ase import units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal
from atoms2st import atom_belong_to_mol1 

def rotate_and_translate(atoms, atoms_index_group1, translation, rotation_x, rotation_y, rotation_z):
    """
    atoms: atoms class in ASE module

    atoms_group1: atoms index that belong to molecule1

    return a new structure that has a,b,c,alpha,beta,gamma perturbed
    and the center of mass of asymmetric unit to the origin changed
    """
    # get all scaled_positions
    space_group = spacegroup.get_spacegroup(atoms)
    scaled_positions = atoms.get_scaled_positions()
    
    #atoms_group1 = scaled_positions[::4]
    atoms_group1 = []
    #for i in range(1, atoms.get_number_of_atoms()):
    #    if atom_belong_to_mol1(i, atoms):
    #        atoms_group1.append(scaled_positions[i-1])
    #         print("dir(atom): ", dir(atoms[1]))
             #print("atoms[0].scaled_position: ",atoms[0])
             #atoms_group1.append(atoms[i].get("scaled_position"))
    #print("atoms_group1_old: ", scaled_positions[::4])
    #print("atoms_group1_new: ", np.array(atoms_group1))

    for i in atoms_index_group1:
        atoms_group1.append(scaled_positions[i-1])

    # apply a translation to atoms_group_1
    atoms_group1_translate = np.add(atoms_group1, translation)
    atoms_group1_rotation = np.dot(atoms_group1_translate, rotation_x)
    atoms_group1_rotation = np.dot(atoms_group1_translate, rotation_y)
    atoms_group1_rotation = np.dot(atoms_group1_translate, rotation_z)
    new_arr = []
    for row in atoms_group1_rotation:
        new_row = []
        for pos in row:
            if pos >= 1:
                pos = pos - 1
            elif pos < 0:
                pos = pos + 1
            new_row.append(pos)
        new_arr.append(new_row)
    atoms_group1_updated = np.array(new_arr)
    atoms_scaled_final = get_equivalent_sites(atoms_group1_updated, space_group)

    return atoms_scaled_final 

def get_equivalent_sites(scaled_pos, space_group):
    ret = []
    for row in scaled_pos:
        four_rows, _ = space_group.equivalent_sites(row)
        for r in four_rows:
            ret.append(r)
    
    return np.array(ret)

def get_new_frame(atoms, atoms_index_group1, x, y, z, theta_x, theta_y, theta_z):

    pi = math.pi
    theta_x = theta_x/180*pi # radius
    theta_y = theta_y/180*pi
    theta_z = theta_z/180*pi
    cos_theta_x, sin_theta_x = math.cos(theta_x), math.sin(theta_x)
    cos_theta_y, sin_theta_y = math.cos(theta_y), math.sin(theta_y)
    cos_theta_z, sin_theta_z = math.cos(theta_z), math.sin(theta_z)

    rotation_x = [[1, 0, 0],
                [0, cos_theta_x, -sin_theta_x],
                [0, sin_theta_x, cos_theta_x]]
    rotation_y = [[cos_theta_y, 0, sin_theta_y],
                [0, 1, 0],
                [-sin_theta_y, 0, cos_theta_y]]
    rotation_z = [[cos_theta_z, -sin_theta_z, 0],
                [sin_theta_z, cos_theta_z, 0],
                [0, 0, 1]]

    translation = [x, y, z]

    atoms_new_scaled = rotate_and_translate(atoms, atoms_index_group1, translation, rotation_x, rotation_y, rotation_z)
    ret_atoms = atoms.copy()
    ret_atoms.set_scaled_positions(atoms_new_scaled)

    return ret_atoms

def change_cells(new_atoms, d_a, d_b, d_c, d_alpha, d_beta, d_gamma):

    scaled_positions = new_atoms.get_scaled_positions() 
    a, b, c, alpha, beta, gamma = new_atoms.get_cell_lengths_and_angles()
    print(a, b, c, alpha, beta, gamma)
    new_a, new_b, new_c, new_alpha, new_beta, new_gamma =  a + d_a, b + d_b, c + d_c, alpha + d_alpha, beta + d_beta, gamma + d_gamma
    print(new_a, new_b, new_c, new_alpha, new_beta, new_gamma)
    new_atoms.set_cell([new_a, new_b, new_c, new_alpha, new_beta, new_gamma])
    new_atoms.set_scaled_positions(scaled_positions)

    return new_atoms

if __name__ == "__main__":
    
    
    atoms = ase_io.read(sys.argv[1])
    #atom_index_group1 = [] 
    #for i in range(1, atoms.get_number_of_atoms()):
    #    if atom_belong_to_mol1(i, atoms):
    #        atom_index_group1.append(i) 
   
    #print("atom_index_group1: ", atom_index_group1)

    #atoms = change_cells(atoms, 1, 2, 3, 0, 0, 0)
    #outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed_aftercell.cif"
    #ase_io.write(outfile, atoms)

    #atoms = get_new_frame(atoms, atom_index_group1, 0.05, -0.03, 0.02, 5, 1, 1)
    #outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed_aftercell_afterxyz.cif" 
    #ase_io.write(outfile, atoms)
    #print("newly perturbed molecule is written in: %s"%outfile)
 
    atoms = change_cells(atoms, 1, 2, 3, 0, 0, 0)
    outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed_aftercell.cif"
    ase_io.write(outfile, atoms)














