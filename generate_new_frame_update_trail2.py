import sys
import math
import random
import os
import numpy as np

from ase import io as ase_io
from ase import spacegroup
from ase.spacegroup import Spacegroup
from ase import units
from ase import Atoms
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal
from atoms2st import atom_belong_to_mol1, ase_atoms_to_structure, molecule_lists
from schrodinger.structutils.analyze import center_of_mass

def random_uniform(x):
 
    return random.uniform(-x, x)

def random_draw(sigma):
    """
    choose a random number from a gaussian distribution
    """
    
    return np.random.normal(0, sigma)

def get_equivalent_sites(scaled_pos, space_group):
    ret = []
    for row in scaled_pos:
        four_rows, _ = space_group.equivalent_sites(row)
        for r in four_rows:
            ret.append(r)
    
    return np.array(ret)

def get_name(arr):
    
    ret = ''
    atomic_dict = { '1':'H', '6':'C', '7':'N', '8':'O', '16':'S', '17':'Cl'}

    for ele in arr:
        ret += atomic_dict[str(ele)]

    return ret

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

def generate_translate_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1):

    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    current_XYZ = st.getXYZ()
    molecule1_XYZ = [current_XYZ[i] for i in molecule1_atoms_index]
    translation = [random_draw(sigma), random_draw(sigma), random_draw(sigma)]
    molecule1_XYZ_after_translation = []
    for row in molecule1_XYZ:
        new_row = row + translation
        molecule1_XYZ_after_translation.append(new_row)
    
    sites = []
    cell_params = my_atoms.get_cell()
    for pos in molecule1_XYZ_after_translation:
        for rot, trans in space_group.get_symop():
            trans_a, trans_b, trans_c = trans[:]
            site = np.dot(rot, pos) + trans_a * cell_params[0] + trans_b * cell_params[1] + trans_c * cell_params[2]
            sites.append(site)
    new_positions = np.array(sites)

    ret_atoms =  Atoms(get_name(atoms.get_atomic_numbers()),
                    positions=new_positions,
                    cell=my_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms

def generate_rotation_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1):
    
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    current_XYZ = st.getXYZ()
    molecule1_XYZ = [current_XYZ[i] for i in molecule1_atoms_index]
    
    delta_theta_x = random_draw(sigma)
    delta_theta_y = random_draw(sigma)
    delta_theta_z = random_draw(sigma)
    pi = math.pi
    theta_x = delta_theta_x/180*pi
    theta_y = delta_theta_y/180*pi
    theta_z = delta_theta_z/180*pi
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

    molecule1_XYZ_after_rotate = np.matmul(np.array(molecule1_XYZ), np.array(rotation_x))
    

    cell_params = my_atoms.get_cell()
    all_new_molecules = []
    for rot, trans in space_group.get_symop():
        trans_a, trans_b, trans_c = trans[:]
        translation = trans_a * cell_params[0] + trans_b * cell_params[1] + trans_c * cell_params[2]
        translation_span = [translation for i in molecule1_atoms_index]
        new_molecule = np.matmul(molecule1_XYZ_after_rotate, rot) + translation_span
        all_new_molecules.append(new_molecule)
    
    new_XYZ_to_set = []
    molecule1, molecule2, molecule3, moleclue4 = all_new_molecules[:]
    for row1, row2, row3, row4 in zip(molecule1, molecule2, molecule3, moleclue4):
        new_XYZ_to_set.append(row1)
        new_XYZ_to_set.append(row2)
        new_XYZ_to_set.append(row3)
        new_XYZ_to_set.append(row4)
    
    writer =  Atoms(get_name(atoms.get_atomic_numbers()),
                    positions=np.array(new_XYZ_to_set),
                    cell=atoms.get_cell(),
                    pbc=[1,1,1])

    return writer

def generate_cell_length_a_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1):
    
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    if a != b and b != c and c != a:
        new_a, new_b, new_c = a + random_draw(sigma), b, c
    elif a != b and b == c:
        new_a, new_b = a + random_draw(sigma), b
        new_c = new_b
    elif a != b and a == c:
        new_a, new_b = a + random_draw(sigma), b
        new_c = new_a
    elif a == b and a != c:
        new_a, new_c = a + random_draw(sigma), c
        new_b = new_a
    else:
        new_a = a + random_draw(sigma)
        new_b = new_a
        new_c = new_a

    # scaled_atoms = True
    new_atoms.set_cell([new_a, new_b, new_c, alpha, beta, gamma], scale_atoms=True)
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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms 

def generate_cell_length_b_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1):
    
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    if a != b and b != c and c != a:
        new_a, new_b, new_c = a, b+random_draw(sigma), c
    elif a != b and b == c:
        new_a, new_b = a, b+random_draw(sigma)
        new_c = new_b
    elif a != b and a == c:
        new_a, new_b = a, b+random_draw(sigma)
        new_c = new_a
    elif a == b and a != c:
        new_b, new_c = b+random_draw(sigma), c
        new_a = new_b
    else:
        new_b = b+random_draw(sigma)
        new_a = new_b
        new_c = new_b

    # scaled_atoms = True
    new_atoms.set_cell([new_a, new_b, new_c, alpha, beta, gamma], scale_atoms=True)
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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms 

def generate_cell_length_c_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1):
    
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])
    print("molecule1_vectors: ", molecule1_vectors)

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    if a != b and b != c and c != a:
       # new_a, new_b, new_c = a, b, c+random_draw(sigma)
       new_a, new_b, new_c = a, b, c+1
       print("new_c: ", new_c)  
    elif a != b and b == c:
        new_a, new_c = a, c+random_draw(sigma)
        new_b = new_c
    elif a != b and a == c:
        new_c, new_b = c+random_draw(sigma), b
        new_a = new_c
    elif a == b and a != c:
        new_b, new_c = b, c+random_draw(sigma)
        new_a = new_b
    else:
        new_c = c+random_draw(sigma)
        new_a = new_c
        new_b = new_c

    # scaled_atoms = True
    new_atoms.set_cell([new_a, new_b, new_c, alpha, beta, gamma], scale_atoms=True)
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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms 

def generate_cell_length_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1):
    
    """
    need fix when a = b or a = b = c
    """
    # get names and molecule idx
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    if a != b and b != c and c != a:
        new_a, new_b, new_c = a+random_draw(sigma), b+random_draw(sigma), c+random_draw(sigma)
    elif a != b and b == c:
        new_a, new_b = a+random_draw(sigma), b+random_draw(sigma) 
        new_c = new_b
    elif a != b and a == c:
        new_a, new_b = a+random_draw(sigma), b+random_draw(sigma)
        new_c = new_a
    elif a == b and a != c:
        new_a, new_c = a+random_draw(sigma), c+random_draw(sigma)
        new_b = new_a
    else:
        new_a = a+random_draw(sigma)
        new_b = new_a
        new_c = new_a

    # scaled_atoms = True
    new_atoms.set_cell([new_a, new_b, new_c, alpha, beta, gamma], scale_atoms=True)
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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms    

def generate_cell_angle_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1): 

    # get names and molecule idx
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    new_cell_params = [a, b, c]
    for angle in [alpha, beta, gamma]:
        if angle != 90:
            new_cell_params.append(angle+random_draw(sigma))
        else:
            new_cell_params.append(angle)

    new_atoms.set_cell(new_cell_params, scale_atoms=True)

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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms    

def generate_cell_angle_alpha_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1): 

    # get names and molecule idx
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    new_cell_params = [a, b, c, alpha + random_draw(sigma), beta, gamma]
    new_atoms.set_cell(new_cell_params, scale_atoms=True)

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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms    

def generate_cell_angle_beta_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1): 

    # get names and molecule idx
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    new_cell_params = [a, b, c, alpha, beta + random_draw(sigma), gamma]
    new_atoms.set_cell(new_cell_params, scale_atoms=True)

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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms    

def generate_cell_angle_gamma_frame(atoms, molecule1_atoms_index, space_group, sigma=0.1): 

    # get names and molecule idx
    name = get_name(atoms.get_atomic_numbers())
    st = ase_atoms_to_structure(atoms)
    my_atoms = atoms.copy()
    my_atoms.set_positions(st.getXYZ())
    molecules = molecule_lists(my_atoms)
    old_positions = my_atoms.get_positions()
    molecule1_vectors = np.array([old_positions[i] for i in molecules[0]])

    # set the perturbation
    new_atoms = my_atoms.copy()
    a, b, c, alpha, beta, gamma = my_atoms.get_cell_lengths_and_angles()
    new_cell_params = [a, b, c, alpha, beta, gamma + random_draw(sigma)]
    new_atoms.set_cell(new_cell_params, scale_atoms=True)

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

    ret_atoms =  Atoms(name,
                    positions=new_positions,
                    cell=new_atoms.get_cell(),
                    pbc=[1,1,1])

    return ret_atoms    

def generate_perturb(atoms_input):
    """
    For debug use
    """
    ret_atoms_cell_length_a = generate_cell_length_a_frame(atoms_input, molecule1_atoms_index, space_group, 10.0)
    print("after cell length a", molecule_lists(ret_atoms_cell_length_a))
    ret_atoms_cell_length_b = generate_cell_length_b_frame(ret_atoms_cell_length_a, molecule1_atoms_index, space_group, 10.0)
    print("after cell length b", molecule_lists(ret_atoms_cell_length_b))
    ret_atoms_cell_length_c = generate_cell_length_c_frame(ret_atoms_cell_length_b, molecule1_atoms_index, space_group, 10.0)
    print("after cell length b", molecule_lists(ret_atoms_cell_length_c))

    ret_atoms_cell_angle_beta = generate_cell_angle_beta_frame(ret_atoms_cell_length_c, molecule1_atoms_index, space_group, 5.0)
    print("after cell angle ", molecule_lists(ret_atoms_cell_angle_beta))

    ret_atoms_rotation = generate_rotation_frame(ret_atoms_cell_angle_beta, molecule1_atoms_index, space_group, 10.0)
    print("after rotation ", molecule_lists(ret_atoms_rotation))

    ret_atoms_translate = generate_translate_frame(ret_atoms_rotation, molecule1_atoms_index, space_group, 2.0)
    print("after translation ", molecule_lists(ret_atoms_translate))
    ret_atoms = ret_atoms_translate.copy()
    
    return ret_atoms

if __name__ == "__main__": 

    print("runing main function for debugging...")
    atoms = ase_io.read(sys.argv[1])
    spacegroup_number = sys.argv[2]
    molecule1_atoms_index = molecule_lists(atoms)[0] 
    print(molecule_lists(atoms))
    print("molecule1_atoms_index: ", molecule1_atoms_index)
    if spacegroup_number:
        space_group = Spacegroup(int(spacegroup_number))
    else:
        space_group = spacegroup.get_spacegroup(atoms)
    print("before perturbation: ", space_group)


    ret_atoms = generate_cell_length_c_frame(atoms, molecule1_atoms_index, space_group, 10.0)
    # first enlarge cell and then change the anlge.
    #ret_atoms = ret_atoms_translate.copy()
    # atoms_input = atoms.copy()
    # ret_atoms = generate_perturb(atoms_input)
    # dummy input
#    ret_atoms = Atoms('Au',
#             positions=[[0, 10 / 2, 10 / 2]],
#             cell=[10, 10, 10],
#             pbc=[1, 0, 0])

#    counter = 1 
#    while(len(molecule_lists(atoms)) != len(molecule_lists(ret_atoms))):
#        print("attemp %d "%counter)
#        counter += 1
#        atoms_input = atoms.copy()
#        try:
#            ret_atoms = generate_perturb(atoms_input)
#        except:
#            print("failed during generate_perturb")
#            continue

    space_group = spacegroup.get_spacegroup(ret_atoms)
    print("after perturbation: ", space_group)
    print(molecule_lists(ret_atoms))
    outfile = os.path.splitext(sys.argv[1])[0] + "_perturbed.cif"
    ase_io.write(outfile, ret_atoms)


