from ase import io as ase_io  
from generate_new_frame_update import generate_translate_frame, \
generate_rotation_frame, generate_cell_length_a_frame, generate_cell_length_b_frame, \
generate_cell_length_c_frame, generate_cell_angle_alpha_frame, generate_cell_angle_beta_frame, \
generate_cell_angle_gamma_frame
from atoms2st import atom_belong_to_mol1, ase_atoms_to_structure, molecule_lists
import sys
from ase import spacegroup
from ase.spacegroup import Spacegroup

input_atoms = ase_io.read(sys.argv[1])
step_size = 0.07
space_group = Spacegroup(14)
molecule1_in_cell = []
for i in range(1, input_atoms.get_number_of_atoms()):
    if atom_belong_to_mol1(i, input_atoms):
        molecule1_in_cell.append(i)

for perturbation in ['1','2','3','4','5','6','7','8']:
    atoms = input_atoms.copy()
    new_atoms = generate_translate_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "translate_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_rotation_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "rotation_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_cell_length_a_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "length_a_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_cell_length_b_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "length_b_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_cell_length_c_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "length_c_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_cell_angle_alpha_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "angle_alpha_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_cell_angle_beta_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "angle_beta_frame.cif"
    ase_io.write(outfile, new_atoms) 
    atoms = input_atoms.copy()
    new_atoms = generate_cell_angle_gamma_frame(atoms, molecule1_in_cell, space_group, step_size)
    outfile = "angle_gamma_frame.cif"
    ase_io.write(outfile, new_atoms) 



