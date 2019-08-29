import os
import argparse
import numpy as  np
import sys
import random
import math
import time

from utils import models, single_point_energy, optimize_molecule

from ase import io as ase_io
from ase import spacegroup
from ase.spacegroup import Spacegroup
from ase.md import langevin
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

from schrodinger.structutils.analyze import center_of_mass 

from generate_new_frame import generate_translate_frame, \
generate_rotation_frame, generate_cell_length_frame, generate_cell_angle_frame 
from atoms2st import atom_belong_to_mol1

TORCHANI = "torchani"
AES_ANI = "aes_ani"
KHAN = "khan"
IMPLEMENTATIONS = [
    TORCHANI,
    AES_ANI,
    KHAN,
]

atom_total = 0

def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="optimize a crystal structure from a cif file with ANI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        'cif_file',
         type=str,
         help='name of cif file (must end in .cif)'
    )

    parser.add_argument(
        '-network-type',
        type=str,
        choices=models.keys(),
        required=True
    )

    parser.add_argument(
        '-implementation',
        type=str,
        choices=IMPLEMENTATIONS,
        default=TORCHANI,
    )

    parser.add_argument(
       '-khan-network',
       type=str,
       default=None,
       help='khan trained network'
    )

    parser.add_argument(
        '-numb-networks',
        default=8,
        type=int,
        help='number of committee members to load'
    )

    parser.add_argument(
        '-MC',
        type=int,
        required=False,
        help='running Monte Carlo for n steps'
    )

    parser.add_argument(
        '-space_group',
        default=1,
        type=int,
        help='under which space group you would like to run, by default it is P1'
    )

    parser.add_argument(
        '-step_size',
        dest='MC_f1_f2_f3_f4',
        metavar='<Xi>',
        default=(0.1, 0.1, 0.1, 0.1),
        nargs=4,
        type=float,
        help='MC step size for translate, rotatioin, cell length, and cell angle, unit(A or degree)'
    )

    return parser.parse_args()

def Monte_Carlo(atoms, calculator, space_group, step_size, perturbation, counter, molecule1_in_cell=[], MC_steps = 1000):
    """
    atoms: atoms class in ASE module
    perturbation: str represent cell and xyz changes
    counter: int for writing files

    return a new frame base on:

    if new_E < old_E:
        accepted, continue MD with new frame
    elif new_E > old_E:
        delta_Emn = new_E - old_E
        p = exp(-delta_Emn/kBT)
        generate a random number nu(0,1)
        if p > nu:
            accepted, continue MD with new frame
        else:
            rejected, continue MD with old frame
    """
    old_atoms = atoms.copy()
    old_atoms.set_calculator(calculator)
    old_E = atoms.get_total_energy()

    if perturbation == "translate":
        new_atoms = generate_translate_frame(atoms, molecule1_in_cell, space_group, step_size)
    elif perturbation == "rotation":
        new_atoms = generate_rotation_frame(atoms, molecule1_in_cell, space_group, step_size)
    elif perturbation == "cell_length":
        new_atoms = generate_cell_length_frame(atoms, molecule1_in_cell, space_group, step_size)
    else:
        new_atoms = generate_cell_angle_frame(atoms, molecule1_in_cell, space_group, step_size)

    if(len(new_atoms.get_atomic_numbers()) !== atom_total){
        print("Frame %d Rejected! due to clash"%counter)
        return old_atoms, "Rejected"
    }
    new_atoms.set_calculator(calculator)
    new_E = new_atoms.get_total_energy()        

    #print("old_E and new_E: ", old_E, new_E)
    if new_E < old_E:
        print("Frame %d Accepted!"%counter)
        if counter % 100 == 0 or MC_steps - counter < 100:
            outfile = "Accepted_%d.cif"%counter
            ase_io.write(outfile, new_atoms)
        return new_atoms, "Accepted"
    else:
        delta_Emn = new_E - old_E
        delta_Emn = delta_Emn * 23 # ev to kcal/mol
        p = math.exp(-delta_Emn/units.kB/300)
        nu = random.uniform(0, 1)
        if p > nu:
            print("Frame %d Accepted!"%counter)
            if counter % 100 == 0 or MC_steps - counter < 100:
                outfile = "Accepted_%d.cif"%counter
                ase_io.write(outfile, new_atoms)
            return new_atoms, "Accepted"
        else:
            print("Frame %d Rejected!"%counter)
            return old_atoms, "Rejected"

def countX(ele, arr):
    """
    return the occurance of element in array
    """
    counter = 0
    for el in arr:
        if el == ele:
            counter += 1

    return counter

def main():
    
    args = parse_args()
    print("args", args)

    if args.implementation == TORCHANI:
        from torchani_calculator import torchani_calculator
        calculator = torchani_calculator(args.network_type)
    elif args.implementation == AES_ANI:
        from ani_ase import ani_ase_calculator
        calculator = ani_ase_calculator(args.network_type)
    elif args.implementation == KHAN:
        from khan_calculator import khan_calculator
        calculator = khan_calculator(
            args.network_type, args.khan_network, args.numb_networks)

    assert args.cif_file.endswith('.cif')

    atoms = ase_io.read(args.cif_file)
    number_of_atoms = len(atoms.get_atomic_numbers())
    atom_total = number_of_atoms
    MC_stpes = args.MC
    space_group = Spacegroup(args.space_group)
    # step size in MC
    f1, f2, f3, f4 =  args.MC_f1_f2_f3_f4 
    
    molecule1_in_cell = []
    for i in range(1, atoms.get_number_of_atoms()):
        if atom_belong_to_mol1(i, atoms):
            molecule1_in_cell.append(i)
    # print("Space group of crystal: %s" % spacegroup.get_spacegroup(atoms))
    print("initial unit cell")
    print(atoms.cell)
    print("To do Monte Carlo for %d step."%MC_stpes)
    print("Molecule index in ASU are: ", molecule1_in_cell)

    translation_stats = []
    rotation_stats = []
    cell_length_stats = []
    cell_angle_stats = []
    counter = 0
    atoms_input = atoms.copy()
    # space_group = spacegroup.get_spacegroup(atoms) 
    while(counter <= MC_stpes):
        random_number = random.choice([1,2,3,4])
        if random_number == 1:
            # translation
            atoms_input.set_calculator(calculator) 
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f1, 'translate', counter, molecule1_in_cell, MC_stpes)
            translation_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 2:
            # rotation
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f2, 'rotation', counter, molecule1_in_cell, MC_stpes)
            rotation_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 3:
            # cell_length
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f3, 'cell_length', counter, [], MC_stpes)
            cell_length_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        else:
            # cell_angle
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f4, 'cell_angle', counter, [], MC_stpes)
            cell_angle_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
    print("Acceptance ratio of translation move is %f"%(countX("Accepted", translation_stats)/len(translation_stats)))
    print("Acceptance ratio of rotation move is %f"%(countX("Accepted", rotation_stats)/len(rotation_stats)))
    print("Acceptance ratio of cell length move is %f"%(countX("Accepted", cell_length_stats)/len(cell_length_stats)))
    print("Acceptance ratio of cell angle move is %f"%(countX("Accepted", cell_angle_stats)/len(cell_angle_stats)))

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

