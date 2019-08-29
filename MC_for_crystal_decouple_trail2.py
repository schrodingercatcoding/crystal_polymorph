import os
import argparse
import numpy as  np
import sys
import random
import math
import time
import json, codecs

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
from atoms2st import atom_belong_to_mol1, ase_atoms_to_structure, molecule_lists

from generate_new_frame_update_trail2 import generate_translate_frame, \
generate_rotation_frame, generate_cell_length_a_frame, generate_cell_length_b_frame, \
generate_cell_length_c_frame, generate_cell_angle_alpha_frame, generate_cell_angle_beta_frame, \
generate_cell_angle_gamma_frame
from atoms2st import atom_belong_to_mol1

TORCHANI = "torchani"
AES_ANI = "aes_ani"
KHAN = "khan"
IMPLEMENTATIONS = [
    TORCHANI,
    AES_ANI,
    KHAN,
]

perturbation_dict = {
    "translation": 1,
    "rotation": 2,
    "cell_length_a": 3,
    "cell_length_b":4,
    "cell_length_c":5,
    "cell_angle_alpha": 6,
    "cell_angle_beta": 7,
    "cell_angle_gamma":8
}

# for storing MC information
#MC_trace = {}

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
        dest='MC_f1_f2_f3_f4_f5_f6_f7_f8',
        metavar='<Xi>',
        default=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        nargs=8,
        type=float,
        help='MC step size for translate, rotatioin, cell length(a,b,c), and cell angle(alpha, beta, gamma), unit(A or degree)'
    )

    parser.add_argument(
        '-perturbation_type',
        choices=['translation', 'rotation', 'cell_length_a', 'cell_length_b', 'cell_length_c', 'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma'],
        type=str,
        default=['translation', 'rotation', 'cell_length_a', 'cell_length_b', 'cell_length_c', 'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma'],
        help='choose one type of perturbation, by default run all types randomly'
    )

    parser.add_argument(
        '-debug',
        default=False,
        dest='debug',
        action='store_true',
        help='debug option to print out all the accepted frames'
    )   

    return parser.parse_args()

def Monte_Carlo(atoms, calculator, space_group, step_size, perturbation, counter, numb_molecules, molecule1_in_cell=[], MC_steps = 1000, debug=False):
    """
    A Monte_Carlo function that takes in an atoms instance and generate a new atoms instance base on metropolis monte carlo

    :type  atoms: Atoms.atoms
    :param atoms: ASE atoms instance which contains a collection of atoms and their related information

    :type  calculator: ase.calculator
    :param calculator: ANI calculator which is used to get force and energy base on atomic numbers and positions

    :type  space_group: ase.spacegroup.Spacegroup
    :param space_group: the space group that MC is performing under

    :type  step_size: array
    :param step_size: step size for T, R, cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma

    :type  perturbation: str
    :param perturbation: the type of perturbation that MC is performing under

    :type  counter: int
    :param counter: the number of MC steps

    :type  numb_molecules: int
    :param numb_molecules: number of molecules in the input cif file

    :type  molecule1_in_cell: array
    :param molecule1_in_cell: atom indexs for the asymmetric molecule in cell

    :type  MC_steps: int
    :param MC_steps: the number of MC steps that will be performed

    :type  debug: bool
    :param debug: show debug information including all accepted and rejected moves

    return a new frame base on metropolis monte carlo shown as below:

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

    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    xyz = atoms.get_positions()
    fraction = atoms.get_scaled_positions()
    old_E = atoms.get_total_energy()

    try:
        if perturbation == "translate":
            generate_translate_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "rotation":
            generate_rotation_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_length_a":
            generate_cell_length_a_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_length_b":
            generate_cell_length_b_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_length_c":
            generate_cell_length_c_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_angle_alpha":
            generate_cell_angle_alpha_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_angle_beta":
            generate_cell_angle_beta_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_angle_gamma":
            generate_cell_angle_gamma_frame(atoms, molecule1_in_cell, space_group, step_size)

        if(len(molecule_lists(atoms)) != numb_molecules:
            print("%s move by %f to Frame %d Rejected due to clash!"%(perturbation, step_size, counter))
            outfile = "Rejected_clash_%d.cif"%counter
            ase_io.write(outfile, atoms)
    except:
        print("failed during generation of frame %d by doing %s change"%(counter, perturbation))
        outfile = "Failed_frame_%d.cif"%counter
        ase_io.write(outfile, atoms)
        return atoms, "Rejected"

    new_E = atoms.get_total_energy()     

    if new_E < old_E:
        print("Frame %d Accepted!"%counter)
        if counter % 100 == 0 or MC_steps - counter < 100 or debug:
            outfile = "Accepted_%d.cif"%counter
            ase_io.write(outfile, atoms)
    else:
        delta_Emn = new_E - old_E
        delta_Emn = delta_Emn * 23 # ev to kcal/mol
        p = math.exp(-delta_Emn/units.kB/300)
        nu = random.uniform(0, 1)
        if p > nu:
            print("Frame %d Accepted!"%counter)
            if counter % 100 == 0 or MC_steps - counter < 100 or debug:
                outfile = "Accepted_%d.cif"%counter
                ase_io.write(outfile, atoms)
        else:
            print("Frame %d Rejected!"%counter)

def countX(ele, arr):
    """
    return the occurance of element in array
    """
    counter = 0
    for el in arr:
        if el == ele:
            counter += 1

    return counter

def get_random_choices(arr):
    """
    :type  arr: array
    :param arr: [f1, f2, f3, f4, f5, f6, f7, f8]

    return an array of random numbers base on perturbation type
    """

    choice_arr = [1,2,3,4,5,6,7,8]
    ret_arr = []
    for f, value in zip(arr, choice_arr):
        if f != 0:
            ret_arr.append(value)

    return ret_arr

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

    print('debug? ', args.debug)
    atoms = ase_io.read(args.cif_file)
    numb_molecules = len(molecule_lists(atoms))
    print('number of molecules: ', numb_molecules)
    MC_stpes = args.MC
    space_group = Spacegroup(args.space_group)
    # step size in MC
    f1, f2, f3, f4, f5, f6, f7, f8 =  args.MC_f1_f2_f3_f4_f5_f6_f7_f8 
    perturbation_type = args.perturbation_type
    molecule1_in_cell = []
    for i in range(1, atoms.get_number_of_atoms()):
        if atom_belong_to_mol1(i, atoms):
            molecule1_in_cell.append(i)
    print("initial unit cell")
    print(atoms.cell)
    print("To do Monte Carlo for %d step."%MC_stpes)
    print("Molecule index in ASU are: ", molecule1_in_cell)

    counter = 0
    atoms.set_calculator(calculator)
    while(counter <= MC_stpes):
        if isinstance(perturbation_type, list):
            # random_number = random.choice([1,2,3,4,5,6,7,8])
            random_choices = get_random_choices([f1,f2,f3,f4,f5,f6,f7,f8])
            random_number = random.choice(random_choices)
        else:
            random_number = perturbation_dict[perturbation_type]
        if random_number == 1:
            # translation
            Monte_Carlo(atoms, calculator, space_group, f1, 'translate', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 2:
            # rotation
            Monte_Carlo(atoms, calculator, space_group, f2, 'rotation', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 3:
            # cell_length a
            Monte_Carlo(atoms, calculator, space_group, f3, 'cell_length_a', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 4:
            # cell_length b
            Monte_Carlo(atoms, calculator, space_group, f4, 'cell_length_b', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 5:
            # cell_length c
            Monte_Carlo(atoms, calculator, space_group, f5, 'cell_length_c', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 6:
            # cell_angle alpha
            Monte_Carlo(atoms, calculator, space_group, f6, 'cell_angle_alpha', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 7:
            # cell_angle beta
            Monte_Carlo(atoms, calculator, space_group, f7, 'cell_angle_beta', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1
        elif random_number == 8:
            # cell_angle gamma
            Monte_Carlo(atoms, calculator, space_group, f8, 'cell_angle_gamma', counter, numb_molecules, molecule1_in_cell, MC_stpes, args.debug)
            counter += 1

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

