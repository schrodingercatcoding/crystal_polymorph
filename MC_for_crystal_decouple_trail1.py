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

from generate_new_frame_update import generate_translate_frame, \
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

def Monte_Carlo(atoms, calculator, space_group, step_size, perturbation, counter, molecule1_in_cell=[], MC_steps = 1000, debug=False):
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
    #MC_trace[counter] = {}
    #MC_trace[counter]['a'] = a
    #MC_trace[counter]['b'] = b
    #MC_trace[counter]['c'] = c
    #MC_trace[counter]['alpha'] = alpha
    #MC_trace[counter]['beta'] = beta
    #MC_trace[counter]['gamma'] = gamma
    #MC_trace[counter]['xyz'] = xyz.tolist()
    #MC_trace[counter]['fraction'] = fraction.tolist()
    #MC_trace[counter]['density'] = len(atoms)/atoms.get_volume()

    old_atoms = atoms.copy()
    old_atoms.set_calculator(calculator)
    old_E = old_atoms.get_total_energy()
    #MC_trace[counter]['Energy'] = old_E

    try:
        if perturbation == "translate":
            new_atoms = generate_translate_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "rotation":
            new_atoms = generate_rotation_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_length_a":
            new_atoms = generate_cell_length_a_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_length_b":
            new_atoms = generate_cell_length_b_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_length_c":
            new_atoms = generate_cell_length_c_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_angle_alpha":
            new_atoms = generate_cell_angle_alpha_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_angle_beta":
            new_atoms = generate_cell_angle_beta_frame(atoms, molecule1_in_cell, space_group, step_size)
        elif perturbation == "cell_angle_gamma":
            new_atoms = generate_cell_angle_gamma_frame(atoms, molecule1_in_cell, space_group, step_size)

        if(len(molecule_lists(new_atoms)) != len(molecule_lists(old_atoms))):
            print("%s move by %f to Frame %d Rejected due to clash!"%(perturbation, step_size, counter))
            outfile = "Rejected_clash_%d.cif"%counter
            ase_io.write(outfile, new_atoms)
            return old_atoms, "Rejected"
    except:
        print("failed during generation of frame %d by doing %s change"%(counter, perturbation))
        outfile = "Failed_frame_%d.cif"%counter
        ase_io.write(outfile, old_atoms)
        return old_atoms, "Rejected"

    new_atoms.set_calculator(calculator)
    new_E = new_atoms.get_total_energy()     

    if new_E < old_E:
        print("Frame %d Accepted!"%counter)
        if counter % 100 == 0 or MC_steps - counter < 100 or debug:
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
            if counter % 100 == 0 or MC_steps - counter < 100 or debug:
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

#    translation_stats = []
#    rotation_stats = []
#    cell_length_a_stats = []
#    cell_length_b_stats = []
#    cell_length_c_stats = []
#    cell_angle_alpha_stats = []
#    cell_angle_beta_stats = []
#    cell_angle_gamma_stats = []
    counter = 0
    atoms_input = atoms.copy()
    while(counter <= MC_stpes):
        if isinstance(perturbation_type, list):
            # random_number = random.choice([1,2,3,4,5,6,7,8])
            random_choices = get_random_choices([f1,f2,f3,f4,f5,f6,f7,f8])
            random_number = random.choice(random_choices)
        else:
            random_number = perturbation_dict[perturbation_type]
        if random_number == 1:
            # translation
            atoms_input.set_calculator(calculator) 
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f1, 'translate', counter, molecule1_in_cell, MC_stpes, args.debug)
#            translation_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 2:
            # rotation
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f2, 'rotation', counter, molecule1_in_cell, MC_stpes, args.debug)
#            rotation_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 3:
            # cell_length a
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f3, 'cell_length_a', counter, molecule1_in_cell, MC_stpes, args.debug)
#            cell_length_a_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 4:
            # cell_length b
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f4, 'cell_length_b', counter, molecule1_in_cell, MC_stpes, args.debug)
#            cell_length_b_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 5:
            # cell_length c
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f5, 'cell_length_c', counter, molecule1_in_cell, MC_stpes, args.debug)
#            cell_length_c_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 6:
            # cell_angle alpha
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f6, 'cell_angle_alpha', counter, molecule1_in_cell, MC_stpes, args.debug)
#            cell_angle_alpha_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 7:
            # cell_angle beta
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f7, 'cell_angle_beta', counter, molecule1_in_cell, MC_stpes, args.debug)
#            cell_angle_beta_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        elif random_number == 8:
            # cell_angle gamma
            atoms_input.set_calculator(calculator)
            ret_atoms, message = Monte_Carlo(atoms_input, calculator, space_group, f8, 'cell_angle_gamma', counter, molecule1_in_cell, MC_stpes, args.debug)
#            cell_angle_gamma_stats.append(message)
            counter += 1
            atoms_input = ret_atoms.copy()
        #if counter % 1000 == 0:
        #    with open("MC_trace_%d.json"%counter, "w") as fh:
        #        json.dump(MC_trace, fh, indent=4)

    #with open("MC_trace.json", "w") as fh:
    #    json.dump(MC_trace, fh, indent=4)
    #print("MC trace are recorded in: MC_trace.json")

#    if len(translation_stats):
#        print("Acceptance ratio of translation move is %f"%(countX("Accepted", translation_stats)/len(translation_stats)))
#    if len(rotation_stats):
#        print("Acceptance ratio of rotation move is %f"%(countX("Accepted", rotation_stats)/len(rotation_stats)))
#    if len(cell_length_a_stats):
#        print("Acceptance ratio of cell length a move is %f"%(countX("Accepted", cell_length_a_stats)/len(cell_length_a_stats)))
#    if len(cell_length_b_stats):
#        print("Acceptance ratio of cell length b move is %f"%(countX("Accepted", cell_length_b_stats)/len(cell_length_b_stats)))
#    if len(cell_length_c_stats):
#        print("Acceptance ratio of cell length c move is %f"%(countX("Accepted", cell_length_c_stats)/len(cell_length_c_stats)))
#    if len(cell_angle_alpha_stats):
#        print("Acceptance ratio of cell angle alpha move is %f"%(countX("Accepted", cell_angle_alpha_stats)/len(cell_angle_alpha_stats)))
#    if len(cell_angle_beta_stats):
#        print("Acceptance ratio of cell angle beta move is %f"%(countX("Accepted", cell_angle_beta_stats)/len(cell_angle_beta_stats)))
#    if len(cell_angle_gamma_stats):
#        print("Acceptance ratio of cell angle gamma move is %f"%(countX("Accepted", cell_angle_gamma_stats)/len(cell_angle_gamma_stats)))

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

