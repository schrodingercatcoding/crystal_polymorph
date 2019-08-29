import os
import argparse
import numpy as  np
import sys
import random
import math

from utils import models, single_point_energy, optimize_molecule

from ase import io as ase_io
from ase import spacegroup
from ase.md import langevin
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal

from schrodinger.structutils.analyze import center_of_mass 

from perturb_molecule import get_new_frame 
from atoms2st import atom_belong_to_mol1

TORCHANI = "torchani"
AES_ANI = "aes_ani"
KHAN = "khan"
IMPLEMENTATIONS = [
    TORCHANI,
    AES_ANI,
    KHAN,
]

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
        type=str,
        required=False,
        help='running Monte Carlo for n steps'
    )

    return parser.parse_args()

def random_draw(x):
    """
    generate a perturb value for x
    """
    mu = 0
    sigma = 0.5
    bottom = 0
    top = 0.1 * x

    a = random.gauss(mu,sigma)
    while (bottom <= a <= top) == False:
        a = random.gauss(mu,sigma)
    return a

def generate_new_frame(atoms):
    """
    atoms: atoms class in ASE module

    F*(ri + A*L)
    return a new structure by randomly perturb the deformation matrix F
    """
    new_atoms = atoms.copy()
    F = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    delta_a, delta_b, delta_c = random_draw(1), random_draw(1), random_draw(1)
    new_F = F + np.matrix([[delta_a, 0, 0],[0, delta_b, 0],[0, 0, delta_c]])
    cell = atoms.get_cell()
    xyz = atoms.get_positions()
    new_cell = np.matmul(cell, new_F)
    #new_xyz = np.matmul(xyz, new_F)
    atoms.translate([dx,dy,dx])


    new_atoms.set_positions(new_xyz)
    new_atoms.set_cell(new_cell)

    return new_atoms

def generate_new_frame_2(atoms, atom_index_group1):
    """
    atoms: atoms class in ASE module

    return a new structure that has a,b,c,alpha,beta,gamma perturbed
    and the center of mass of asymmetric unit to the origin changed
    """

    delta_x, delta_y, delta_z = random_draw(0.001), random_draw(0.001), random_draw(0.001)
    delta_theta_x = random_draw(0.1)
    delta_theta_y = random_draw(0.1)
    delta_theta_z = random_draw(0.1)
    return get_new_frame(atoms, atom_index_group1, delta_x, delta_y, delta_z, delta_theta_x, delta_theta_y, delta_theta_z)


def Monte_Carlo(atoms, atom_index_group1, calculator):
    """
    atoms: atoms class in ASE module

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
    old_frame = atoms.copy()
    old_frame.set_calculator(calculator)
    old_E = old_frame.get_total_energy()
    #new_frame = generate_new_frame(old_frame)
    new_frame = generate_new_frame_2(old_frame, atom_index_group1)
    new_frame.set_calculator(calculator)
    new_E = new_frame.get_total_energy()

    if new_E < old_E:
        print("Accepted!")
        return new_frame, "Accepted"
    else:
        delta_Emn = new_E - old_E
        delta_Emn = delta_Emn * mol / kcal
        p = math.exp(-delta_Emn/units.kB/500)
        nu = random.uniform(0, 1)
        if p > nu:
            print("Accepted!")
            return new_frame, "Accepted"
        else:
            print("Rejected!")
            return old_frame, "Rejected"

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
    print("Space group of crystal: %s" % spacegroup.get_spacegroup(atoms))
    print("initial unit cell")
    print(atoms.cell)

    atom_index_group1 = []
    
    for i in range(1, atoms.get_number_of_atoms()): 
        if atom_belong_to_mol1(i, atoms):
            atom_index_group1.append(i) 

    MC_steps = int(args.MC)
    all_frames = 'all_frames.xyz'
    for i in range(0,int(MC_steps)):
        atoms.set_calculator(calculator)
        old_frame = atoms.copy()
        old_frame.set_calculator(calculator)
        new_frame, message = Monte_Carlo(old_frame, atom_index_group1, calculator)
        print("MC step %d"%i)
        if message == 'Accepted':
            outfile = "Accepted_%d.cif"%i
            ase_io.write(all_frames, atoms, append=True)
            ase_io.write(outfile, atoms)
        atoms = new_frame.copy()

if __name__ == "__main__":
    main()
