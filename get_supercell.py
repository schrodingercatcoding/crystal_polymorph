import os
import argparse
import numpy as  np
import sys
import time
from collections import defaultdict

from utils import models, single_point_energy, optimize_molecule, molecular_dynamics

from ase import io as ase_io
from ase import spacegroup
from ase import Atoms

def build_supercell(atoms, n): 
    """
    build and return a supercell using the atoms
    
    :type n: list of ints
    :param n: number of cell replicates in each of three dimensions
    :return: Atoms
    """

    unit_cell = np.transpose(atoms.get_cell())
    if False in atoms.get_pbc():
        raise RuntimeError("can only make a supercell in a periodic system")

    ntotal = n[0] * n[1] * n[2]
    assert ntotal > 0

    natoms = len(atoms)
    species = list(atoms.symbols)
   
    carts = atoms.get_positions()

    all_species = []
    all_coords = np.zeros((ntotal * natoms, 3))

    print("build supercell")
    print("number of initial atoms", len(atoms))
    print("initial unit cell", unit_cell)
    cnt = 0
    dr = np.zeros(3)
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                all_species.extend(species)
                dr = i * unit_cell[:, 0] + j * unit_cell[:, 1] + k * unit_cell[:, 2]
                for iat in range(natoms):
                    all_coords[natoms*cnt + iat, :] = carts[iat, :] + dr[:]
                cnt += 1

    species = "".join(all_species)
    supercell = Atoms(all_species, positions=all_coords)

    N_mult = np.zeros((3, 3))
    for i in [0, 1, 2]:
        N_mult[i, i] = n[i]

    unit_cell = np.dot(N_mult, unit_cell)
    supercell.set_cell(np.transpose(unit_cell))
    supercell.set_pbc([True, True, True])

    print("setting unit cell")
    print(unit_cell)

    print("atoms in supercell", len(supercell))
    print("unit_cell")
    print(np.transpose(supercell.get_cell()))

    assert len(supercell) == ntotal * natoms

    supercell.wrap(center=[0.0, 0.0, 0.0])

    return supercell

def main():

    atoms = ase_io.read(sys.argv[1])
    ret_atoms = build_supercell(atoms, [5,5,5])
    outfile = os.path.splitext(sys.argv[1])[0] + "_super.cif"
    ase_io.write(outfile, ret_atoms)

if __name__ == "__main__":
    main()
