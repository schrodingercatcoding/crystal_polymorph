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

from schrodinger.structure import StructureReader, StructureWriter
from schrodinger.structutils.analyze import center_of_mass
from sampling.schrodinger_utils import rawdataset_from_st
from khan.ase_interface.utilities import khan_molec_to_ase_atoms

TORCHANI = "torchani"
AES_ANI = "aes_ani"
KHAN = "khan"
IMPLEMENTATIONS = [
    TORCHANI,
    AES_ANI,
    KHAN,
]

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
        '-time-step',
        default=0.1,
        type=float,
        help="time step in fs"
    )

    parser.add_argument(
        '-total-time',
        default=10.0,
        type=float,
        help="total MD time in ps"
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
        '-optimize',
        action='store_true',
        help='optimize crystal prior to computing energy'
    )

    parser.add_argument(
        '-relax-cell',
        action='store_true',
        help='allow cell shape to change when optimizing',
    )

    parser.add_argument(
        '-dynamics',
        action='store_true',
        help='perform molecular dynamics'
    )

    parser.add_argument(
        '-numb-networks',
        default=8,
        type=int,
        help='number of committee members to load'
    )

    parser.add_argument(
        '-truncate-box',
        default=False,
        action="store_true",
        help='truncate a box at 12 angstrom'
    )

    parser.add_argument(
        '-torsion',
        default=None,
        nargs=5,
        help="list of one float and four integers defining a torsion to scan from a value of zero"
    )

    parser.add_argument(
        '-box-length',
        type=float,
        default=None,
        help='if system is not periodic, place it in a box of this length'
    )

    parser.add_argument(
        '-supercell',
        type=int,
        default=None,
        help='make a supercell of this size from initial system'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("args", args)

    if args.implementation == TORCHANI:
        from torchani_calculator import torchani_calculator
        calculator = torchani_calculator(
            args.network_type, args.numb_networks)
    elif args.implementation == AES_ANI:
        from ani_ase import ani_ase_calculator
        calculator = ani_ase_calculator(args.network_type)
    elif args.implementation == KHAN:
        from khan_calculator import khan_calculator
        calculator = khan_calculator(
            args.network_type, args.khan_network, args.numb_networks)

    #assert args.cif_file.endswith('.cif') or args.cif_file.endswith('.xyz')
    if args.cif_file.endswith('.mae'):
        st = next(StructureReader(args.cif_file))

        if args.truncate_box:
            # reduced size box
            a = 12.0
            deletions = []
            for mol in st.molecule:
                rcom = center_of_mass(st, mol.getAtomIndices())
                inbox = [abs(x) < a/2 for x in rcom]
                if False in [abs(x) < a/2 for x in rcom]:
                    deletions.extend(mol.getAtomIndices())
            st.deleteAtoms(deletions)

        rd = rawdataset_from_st([st])
        ase_molecs = khan_molec_to_ase_atoms(rd.all_Xs)
        atoms = ase_molecs[0]

        if args.truncate_box:
            # hard coded box
            #a = 22.6868
            vecs = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
            atoms.set_cell(vecs)
            atoms.set_pbc([True, True, True])
            print("after set cell", atoms.get_pbc())

    if args.cif_file.endswith('.cif'):
        atoms = ase_io.read(args.cif_file)

    if args.box_length is not None:
        atoms.set_cell(
            [
                [args.box_length, 0.0, 0.0],
                [0.0, args.box_length, 0.0],
                [0.0, 0.0, args.box_length],
            ]
        )
        atoms.set_pbc([True, True, True])
                
    if args.supercell is not None:
        atoms = build_supercell(atoms, [args.supercell] * 3)

    periodic = False not in atoms.get_pbc()

    print("cell")
    print(atoms.get_cell())

    if periodic:
        #atoms.wrap()
        space_group = spacegroup.get_spacegroup(atoms)
        print("Space group of crystal: %s" % space_group) 
        print("initial unit cell")
        print(atoms.cell)

        # this shows how to get unique atoms and there equivalents
        #scaled_positions = atoms.get_scaled_positions()
        #unique_positions = space_group.unique_sites(scaled_positions)
        #all_positions, symmetry_map = space_group.equivalent_sites(unique_positions)
        #symmetry_groups = defaultdict(list)
        #for igroup, position in zip(symmetry_map, all_positions):
        #    symmetry_groups[igroup].append(position)

        #print("unique positions")
        #for xyz in unique_positions:
        #    print(xyz)

        #for igroup in sorted(symmetry_groups.keys()):
        #    print("positions in symmetry group %d" % igroup)
        #    for xyz in symmetry_groups[igroup]:
        #        print(xyz)

    torsions = ()
    if args.torsion is not None:
        torsions = [(
            float(args.torsion[0]),
            int(args.torsion[1]),
            int(args.torsion[2]),
            int(args.torsion[3]),
        )]
        
    if args.optimize:
        # cannot optimize cell until we implement a stress tensor
        energy = optimize_molecule(
            atoms,
            calculator,
            torsions=torsions,
            thresh=0.05,
            optimize_cell=args.relax_cell,
        )
    else:
        start_time = time.time()   
        energy = single_point_energy(atoms, calculator)
        print("--- %s seconds ---" % (time.time() - start_time)) 
    if args.dynamics:
        traj = molecular_dynamics(
            atoms,
            calculator,
            total_time=args.total_time,
            time_step=args.time_step
        )
        if args.cif_file.endswith('.mae'):
            st = next(StructureReader(args.cif_file))
            with StructureWriter("md_path.mae") as writer:
                for m in traj:
                    st0 = st.copy()
                    st0.setXYZ(m.get_positions())
                    writer.append(st0)
        else:
            with open("md_path.xyz", "w") as fout:
                xyz_lst = []
                for atoms in traj:
                    xyz_lst.append("%d\n" % len(atoms))
                    species = atoms.get_chemical_symbols()
                    carts = atoms.get_positions()
                    for ch, xyz in zip(species, carts):
                        xyz_lst.append(
                            "%s %.8f %.8f %.8f" % (ch, xyz[0], xyz[1], xyz[2])
                        )

                fout.write("\n".join(xyz_lst))
                    
    print("energy of cell (au):", energy)

    if periodic:
        space_group = spacegroup.get_spacegroup(atoms)
        print("Final Space group of crystal: %s" % space_group) 
        print("energy of cell/volume (au):", energy/atoms.get_volume())
        print("final unit cell")
        print(atoms.cell)

    print("Uncertainty %.4f (kcal/mol)" % (calculator.uncertainty(atoms) * 627.509)) 

    if periodic:
        outfile = os.path.splitext(args.cif_file)[0] + "_optimized.cif"
    else:
        outfile = os.path.splitext(args.cif_file)[0] + "_optimized.xyz"
    ase_io.write(outfile, atoms)
    print("final structure written to file: %s" % outfile)

if __name__ == "__main__":
    main()
