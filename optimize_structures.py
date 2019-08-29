import os
import argparse
import numpy as  np
import sys
from collections import defaultdict

from utils import models, single_point_energy, optimize_molecule, molecular_dynamics

from ase import io as ase_io
from ase import spacegroup

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

def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="optimize structures with ANI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        'mae_file',
         type=str,
         help='name of mae file',
    )

    parser.add_argument(
        '-network-type',
        type=str,
        choices=models.keys(),
        default="ani-1x",
    )

    parser.add_argument(
        '-implementation',
        type=str,
        choices=IMPLEMENTATIONS,
        default=KHAN,
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
        help='optimize'
    )

    parser.add_argument(
        '-numb-networks',
        default=8,
        type=int,
        help='number of committee members to load'
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

    strs = list(StructureReader(args.mae_file))
    cvgd = optimize_structures(strs, calculator.committee, maxit=500, thresh=0.01)

    rd = rawdataset_from_st(strs)
    energy_data = calculator.committee.compute_data(rd)
    for st, data in zip(strs, energy_data):
        st.property["r_j_ANI_ENERGY"] = data.energy
        st.property["r_j_ANI_RHO"] = data.rho
    
    base, ext = os.path.splitext(args.mae_file)
    outfile = base + "_optimized" + ext
    print("final structure written to file: %s" % outfile)
    with StructureWriter(outfile) as writer:
        writer.extend(strs)

if __name__ == "__main__":
    main()
