import os
import argparse
import numpy as  np
import ase
import sys

from utils import models, single_point_energy, optimize_molecule
from utils import KCAL
from data_utils import read_json_data, khan_molec_to_ase_atoms
CCSD = "CCSD_T_CBS_MP2"


TORCHANI = "torchani"
AES_ANI = "aes_ani"
KHAN = "khan"
IMPLEMENTATIONS = [
    TORCHANI,
    AES_ANI,
    KHAN
]

def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="Run ani on a testset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        'datadir',
         type=str,
         help='directory holding json files to use as ref data'
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
        '-optimize',
        action='store_true',
        help='optimize molecule prior to computing energy'
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
        calculator = torchani_calculator(args.network_type, args.numb_networks)
    elif args.implementation == AES_ANI:
        from ani_ase import ani_ase_calculator
        calculator = ani_ase_calculator(args.network_type, args.numb_networks)
    elif args.implementation == KHAN:
        from khan_calculator import khan_calculator
        calculator = khan_calculator(args.network_type, None, args.numb_networks)

    data = read_json_data(args.datadir, remove_atomic_energy=False, energy_cutoff=1.0e10)

    all_abs_errors = []
    all_conf_errors = []
    for k in data:
        json_data = data[k]
        X = [np.array(carts) for carts in json_data["X"]]
        #y = [float(ener) for ener in json_data["energy"]]
        y = [float(ener) for ener in json_data["Y"]]

        n_geoms = len(y)

        molecules = khan_molec_to_ase_atoms(X)
        energies = []
        refs = []
        print("data for input", k)
        for i, m in enumerate(molecules):

            #if not all(method == CCSD for method in json_data["min_method"]):
            #    print(json_data["min_method"])
            #    raise ValueError("what the hell")
            
            if args.optimize:
                indices = list(map(int, json_data["scan_coord"][i]))
                value = float(json_data["scan_value"][i]) * np.pi / 180.0
                torsion = [value, indices] 
                energy = optimize_molecule(m, calculator, [torsion])
            else:
                energy = single_point_energy(m, calculator)

            energies.append(energy)
            refs.append(y[i])

            print("ani energy: %.8f ref energy: %.8f" % (energy, y[i]))

        np_energies = np.array(energies)
        np_refs = np.array(refs)

        dE = np_energies - np_refs
        ndata = len(dE)
        abs_error = np.sqrt(np.dot(dE, dE) / ndata)
        all_abs_errors.extend(dE)

        min_ref = 0#np.argmin(np_refs)
        np_energies = np_energies - np_energies[min_ref]
        np_refs = np_refs - np_refs[min_ref]
        dE = np_energies - np_refs
        dE = dE[1:]
        conf_error = np.sqrt(np.dot(dE, dE) / (ndata-1))
        mad = np.mean(abs(dE))
        #  skip the one reference
        all_conf_errors.extend(dE)

        print("datset %s: absolute error %.4f relative rmse %.4f relative mad %.4f" % (k, abs_error * KCAL, conf_error * KCAL, mad * KCAL))

    dE = np.array(all_abs_errors)
    ndata = len(dE)
    abs_error = np.sqrt(np.dot(dE, dE) / ndata)

    dE = np.array(all_conf_errors)
    ndata = len(dE)
    conf_error = np.sqrt(np.dot(dE, dE) /ndata)
    mad = np.mean(abs(dE))

    print("Total errors: absolute error: %f rmse relative error %.4f mad relative error %.4f" % (abs_error * KCAL, conf_error*KCAL, mad * KCAL))

if __name__ == "__main__":
    main()

