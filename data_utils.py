import os
import numpy as np
import json
from ase import Atoms
KCAL = 627.509
ENERGY_CUTOFF = 200.0/KCAL

element2id = {"H": 0, "C": 1, "N": 2, "O": 3}
id2element = {0: "H", 1: "C", 2: "N", 3: "O"}

def khan_molec_to_ase_atoms(molecules):
    """
    molecules is a list of lists of atomid x, y, z
    in khan format, this is converted to a list
    of ase Atoms instances
    """
    atoms = []
    for molecule in molecules:
        elements = []
        positions = []
        for atom in molecule:
            elements.append(id2element[atom[0]])
            positions.append(atom[1:])

        atoms.append(
            Atoms("".join(elements), positions=positions)
        )

    return atoms
        

def read_json_data(json_data_dir, remove_atomic_energy=True, energy_cutoff=ENERGY_CUTOFF):
    """
    Read json data and return as a dict 

    Returns dict for examples.
    The keys are filenames and values are lists of
    examples.  Example is a tuple (X, Y) for that reaction
    """

    skipped = 0
    cnt = 0
    reactions = {} 
    for root, dirs, files in os.walk(json_data_dir):
        for fname in files:
            if fname.endswith(".json"):
                json_data = _read_json_data(
                    os.path.join(root, fname),
                    remove_atomic_energy,
                    energy_cutoff
                )
                reactions[fname[:-5]] = json_data

    return reactions

def _read_json_data(fname, remove_atomic_energy, energy_cutoff):
    """
    Read data from json file prepared for QM data
    there are two fields X and Y which hold the molecule definition
    and a total energy respectively.
    """

    with open(fname) as fin:
        data = json.load(fin)

    return data

