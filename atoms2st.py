from schrodinger.infra import mm
from schrodinger.application.matsci.nano import xtal
from schrodinger.structure import create_new_structure
from schrodinger import structure
from ase import spacegroup   

import ase.io
import sys
import os

def ase_atoms_to_structure(atoms):
    """
    Convert an ase atoms instance into a structure instance
    """
    #print("atoms[0]: ", atoms[0])
#    for i in range(0, atoms.get_number_of_atoms()):
#        print("atoms[%d]: %s"%(i, atoms[i]))
    st = create_new_structure()

    elements = list(map(mm.mmat_get_element_by_atomic_number, atoms.get_atomic_numbers()))
    carts = atoms.get_positions()

    natoms = len(atoms)
    for iat in range(natoms):
        st.addAtom(elements[iat], carts[iat, 0], carts[iat, 1], carts[iat, 2])

    if all(atoms.get_pbc()):
        unit_cell = atoms.get_cell()
        abc_keys = [
            xtal.Crystal.CHORUS_BOX_A_KEYS,
            xtal.Crystal.CHORUS_BOX_B_KEYS,
            xtal.Crystal.CHORUS_BOX_C_KEYS,
        ]

        # lattice vecs are rows of ASE unit cell
        for ivec, abc_key in enumerate(abc_keys):
            for jvec, key in enumerate(abc_key):
                st.property[key] = unit_cell[ivec, jvec]

    # tell xtal this is P1
    xtal.make_p1(st, in_place=True)
    maximally_bonded, bonds, pbc_bonds = xtal.connect_atoms(st)
    st.property[xtal.PBC_POSITION_KEY] = xtal.ANCHOR_PBC_POSITION % ('0', '0', '0')
    st_out = xtal.get_cell(
        st,
        xtal_kwargs={'bonding': 'on', 'bond_orders': 'off', 'translate': 'off'}
    )
        

    #print("maximally_bonded", maximally_bonded)
    #print("bonds", bonds)
    #print("pbc_bonds", pbc_bonds)
    #print(st_out.getXYZ())

    return st_out

def atom_belong_to_mol1(index, atoms):
    """
    type  index: int
    param index: atom index 

    type  atoms: ASE atoms class
    param atoms: atoms object(containing serveral molecules in ASU)

    return index belongs to first molecule ? true : false 
    """
      
    st = ase_atoms_to_structure(atoms)
    if st.atom[index].molecule_number == 1:
        return True

    return False

def molecule_lists(atoms):
    """
    return list of lists of atoms belong to each molecule
    """

    st = ase_atoms_to_structure(atoms)
    ret_list = []
    for molecule in st.molecule:
        mole_list = []
        for atom in molecule.atom:
            mole_list.append(atom.index - 1)
        ret_list.append(mole_list)

    return ret_list

if __name__ == "__main__":

    ### testing functions above by HY
    atoms = ase.io.read(sys.argv[1])
    space_group = spacegroup.get_spacegroup(atoms) 
    print("space_group: ", space_group)
    my_st = ase_atoms_to_structure(atoms)
    writer = structure.StructureWriter(os.path.splitext(sys.argv[1])[0] + "_toView.mae")
    writer.append(my_st)
    writer.close()
 
    print(molecule_lists(atoms))

