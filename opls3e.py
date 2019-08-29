"""
Implementation of calculator for interfacing with ASE
"""

import numpy as np

from ase import io as ase_io
import ase.calculators.calculator
import sys
import ase.units

from schrodinger.infra import mm
from schrodinger.infra import mmerr
from atoms2st import atom_belong_to_mol1, ase_atoms_to_structure, molecule_lists

class Calculator(ase.calculators.calculator.Calculator):
    """
    OPLS3e implementation of ASE calculator

    """
    implemented_properties = ['energy','forces']
    def __init__(self, mmffld_handle):

        super(Calculator, self).__init__()

        self.mmffld_handle = mmffld_handle

    def calculate(self, atoms=None, properties=['energy'],
        system_changes=ase.calculators.calculator.all_changes,
        compute_hessian=False):
        """
        calculate properties
        """
        super(Calculator, self).calculate(atoms, properties, system_changes)

        st = ase_atoms_to_structure(atoms)
        mm.mmlewis_apply(st)
        mm.mmffld_enterMol(self.mmffld_handle, st.handle)
        mm.mmffld_deleteMol(self.mmffld_handle, st.handle)

        (ret_force_array, ret_energy_array) = mm.mmffld_getEnergyForce( self.mmffld_handle)

       	self.results['energy'] = ret_energy_array

        if 'force' in properties:
            self.results['force'] = ret_force_array


###main
mm.mmerr_initialize()
error_handler = mm.MMERR_DEFAULT_HANDLER
mm.mmlewis_initialize(error_handler)
mm.mmffld_initialize(error_handler)
calculator = Calculator(mm.mmffld_new(16))

atoms = ase_io.read(sys.argv[1])
atoms.set_calculator(calculator)
energy = atoms.get_potential_energy()

print("energy")
