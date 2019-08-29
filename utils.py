import os
import time
import numpy as np

from ase import optimize 
from ase.md import velocitydistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
import ase

EV = ase.units.Hartree 
KCAL = 627.509211

ANI_ASE = os.environ["ROITBERG_GROUP_NETWORKS"]
models = {
    "ani-ccx": os.path.join(ANI_ASE, "ani-1ccx_8x.info"),
    "ani-1x":  os.path.join(ANI_ASE, "ani-1x_8x.info")
}

def single_point_energy(molecule, calculator):
    """
    Compute the energy of a molecule in Hartree

    :type molecule: ase.Atoms
    :type Calculator: ase Calculator
    """
    molecule.set_calculator(calculator)
    energy = molecule.get_potential_energy()/EV

    return energy

def optimize_molecule(molecule, calculator, torsions=(), thresh=0.001, optimize_cell=False):
    """
    Optimize the molecule and return the optimized energy
    torsions should be a list of lists with value, i, j, k, l

    :type molecule: ase.Atoms
    :type Calculator: ase Calculator
    :type torsions: iterable of (value, lists of ints) tuple defining torsions to be constrained
    :type thresh: float
    :type optimize_cell: bool
    :return: the energy in hartree
    """

    if torsions:
        constraints = ase.constraints.FixInternals(dihedrals=torsions)
        molecule.set_constraint(constraints)

    molecule.set_calculator(calculator)

    if optimize_cell:
        ucf = ase.constraints.ExpCellFilter(molecule) 
        opt = optimize.BFGS(ucf)
    else:
        opt = optimize.BFGS(molecule)

    opt.run(fmax=thresh, steps=500)

    return molecule.get_potential_energy()/EV

def molecular_dynamics(
    molecule,
    calculator,
    temperature=300.0,
    sample_interval=10,
    total_time=10.0,
    time_step=0.1,
    seed=None):
    """
    Run molecular dynamics on a structure
    """

    if seed is not None:
        np.random.seed(seed)

    dt = time_step * ase.units.fs
    temp = temperature * ase.units.kB

    steps = int(total_time * 1000.0 / time_step)

    velocitydistribution.MaxwellBoltzmannDistribution(molecule, temp, force_temp=True)
    velocitydistribution.Stationary(molecule)
    velocitydistribution.ZeroRotation(molecule)

    print("Initial temperature from velocities %.2f" % molecule.get_temperature())
    print("Performing %d steps of %f fs for a total time of %f ps" % (
        steps, time_step, total_time))
    
    molecule.set_calculator(calculator)

#    dyn = Langevin(
#        molecule,
#        1.0*ase.units.fs,
#        temp,
#        friction=0.001,
#        logfile='-',
#        loginterval=sample_interval
#    )

    dyn = VelocityVerlet(
        molecule,
        dt,
        logfile='-',
        loginterval=sample_interval,
        trajectory='dynamics.traj'
    )

    md_path = []
    count_steps = 0
    ave_temp = 0
    def log_md_path(atoms):
        nonlocal md_path
        nonlocal count_steps
        nonlocal ave_temp
        if count_steps % sample_interval == 0:
            md_path.append(atoms.copy())
            ave_temp += atoms.get_temperature()
        count_steps += 1

    start_time = time.time()
    dyn.attach(log_md_path, 10, molecule)
    dyn.run(steps)

    end_time = time.time()

    print("Time to sample dynamics %.2f s" % (end_time - start_time))

    #ave_temp /= len(md_path)
    #print("Total MD time %.2f" % (steps / 1000.0))
    #print("Average Temperature during MD %.2f" % (ave_temp))

    return md_path
    
