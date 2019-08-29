import ase
import ase.io
import ase.optimize
import torchani
import torch
from utils import models
import numpy as np

def torchani_calculator(ani_model, numb_networks):
    """
    Return a calculator from a model.
    choices are %s
    """ % " ".join(models.keys())

    model_file = models[ani_model]

    wkdir = model_file.rsplit('/', 1)[0] + '/'

    data = np.loadtxt(model_file, dtype=str)
    cnstfile = wkdir + data[0]  # AEV parameters
    saefile = wkdir + data[1]  # Atomic shifts
    nnfdir = wkdir + data[2]  # network prefix
    Nn = int(data[3])  # Number of networks in the ensemble
    assert numb_networks <= Nn

    constants = torchani.neurochem.Constants(cnstfile)
    energy_shifter = torchani.neurochem.load_sae(saefile)
    aev_computer = torchani.AEVComputer(**constants)
    aev_computer.to(torch.double)
    nn_models = torchani.neurochem.load_model_ensemble(
        constants.species,
        nnfdir, 
        numb_networks)

    # go to double precision
    for model in nn_models:
        model.to(torch.double)

    calculator = torchani.ase.Calculator(
        constants.species,
        aev_computer,
        nn_models,
        energy_shifter)

    return calculator

