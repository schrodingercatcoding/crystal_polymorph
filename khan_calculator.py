import os
import tensorflow as tf
import numpy as np

import ase
import ase.io
import ase.optimize

from khan.ase_interface.calculator import Calculator
from khan.training.trainer_multi_tower import initialize_module
from khan.model.nn import read_sae_parameters

from utils import models

def khan_calculator(ani_model, khan_network, numb_networks):
    """
    Return a calculator from a roitberg model.
    choices are %s
    """ % " ".join(models.keys())

    if khan_network is None:
        print("setting up Roitberg network")
        model_file = models[ani_model]

        wkdir = model_file.rsplit('/', 1)[0] + '/'

        data = np.loadtxt(model_file, dtype=str)
        cnstfile = wkdir + data[0]  # AEV parameters
        saefile = wkdir + data[1]  # Atomic shifts
        nnfdir = wkdir + data[2]  # network prefix
        Nn = int(data[3])  # Number of networks in the ensemble

        assert numb_networks <= Nn

        network_dir = nnfdir[:-5]

    else:
        print("setting up khan network")
        saefile = os.path.join(
             os.environ["KHAN"],
             "data",
             "sae_linfit.dat"
             #"sae_wb97x.dat"
        )
        network_dir = None

    ani_lib = os.path.join(
        os.environ["KHAN"],
        "gpu_featurizer",
        "ani_cpu.so"
    )
    initialize_module(ani_lib)

    cp = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    tf_sess = tf.Session(config=cp)

    atomic_energies = read_sae_parameters(saefile)

    calculator = Calculator(
        tf_sess,
        atomic_energies,
        numb_networks,
        khan_saved_network=khan_network,
        roitberg_network=network_dir
    ) 
    
    return calculator

