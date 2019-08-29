import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    """
    parse commandline arguments
    """

    parser = argparse.ArgumentParser(
        description="plot MC trace of a particular run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'json_file',
        type=str,
        help='name of the json file that store trace data'
    )

    parser.add_argument(
        'name',
        type=str,
        help='name of the png file that saved'
    )

    parser.add_argument(
        '-perturbation_type',
        choices=['translation', 'rotation', 'cell_length_a', 'cell_length_b', 'cell_length_c', 'cell_angle_alpha', 'cell_angle_beta', 'cell_angle_gamma', 'energy'],
        type=str,
        default='energy',
        help='choose one type of perturbation, by default run energy type'
    )

    return parser.parse_args()

def plot_energy(data_file, name):

    """
    :type  data_file: str
    :param data_file: MC_trace dict
    """

    steps = []
    energies = []

    with open(data_file) as fh:
        data = json.load(fh)
        for key, value in data.items():
            steps.append(key)
            energies.append((value['Energy'] - -70592.50358030746)*23)
            print(energies[0])

    fig, ax = plt.subplots()
    fig.set_size_inches(16,12)
    ax.scatter(steps, energies, label='MC trace for Energy', color='k', s=10)
    # # ax.scatter(x_ani1_final, y_ani1_final, label='Other Existing Forms by ANI1', color='b', marker='^', s=150)
    # ax.scatter(x_exp1, y_exp1, label='Exp. Most Stable Forms by DFT', color='r', s=150)

    plt.xlim((0,len(steps)-1))
    plt.ylim((-20,20))
    x_new_ticks = np.linspace(0,len(steps)-1,11)
    y_new_ticks = np.linspace(-20,20,10)
    plt.xticks(x_new_ticks, fontsize=10)
    plt.yticks(y_new_ticks, fontsize=10)
    plt.xlabel('step', fontsize=10)
    plt.ylabel('Energy in kcal', fontsize=10)
    plt.title('Crystal Polymorph MC Trace', fontsize=10, y=1.05)
    plt.legend(loc='best', fontsize=10)
    # plt.show()
    plt.savefig('%s.png'%name)

def plot_params(data_file,name):

    """
    :type  data_file: str
    :param data_file: MC_trace dict
    """

    steps = []
    a = []

    with open(data_file) as fh:
        data = json.load(fh)
        for key, value in data.items():
            steps.append(key)
            a.append(value['a'])

    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    plt.axhline(y=11.186, color='r', linestyle='-', label="reference for a")
    ax.scatter(steps, a, label='MC trace for a', color='k', s=10)

    plt.xlim((0,len(steps)-1))
    plt.ylim((0, 20))
    x_new_ticks = np.linspace(0,len(steps)-1,11)
    y_new_ticks = np.linspace(0,19,20)
    plt.xticks(x_new_ticks, fontsize=10)
    plt.yticks(y_new_ticks, fontsize=10)
    plt.xlabel('step', fontsize=10)
    plt.ylabel('lattice constant a in Ans', fontsize=10)
    plt.title('Crystal Polymorph MC Trace', fontsize=10, y=1.05)
    plt.legend(loc='best', fontsize=10)
    plt.show()
    plt.savefig('%s.png'%name)

def main():

    args = parse_args()
    print("args: ", args)

    assert args.json_file.endswith('.json')
    perturbation_type = args.perturbation_type

    if perturbation_type == 'energy':
        plot_energy(args.json_file, args.name)
    else:
        plot_params(args.json_file, args.name)
    
if __name__ == "__main__":
    main()

