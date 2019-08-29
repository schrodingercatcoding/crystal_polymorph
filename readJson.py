import json
import sys
import matplotlib.pyplot as plt
import numpy as np

step = []
a = []
b = []
c = []
beta = []

with open(sys.argv[1]) as json_file:
    data = json.load(json_file)
    for key, value in data.items():
        step.append(key)
        a.append(value['a'])
        b.append(value['b'])
        c.append(value['c'])
        beta.append(value['beta'])

# plot
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
plt.axhline(y=11.186, color='r', linestyle='-', label="reference for a")
plt.axhline(y=6.540, color='b', linestyle='-', label="reference for b")
plt.axhline(y=11.217, color='k', linestyle='-', label="reference for c")
ax.scatter(step, a, label='MC trace for a', color='r', s=10)
ax.scatter(step, b, label='MC trace for b', color='b', s=10)
ax.scatter(step, c, label='MC trace for c', color='k', s=10)
# # ax.scatter(x_ani1_final, y_ani1_final, label='Other Existing Forms by ANI1', color='b', marker='^', s=150)
# ax.scatter(x_exp1, y_exp1, label='Exp. Most Stable Forms by DFT', color='r', s=150)

plt.xlim((0,len(step)-1))
plt.ylim((0, 20))
x_new_ticks = np.linspace(0,len(step)-1,11)
y_new_ticks = np.linspace(0,19,20)
plt.xticks(x_new_ticks, fontsize=10)
plt.yticks(y_new_ticks, fontsize=10)
plt.xlabel('step', fontsize=10)
plt.ylabel('lattice a', fontsize=10)
plt.title('Crystal Polymorph MC Trace', fontsize=10, y=1.05)
plt.legend(loc='best', fontsize=10)
plt.show()