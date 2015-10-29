import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(m):
    keyList = m.program.gps[0].result['variables'].keys()

    newDict = {}

    fig = plt.figure()
    ax = plt.subplot(111)

    for k in keyList:
        a = np.array([])
        for j in range(len(m.program.gps)):
            a = np.append(a, m.program.gps[j].result['variables'][k])
        newDict[k] = a/max(a)
        plt.plot(newDict[k], label=k.name + ' ({:6.2f})'.format(max(a)))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.xlabel('Number of iterations')
    plt.ylabel('Normalised variable values')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
