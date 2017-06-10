import matplotlib.pyplot as plt
from random import randint


def rand_scatter(xval, yval, init_fig):
    if init_fig:
        plt.ion()
        f, ax = plt.subplots(1, figsize=(6 , 6))
        im = ax.scatter(xval, yval)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
    else:
        plt.figure(1)
        ax = plt.gca()
        for coll in plt.gca().collections:
            coll.remove()
        im = ax.scatter(xval, yval)

        plt.draw()
        plt.pause(0.000001)

if __name__ == "__main__":
    xval = []
    yval = []
    init = True
    while 0 == 0:
        xval.append(randint(1,9))
        yval.append(randint(1, 9))
        rand_scatter(xval, yval, init)
        init = False
