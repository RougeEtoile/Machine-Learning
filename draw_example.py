import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from random import randint

plt.ion()
scatter_plots = []


def rand_scatter(xval, yval, init_fig):
    if init_fig:
        f, ax = plt.subplots(1, figsize=(6, 6))
        im = ax.scatter(xval, yval)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
    else:
        ax = plt.gca()
        for coll in plt.gca().collections:
            coll.remove()
        im = ax.scatter(xval, yval)
    plt.draw()
    plt.pause(0.000001)
    scatter_plots.append(mplfig_to_npimage(plt.gcf()))


if __name__ == "__main__":
    xval = []
    yval = []
    init = True
    i = 0
    movie = []
    while i != 10:
        xval.append(randint(1,9))
        yval.append(randint(1, 9))
        #rand_boxplot(xval, yval, init)
        rand_scatter(xval, yval, init)
        init = False
        i += 1

    clip = mpy.ImageSequenceClip(scatter_plots, fps=1, load_images=True)  # Accepts folders
    clip.write_videofile('draw_example.mp4', fps=1, codec='mpeg4', audio=False)

