import numpy
import seaborn as sns


def sigmoid(t):
    return 1 / (1 + numpy.exp(-t))


def linear(a, b, step, steps):
    return a + step / steps * (b - a)


def translate(a, b, step, steps, saturation=10):
    return a + (sigmoid(step / steps * saturation) - 0.5) * 2 * (b - a)


def despine(ax):
    sns.despine(ax=ax, bottom=True, left=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)



class ZOrder:
    def __init__(self):
        self.z = 1

    def __call__(self, inc=1):
        self.increase(inc)
        return self.get()

    def increase(self, inc=1):
        self.z += inc

    def get(self):
        return self.z
