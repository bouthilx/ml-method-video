import json

import scipy.stats
import numpy
from matplotlib.lines import Line2D
from matplotlib import patches
import matplotlib.pyplot as plt


def cum_argmax(x, duplicate_jump=False):
    """Return the indices corresponding to an cumulative minimum
    (numpy.minimum.accumulate)
    """
    minima = numpy.maximum.accumulate(x)
    diff = numpy.diff(minima)
    jumps = numpy.arange(len(x))
    jumps[1:] *= diff != 0
    jumps = numpy.maximum.accumulate(jumps)
    if duplicate_jump:
        indices = []
        for i, index in jumps[:-2]:
            indices.append(index)
            if index != jumps[i + 1]:
                indices.append(index)
        return indices
    return jumps


def plot_line(axe, x, y, err, label=None, alpha=0.5, color=None, linestyle=None):
    # axe.plot(x, y, label=label, color=color, linestyle=linestyle)
    if err is not None:
        y = numpy.array(y)
        err = numpy.array(err)

        min_y = y - err
        min_y = min_y * (min_y > 0)
        max_y = y + err
        max_y = max_y * (max_y <= 100) + 100 * (max_y > 100)
        # axe.fill_between(x, max_y, min_y, linewidth=0, alpha=alpha, color=color)
        axe.fill_between(x, min_y, max_y, linewidth=0, alpha=alpha, color=color)


class PapersWithCodePlot:
    stat_keys = {
        "sst2": "bert-sst2",
        "cifar10": "vgg",
        "rte": "bert-rte",
    }

    def __init__(self):
        pass

    def load(self):
        with open("paperswithcode.json", "r") as f:
            self.data = json.load(f)

        with open("stats.json", "r") as f:
            self.stats = json.load(f)

        for key in self.data.keys():
            self.data[key] = numpy.array(self.data[key])

    def plot(self, ax, key, alpha=0.05, beta=0.05):

        x = self.data[key][:, 0]
        y = self.data[key][:, 1]
        ax.scatter(x, y, [1 for i in range(len(y))])

        coeff = scipy.stats.norm.ppf(1 - alpha)  #  - scipy.stats.norm.ppf(beta)

        idx = cum_argmax(self.data[key][:, 1])
        ax.plot(x[idx], y[idx])

        i = 0
        while i < len(idx) - 1:
            j = i + 1

            effect_size = (
                numpy.sqrt(2) * coeff * self.stats[self.stat_keys[key]]["sigma"] * 100
            )

            # while j < len(idx) - 1 and y[idx][i] + effect_size > y[idx][j]:
            #     j += 1

            if j == len(idx):
                xaxis = [x[idx][i], x[-1]]
            else:
                xaxis = [x[idx][i], x[idx][i + 1]]

                # if y[idx][i+1] - y[idx][i] > effect_size:

            plot_line(ax, xaxis, [y[idx][i]] * 2, err=effect_size, color="blue")
            plot_line(
                ax,
                xaxis,
                [y[idx][i]] * 2,
                err=self.stats[self.stat_keys[key]]["sigma"] * 100,
                color="red",
            )

            i = j

            # axes.plot(, color='blue')

        print(min(x), max(x))
        ax.set_xlim(min(x), max(x))

        if key == "sst2":
            ax.set_ylim(80, 100)
        elif key == "cifar10":
            ax.set_ylim(50, 100)

    def save(self, name="paperswithcode"):
        plt.savefig(f"{name}.png", dpi=300)
        plt.savefig(f"{name}.pdf", dpi=300)


if __name__ == "__main__":

    (8.5, 11)

    WIDTH = (8.5 - 1.0) / 2
    HEIGHT = (11 - 1.5) / 5

    # Prepare matplotlib
    plt.rcParams.update({"font.size": 8})
    plt.close("all")
    # plt.rc('font', family='serif', serif='Times')
    plt.rc("font", family="Times New Roman")
    # plt.rc('text', usetex=True)
    plt.rc("xtick", labelsize=7)
    plt.rc("ytick", labelsize=8)
    plt.rc("axes", labelsize=8)

    fig = plt.figure(figsize=(4, 6.5))
    axes = fig.subplots(
        ncols=1,
        nrows=1,
        # sharex=True, sharey='row',
        # gridspec_kw={'left': .41,
        #     'top': 1, 'right': 1.01, 'bottom': .25,
        #     'hspace': 0.7, 'wspace': 0.0}
    )
    key = "sst2"
    # key = 'cifar10'
    # key = 'rte'

    paperswithcode = PapersWithCodePlot()
    paperswithcode.load()
    paperswithcode.plot(axes, key)
    fig.set_size_inches(WIDTH, HEIGHT)
    paperswithcode.save()
