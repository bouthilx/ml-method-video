import argparse
import copy
import functools
from multiprocessing import Pool
import warnings
import itertools
import cProfile

import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy
import scipy.stats
import seaborn as sns
from matplotlib import patches
from matplotlib import pyplot as plt
import matplotlib.cm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import scipy.optimize


from moustachos import adjust_moustachos, moustachos, h_moustachos, adjust_h_moustachos
from rained_histogram import rained_histogram, rain_std, RainedHistogram
from utils import translate, linear, ZOrder, despine, show_text, ornstein_uhlenbeck_step
from paperswithcode import PapersWithCodePlot, cum_argmax
from variances import VariancesPlot
from estimator_bubbles import EstimatorBubbles
from estimators import EstimatorsPlot, LABELS
from estimators import COLORS as EST_COLORS
from simulations import (
    pab,
    percentile_bootstrap,
    normal_ci,
    SimulationPlot,
    SimulationScatter,
    AverageTestViz,
    AverageTest,
    PABTestViz,
    PABTest,
)


warnings.filterwarnings("ignore")

# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore", r"RuntimeWarning: Degrees of freedom <= 0 for slice"
#     )

END = 10
FPS = 60

FADE_OUT = FPS * 2
TEXT_SPEED = 35  # Characters per second


zorder = ZOrder()


N_INTRO = 25

PAB = 0.75


norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
variances_colors = matplotlib.cm.get_cmap("tab10")

# fig = plt.figure()  # figsize=(width, height))
# fig.tight_layout()
# ax = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1))
# plt.gca().set_position([0, 0, 1, 1])
# scatter_solid = ax.scatter([], [], alpha=1)


def cum_argmax(x, y):
    """Return the indices corresponding to an cumulative minimum
    (numpy.minimum.accumulate)
    """
    minima = numpy.maximum.accumulate(y)
    diff = numpy.diff(minima)
    jumps = numpy.arange(len(y))
    jumps[1:] *= diff != 0
    jumps = numpy.maximum.accumulate(jumps)
    xs = []
    ys = []
    for i, index in enumerate(jumps[1:]):
        i += 1
        if index != jumps[i - 1]:
            xs.append(x[i])
            ys.append(y[jumps[i - 1]])
        else:
            xs.append(x[i])
            ys.append(y[index])
        xs.append(x[i])
        ys.append(y[index])
    return numpy.array(xs), numpy.array(ys)


class Section:
    def __init__(self, plots):
        self.plots = plots
        self.j = 0
        self.last_i = 0
        self.counter = 0
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.fig = fig
        self.initialized = True

    @property
    def n_frames(self):
        return sum([plot.n_frames for plot in self.plots])

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(fig, ax, last_animation)

        step = i - self.last_i
        self.counter += step
        self.last_i = i

        if self.j >= len(self.plots):
            return

        plot = self.plots[self.j]
        if plot.n_frames <= self.counter and self.j + 1 >= len(self.plots):
            plot.leave()
            return 0

        elif plot.n_frames <= self.counter:
            # self.j += 1
            # self.counter -= plot.n_frames
            plot = self.plots[self.j]
            plot.initialize(fig, ax, self.plots[self.j - 1] if self.j > 0 else None)
            while plot.n_frames <= self.counter:
                plot(
                    plot.n_frames,
                    fig,
                    ax,
                    self.plots[self.j - 1] if self.j > 0 else None,
                )
                plot.leave()
                self.j += 1
                if self.j >= len(self.plots):
                    return 0
                self.counter -= plot.n_frames
                plot = self.plots[self.j]
                plot.initialize(fig, ax, self.plots[self.j - 1] if self.j > 0 else None)

            self.counter = max(self.counter, 0)
        plot.initialize(fig, ax, self.plots[self.j - 1] if self.j > 0 else None)
        plot(
            self.counter,
            fig,
            ax,
            self.plots[self.j - 1] if self.j > 0 else None,
        )
        return step

    def leave(self):
        self(self.n_frames, None, None, None)


class Chapter(Section):
    def __init__(self, name, plots, pbar_position=None):
        super(Chapter, self).__init__(plots)
        self.name = name
        self.pbar_position = pbar_position

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.name:
            self.pbar = tqdm(
                total=self.n_frames,
                leave=True,
                desc=self.name,
                position=self.pbar_position,
            )
        super(Chapter, self).initialize(fig, ax, last_animation)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        step = i - self.last_i
        super(Chapter, self).__call__(i, fig, ax, last_animation)
        if self.name:
            self.pbar.update(step)

    def leave(self):
        self.fig.clear()
        if self.name:
            self.pbar.close()


class Animation:
    def __init__(self, plots, width, height, start=0, end=-1, fps=FPS):
        self.plots = plots
        self.start = start
        self._end = end
        self.fps = fps
        self.j = 0
        self.counter = 0

        self.fig = plt.figure(figsize=(width, height))
        self.fig.tight_layout()
        self.ax = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1), label="main")
        # self.ax.plot([0.6, 4], [0.1, 0.8])
        plt.gca().set_position([0, 0, 1, 1])
        self.scatter = self.ax.scatter([], [], alpha=1, zorder=zorder())
        self.initialized = False

    def initialize(self):
        # self.pbar = tqdm(total=self.n_frames, desc="Full video")
        self.initialized = True
        total = 0
        for j, plot in enumerate(self.plots):
            if total + plot.n_frames > self.start:
                self.counter = int(self.start - total)
                self.j = j
                break
            else:
                plot.initialize(self.fig, self.ax, self.plots[j - 1] if j > 0 else None)
                plot.leave()
            total += plot.n_frames

    @property
    def n_frames(self):
        return self.end - self.start

    @property
    def end(self):
        if self._end < 0:
            return sum(plot.n_frames for plot in self.plots)
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    @property
    def step(self):
        return int(FPS / self.fps)

    def __call__(self, i):
        if not self.initialized:
            self.initialize()
            return (self.scatter,)

        i *= self.step
        i += self.start
        i = int(i)
        plot = self.plots[self.j]
        if plot.n_frames <= self.counter and self.j + 1 >= len(self.plots):
            plot.leave()
            return (self.scatter,)
        elif plot.n_frames <= self.counter:
            plot.leave()

            # self.j += 1
            # self.counter -= plot.n_frames
            plot = self.plots[self.j]
            plot.initialize(
                self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
            )
            while plot.n_frames <= self.counter:
                plot.leave()
                self.j += 1
                self.counter -= plot.n_frames
                plot = self.plots[self.j]
                plot.initialize(
                    self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
                )

            self.counter = max(self.counter, 0)
        plot.initialize(
            self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
        )
        plot(
            self.counter,
            self.fig,
            self.ax,
            self.plots[self.j - 1] if self.j > 0 else None,
        )
        self.counter = int(self.counter + self.step)
        # self.pbar.update(self.step)
        return (self.scatter,)


class Black:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.black_patch = patches.Rectangle((0, 0), 0, 0, fill=True, color="black")
        ax.add_patch(self.black_patch)
        self.black_patch.set_width(5)
        self.black_patch.set_height(5)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(ax)

    def leave(self):
        self.black_patch.set_width(0)
        self.black_patch.set_height(0)


class PapersWithCode:
    def __init__(
        self,
        n_frames,
        title_fontsize=38,
        axis_label_fontsize=32,
        axis_tick_fontsize=16,
        ax=None,
    ):
        self.n_frames = n_frames
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.axis_tick_fontsize = axis_tick_fontsize
        self.ax = ax
        self.key = "sst2"
        self.initialized = False

    def update(self, y):
        self.y = y
        self.scatter.set_offsets(list(zip(self.x, y)))

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.ax is None:
            self.ax = ax

        # ax.plot([0.6, 4], [0.1, 0.8])  #  = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1))
        paperswithcode = PapersWithCodePlot()
        paperswithcode.load()

        self.x = paperswithcode.data[self.key][:, 0]
        self.y = paperswithcode.data[self.key][:, 1]
        self.p = self.y
        self.scatter = self.ax.scatter(
            self.x,
            self.p,
            numpy.ones(len(self.p)) * 15,
            zorder=zorder(2),
            c=matplotlib.cm.get_cmap("tab10")(1),
        )
        cum_x, cum_y = cum_argmax(self.x, self.p)
        self.line = self.ax.plot(
            cum_x,
            cum_y,
            color=matplotlib.cm.get_cmap("tab10")(0),
            zorder=zorder.get() - 1,
        )[0]

        if self.ax is ax:
            self.ax.set_position([0.2, 0.2, 0.6, 0.6])

        if self.key == "sst2":
            self.ax.set_ylim(80, 100)
        elif self.key == "cifar10":
            self.ax.set_ylim(85, 100)
        self.ax.set_xlim(min(self.x), max(self.x))

        self.ax.set_xlabel("Year", fontsize=self.axis_label_fontsize)
        self.ax.set_ylabel("Accuracy", fontsize=self.axis_label_fontsize)
        self.ax.set_title(
            "Sentiment Analysis on\nSST-2 Binary classification",
            fontsize=self.title_fontsize,
        )

        self.ax.tick_params(
            axis="both", which="major", labelsize=self.axis_tick_fontsize
        )
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class NoisyPapersWithCode:
    def __init__(self, n_frames, paperswithcode):
        self.n_frames = n_frames
        self.initialized = False
        self.paperswithcode = paperswithcode
        self.sizes = {"rte": 277, "sst2": 872, "cifar10": 10000}

    def _generate_noise(self, y):
        dataset_size = self.sizes[self.key]
        std = scipy.stats.binom(n=dataset_size, p=y / 100).std() / dataset_size
        return numpy.random.normal(y / 100, std) * 100

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.key = self.paperswithcode.key
        self.x = self.paperswithcode.x
        self.p = self.paperswithcode.p
        self.old_y = copy.deepcopy(self.paperswithcode.y)
        self.new_y = self._generate_noise(self.p)
        self.old_cum_x, self.old_cum_y = cum_argmax(self.x, self.old_y)
        self.new_cum_x, self.new_cum_y = cum_argmax(self.x, self.new_y)

        self.line = self.paperswithcode.ax.plot(
            self.old_cum_x,
            self.old_cum_y,
            color=matplotlib.cm.get_cmap("tab10")(0),
            zorder=zorder.get() - 1,
        )[0]
        # self.line = last_animation.line
        # last_animation.line.set_alpha(0.2)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(fig, ax, last_animation)

        ratio = 0.5
        saturation = 7
        y = translate(
            self.old_y, self.new_y, i, self.n_frames * ratio, saturation=saturation
        )
        self.paperswithcode.update(y)

        new_cum_argmax = translate(
            self.old_cum_x,
            self.new_cum_x,
            i,
            self.n_frames * ratio,
            saturation=saturation,
        )
        self.line.set_xdata(new_cum_argmax)
        new_cum_argmax = translate(
            self.old_cum_y,
            self.new_cum_y,
            i,
            self.n_frames * ratio,
            saturation=saturation,
        )
        self.line.set_ydata(new_cum_argmax)

    def leave(self):
        pass


class Still:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class VarianceLabel:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.text = "1. Different sources\n    of varations"
        self.initialized = False

        self.moustacho_x = 2016.9
        self.moustacho_y = 90.5
        self.moustacho_center_width = 0.2
        self.moustacho_whisker_length = 2
        self.moustacho_whisker_length_noisy = 2
        self.moustacho_whisker_width = 0.125

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.label = ax.text(
            2019, 99, "", va="top", ha="left", fontsize=28, zorder=zorder(2)
        )

        self.annotation = ax.annotate(
            "",
            xy=(
                self.moustacho_x,
                self.moustacho_y + self.moustacho_whisker_length * 1.25,
            ),
            xycoords="data",
            xytext=(2019, 97),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                facecolor="black",
                patchB=None,
                shrinkB=0,
                connectionstyle="angle3,angleA=0,angleB=90",
                # connectionstyle="arc3,rad=0.3",
            ),
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=32,
            zorder=zorder.get(),
        )

        self.moustacho = moustachos(
            ax,
            x=self.moustacho_x,
            y=self.moustacho_y,
            whisker_width=self.moustacho_whisker_width * 0.01,
            whisker_length=self.moustacho_whisker_length * 0.01,
            center_width=self.moustacho_center_width * 0.01,
            zorder=zorder.get() - 1,
        )

        self.white_patch = patches.Rectangle(
            (2018.75, 95),
            3,
            5,
            fill=True,
            color="white",
            zorder=zorder.get() - 1,
            alpha=0.75,
        )
        ax.add_patch(self.white_patch)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n = int(translate(3, len(self.text) + 1, i, self.n_frames / 15, saturation=5))
        self.label.set_text(self.text[:n] + (" " * (len(self.text) - n)))

        self.moustacho_whisker_length_noisy += ornstein_uhlenbeck_step(
            self.moustacho_whisker_length,
            self.moustacho_whisker_length_noisy,
            stability=0.5,
            standard_deviation=0.1,
        )

        adjust_moustachos(
            self.moustacho,
            x=self.moustacho_x,
            y=self.moustacho_y,
            whisker_width=translate(
                self.moustacho_whisker_width * 0.01,
                self.moustacho_whisker_width,
                i,
                self.n_frames / 10,
                saturation=5,
            ),
            whisker_length=translate(
                self.moustacho_whisker_length * 0.01,
                self.moustacho_whisker_length_noisy,
                i,
                self.n_frames / 10,
                saturation=5,
            ),
            center_width=self.moustacho_center_width * 0.01,
        )

    def leave(self):
        pass


class EstimatorLabel:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.text = "2. Estimating the\n    average performance"
        self.initialized = False

        self.moustacho_x = 2016.9
        self.moustacho_y = 90.5
        self.moustacho_y_noisy = 90.5
        self.moustacho_center_width = 0.2
        self.moustacho_whisker_length = 2

        self.moustacho_whisker_width = 0.125

    def initialize(self, fig, ax, last_animation):

        if self.initialized:
            return

        self.label = ax.text(
            2019, 92, "", va="top", ha="left", fontsize=28, zorder=zorder.get()
        )

        d_h_center = 2016.54  # 2016.625
        d_v_center = 90
        d_h_width = 0.2

        d_h_center = 2016.9
        d_h_middle = (d_h_center + 2016.54) / 2
        d_v_center = 90.5
        d_v_width = 2
        d_v_middle = d_v_center - d_v_width * 2
        d_h_width = 0.2

        self.annotation = ax.annotate(
            "",
            xy=(
                self.moustacho_x + self.moustacho_center_width * 1.25,
                self.moustacho_y,
            ),
            xycoords="data",
            xytext=(2019, 90),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                facecolor="black",
                patchB=None,
                shrinkB=0,
                # connectionstyle="angle3,angleA=0,angleB=1",
                # connectionstyle="arc3,rad=0.3",
            ),
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=32,
            zorder=zorder.get(),
        )

        self.white_patch = patches.Rectangle(
            (2018.75, 88),
            3,
            5,
            fill=True,
            color="white",
            zorder=zorder.get() - 1,
            alpha=0.75,
        )

        self.moustacho = last_animation.moustacho
        self.moustacho_whisker_length_noisy = (
            last_animation.moustacho_whisker_length_noisy
        )

        # ax.plot(
        #     [d_h_center, d_h_center],
        #     [d_v_center - d_v_width, d_v_center + d_v_width],
        #     color="black",
        # )

        ax.add_patch(self.white_patch)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n = int(translate(3, len(self.text) + 1, i, self.n_frames / 15, saturation=5))
        self.label.set_text(self.text[:n] + (" " * (len(self.text) - n)))

        self.moustacho_y_noisy += ornstein_uhlenbeck_step(
            self.moustacho_y,
            self.moustacho_y_noisy,
            stability=0.05,
            standard_deviation=0.05,
        )

        adjust_moustachos(
            self.moustacho,
            x=self.moustacho_x,
            y=self.moustacho_y_noisy,
            whisker_width=self.moustacho_whisker_width,
            whisker_length=self.moustacho_whisker_length_noisy,
            center_width=translate(
                self.moustacho_center_width * 0.01,
                self.moustacho_center_width,
                i,
                self.n_frames / 10,
                saturation=10,
            ),
        )

    def leave(self):
        pass


class ComparisonLabel:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.text = "3. How to compare and\n    account for variance"
        self.initialized = False

        self.moustacho_x = 2016.54
        self.moustacho_y = 90
        self.moustacho_y_noisy = 90
        self.moustacho_center_width = 0.2
        self.moustacho_whisker_length = 2
        self.moustacho_whisker_length_noisy = 2
        self.moustacho_whisker_width = 0.125

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.label = ax.text(
            2019, 84.5, "", va="top", ha="left", fontsize=28, zorder=zorder(2)
        )

        d_h_center = 2016.9
        d_h_middle = (d_h_center + 2016.54) / 2
        d_v_center = 90.5
        d_v_width = 2.5
        d_v_middle = d_v_center - d_v_width * 1.5
        d_h_width = 0.2

        d_h_center = 2016.54  # 2016.625
        d_v_center = 90
        d_h_width = 0.2

        self.annotation = ax.annotate(
            "",
            xy=(d_h_middle, d_v_middle),
            xycoords="data",
            xytext=(2019, 82.5),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                facecolor="black",
                patchB=None,
                shrinkB=0,
                connectionstyle="angle3,angleA=0,angleB=90",
                # connectionstyle="arc3,rad=0.3",
            ),
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=32,
            zorder=zorder.get(),
        )

        self.moustacho = moustachos(
            ax,
            x=self.moustacho_x,
            y=self.moustacho_y,
            whisker_width=self.moustacho_whisker_width * 0.01,
            whisker_length=self.moustacho_whisker_length * 0.01,
            center_width=self.moustacho_center_width * 0.01,
            zorder=zorder.get() - 1,
        )

        self.white_patch = patches.Rectangle(
            (2018.75, 80.5),
            3,
            5,
            fill=True,
            color="white",
            zorder=zorder.get() - 1,
            alpha=0.75,
        )
        ax.add_patch(self.white_patch)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n = int(translate(3, len(self.text) + 1, i, self.n_frames / 15, saturation=5))
        self.label.set_text(self.text[:n] + (" " * (len(self.text) - n)))

        for animation in [self, last_animation]:

            animation.moustacho_y_noisy += ornstein_uhlenbeck_step(
                animation.moustacho_y,
                animation.moustacho_y_noisy,
                stability=0.005,
                standard_deviation=0.05,
            )

            animation.moustacho_whisker_length_noisy += ornstein_uhlenbeck_step(
                animation.moustacho_whisker_length,
                animation.moustacho_whisker_length_noisy,
                stability=0.5,
                standard_deviation=0.01,
            )

            if animation is self:
                whisker_width = translate(
                    animation.moustacho_whisker_width * 0.01,
                    animation.moustacho_whisker_width,
                    i,
                    animation.n_frames / 10,
                    saturation=10,
                )

                whisker_length = translate(
                    animation.moustacho_center_width * 0.01,
                    animation.moustacho_whisker_length_noisy,
                    i,
                    animation.n_frames / 10,
                    saturation=10,
                )

                center_width = translate(
                    animation.moustacho_center_width * 0.01,
                    animation.moustacho_center_width,
                    i,
                    animation.n_frames / 10,
                    saturation=10,
                )
            else:
                whisker_width = animation.moustacho_whisker_width
                whisker_length = animation.moustacho_whisker_length_noisy
                center_width = animation.moustacho_center_width

            adjust_moustachos(
                animation.moustacho,
                x=animation.moustacho_x,
                y=animation.moustacho_y_noisy,
                whisker_width=whisker_width,
                whisker_length=whisker_length,
                center_width=center_width,
            )

    def leave(self):
        pass


class Zoom:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        old_x = 0.2
        new_x = -2
        old_y = 0.6
        new_y = 5
        saturation = 3
        tmp_x = translate(old_x, new_x, i, self.n_frames, saturation=saturation)
        tmp_y = translate(old_y, new_y, i, self.n_frames, saturation=saturation)
        ax.set_position([tmp_x, tmp_x, tmp_y, tmp_y])

    def leave(self):
        pass


def get_variance_color(label):
    i = 0
    for ith_label in Variances.label_order:
        if label == ith_label:
            break
        if not label.startswith("empty_"):
            i += 1

    return i


class Variances:

    labels = {
        "empty_below_hpo": "",
        "empty_above_hpo": "",
        "bootstrapping_seed": "Data splits",
        "init_seed": "Weights init",
        "sampler_seed": "Data order",
        "global_seed": "Dropout",
        "transform_seed": "Data augment",
        "reference": "Numerical noise",
        "random_search": "Random Search",
        "noisy_grid_search": "Noisy Grid Search",
        "bayesopt": "Bayes Opt",
        "everything": "Everything",
    }

    label_order = [
        "reference",
        "global_seed",
        "init_seed",
        "sampler_seed",
        "transform_seed",
        "bootstrapping_seed",
        "empty_below_hpo",
        "random_search",
        "noisy_grid_search",
        "bayesopt",
        "empty_above_hpo",
        "everything",
    ]

    def __init__(self, data_folder, with_ax=True):
        self.data_folder = data_folder
        self.with_ax = with_ax
        self.n_frames = 0  # FPS * 1
        self.initialized = False
        self.keys = ["vgg", "segmentation", "bio-task2", "bert-sst2", "bert-rte"]
        self.titles = {
            "vgg": "CIFAR10\nVGG11",
            "segmentation": "PascalVOC\nResNet18",
            "bio-task2": "MHC\nMLP",
            "bert-sst2": "Glue-SST2\nBERT",
            "bert-rte": "Glue-RTE\nBERT",
        }

        self.hist_height = 1
        self.hist_heights = len(self.label_order) * self.hist_height
        self.n_columns = 20
        self.label_width = 0.2
        self.ax_height = 0.7
        self.ax_bottom_padding = 0.1
        self.ax_max_width = 0.3
        self.ax_max_padding = 0.01
        self.title_y = 0.9
        self.x_axis_label_y = 0.05

    def _get_n_points(self, task):
        data = self.get_data(task)
        return sum(data[label].shape[0] for label in self.label_order if label in data)

    def _get_max_std(self, task):
        return max(
            array.std() for array in self.variances_plot.variances[task].values()
        )

    def get_data(self, task):
        return self.variances_plot.variances[task]

    def get_slice(self, task, noise_type):
        i = 0
        data = self.get_data(task)
        for label in self.label_order:
            if label == noise_type:
                break

            elif label not in data:
                continue

            i += data[label].shape[0]

        return slice(i, i + data[noise_type].shape[0])

    def get_weights_y(self, noise_type):
        index = self.label_order.index(noise_type)
        return index * self.hist_height

    def get_label_height(self):
        return 1 / (len(self.labels) + 5 + 0.1)

    def get_label_y(self, label):
        i = self.label_order.index(label)
        return (i + 2) / (len(self.labels) + 5 + 0.1) + 0.005

    def get_bar(self, task, noise_type):
        labels = [label for label in self.label_order if label]
        return self.bars[task][labels.index(noise_type)]

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.variances_plot = VariancesPlot(self.data_folder)
        self.variances_plot.load()

        if not self.with_ax:
            self.initialized = True
            return

        self.bar_axes = {}
        self.hist_axes = {}
        self.bars = {}
        self.scatters = {}
        ax_zorder = zorder(2)
        for i, key in enumerate(self.keys):
            self.bar_axes[key] = fig.add_subplot(
                1,
                2 * len(self.keys),
                i * 2 + 1,
                facecolor=["red", "blue", "purple", "yellow", "orange"][i],
                frameon=False,
                zorder=ax_zorder,
            )

            self.hist_axes[key] = fig.add_subplot(
                1,
                2 * len(self.keys),
                i * 2 + 2,
                facecolor=["red", "blue", "purple", "yellow", "orange"][i],
                frameon=False,
                zorder=ax_zorder - 1,
            )

            self.bars[key] = self.bar_axes[key].barh(
                range(len(self.label_order)),
                numpy.zeros(len(self.label_order)),
                align="edge",
                clip_on=False,
                color=[
                    variances_colors(get_variance_color(label))
                    for label in self.label_order
                ],
                zorder=ax_zorder,
            )

            self.bar_axes[key].set_xlim((0, self._get_max_std(key) * 1.05))
            self.bar_axes[key].set_ylim((0, 12))

            data = numpy.ones(self._get_n_points(key)) * -1000
            # self.scatters[key].set_offsets(list(zip(data, data)))

            arrays = self.get_data(key)
            colors = numpy.concatenate(
                [
                    numpy.ones(arrays[label].shape) * get_variance_color(label)
                    for label in self.label_order
                    if label in arrays
                ]
            ).astype(int)

            self.scatters[key] = self.hist_axes[key].scatter(
                data,
                data,
                c=colors,
                marker="s",
                cmap=variances_colors,  # vmin=0, vmax=1
                vmin=0,
                vmax=9,
                zorder=ax_zorder - 1,
                clip_on=False,
            )

            self.hist_axes[key].set_xlim((0, 1))
            self.hist_axes[key].set_ylim((0, self.hist_heights))
            for axis in [self.hist_axes[key], self.bar_axes[key]]:
                for side in ["top", "right", "bottom", "left"]:
                    axis.spines[side].set_visible(False)
                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)

            # self.bar_axes[key].set_position([i * 2 * 0.08 + 0.2, 0.1, 0.08, 0.8])
            # self.hist_axes[key].set_position(
            #     [i * 2 * 0.08 + 0.2 + 0.08, 0.1, 0.08, 0.8]
            # )
            self.bar_axes[key].set_position(
                [2, self.ax_bottom_padding, self.ax_max_width, self.ax_height]
            )
            self.hist_axes[key].set_position(
                [2, self.ax_bottom_padding, self.ax_max_width, self.ax_height]
            )

        self.hist_axes["vgg"].set_position(
            [
                self.label_width,
                self.ax_bottom_padding,
                self.ax_max_width,
                self.ax_height,
            ]
        )
        self.bar_axes["vgg"].set_position(
            [
                self.label_width + self.ax_max_padding + self.ax_max_width,
                self.ax_bottom_padding,
                self.ax_max_width,
                self.ax_height,
            ]
        )

        self.labels_objects = {}
        for label in self.label_order:
            self.labels_objects[label] = self.bar_axes["vgg"].text(
                self.label_width * 0.95,
                self.get_label_y(label),
                "",
                transform=fig.transFigure,
                va="center",
                ha="right",
                fontsize=18,
            )

        self.task_label_objects = {}
        for i, task in enumerate(self.keys):
            self.task_label_objects[task] = self.bar_axes["vgg"].text(
                self.label_width
                + (2 * i) * (self.ax_max_width + self.ax_max_padding)
                + self.ax_max_width
                + self.ax_max_padding / 2,
                self.title_y,
                self.titles[task] if task != "vgg" else "",
                transform=fig.transFigure,
                va="center",
                ha="center",
                fontsize=24,
            )

        self.performances_label = self.bar_axes["vgg"].text(
            self.label_width + self.ax_max_width / 2,
            self.x_axis_label_y,
            "",
            transform=fig.transFigure,
            va="center",
            ha="center",
            fontsize=24,
        )

        self.standard_deviation_label = self.bar_axes["vgg"].text(
            self.label_width + self.ax_max_width * 3 / 2 + self.ax_max_padding,
            self.x_axis_label_y,
            "",
            transform=fig.transFigure,
            va="center",
            ha="center",
            fontsize=24,
        )

        # self.bar_axes[self.keys[-1]].set_position([0.2, 0.1, 0.6, 0.2])
        # self.hist_axes["vgg"].set_position([0.2, 0.2, 0.6, 0.6])
        # self.hist_axes[self.keys[-1]].set_position([0.2, 0.2, 0.6, 0.6])

        # ax.remove()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class VarianceSourceLabel:
    def __init__(self, n_frames, variances, noise_type):
        self.n_frames = n_frames
        self.variances = variances
        self.noise_type = noise_type
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.text = WriteText(
            self.variances.labels[self.noise_type],
            self.variances.labels_objects[self.noise_type],
            fill=False,
        )
        self.text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class SpeedUpVarianceSource:
    def __init__(
        self,
        n_frames,
        n_frames_start,
        n_frames_speedup,
        variances,
        task,
        noise_type,
        start_spacing=500,
        end_spacing=10,
        start_delta=5,
        end_delta=20,
        n_slow=3,
        n_speedup=50,
        with_label=True,
    ):

        self.n_frames = n_frames
        self.n_frames_start = n_frames_start
        self.n_frames_speedup = n_frames_speedup

        self.variances = variances
        self.start_spacing = start_spacing
        self.end_spacing = end_spacing
        self.start_delta = start_delta
        self.end_delta = end_delta

        self.n_slow = n_slow
        self.n_speedup = n_speedup

        self.variance_source = VarianceSource(
            n_frames,
            variances,
            task,
            noise_type,
            with_label=False,
            spacing=start_spacing,
            delta=start_delta,
        )

        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.variance_source.initialize(fig, ax, last_animation)

        num = self.variance_source.rained_histogram.num
        spacing = numpy.zeros(num)
        spacing[1 : self.n_slow] = self.start_spacing
        spacing[self.n_slow : self.n_slow + self.n_speedup] = numpy.exp(
            numpy.linspace(
                numpy.log(self.start_spacing),
                numpy.log(self.end_spacing),
                num=self.n_speedup,
            )
        )

        # )
        #
        spacing[self.n_slow + self.n_speedup :] = self.end_spacing
        self.variance_source.rained_histogram.data_steps = -spacing.cumsum()
        # (
        #     numpy.arange(num)[::-1] - num
        # ).astype(float) * spacing

        # print(spacing)
        # delta = numpy.linspace(self.start_delta, self.end_delta, num=data.shape[0])
        # print(delta)
        # self.variance_source.spacing = spacing
        # self.variance_source.delta = delta

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if i < self.n_frames_start:
            spacing = self.start_spacing
            delta = self.start_delta
        elif i < self.n_frames_start + self.n_frames_speedup:
            spacing = linear(
                self.start_spacing,
                self.end_spacing,
                i - self.n_frames_start,
                self.n_frames_speedup,
            )
            delta = linear(
                self.start_delta,
                self.end_delta,
                i - self.n_frames_start,
                self.n_frames_speedup,
            )
        else:
            spacing = self.end_spacing
            delta = self.end_delta

        # self.variance_source.spacing = spacing
        self.variance_source.rained_histogram.delta = delta
        self.variance_source(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class VarianceSource:
    def __init__(
        self,
        n_frames,
        variances,
        task,
        noise_type,
        spacing=10,
        delta=20,
        with_label=True,
    ):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.noise_type = noise_type
        self.spacing = spacing
        self.delta = delta
        self.initialized = False

    @property
    def hist_y(self):
        return self.variances.get_weights_y(self.noise_type)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.noise_type not in self.variances.get_data(self.task):
            self.initialized = True
            return

        self.rained_histogram = RainedHistogram(
            self.variances.scatters[self.task],
            self.variances.get_data(self.task)[self.noise_type],
            y_min=self.hist_y,
            y_max=self.hist_y + self.variances.hist_height * 0.5,
            y_sky=self.variances.hist_heights * 1.2,
            steps=self.n_frames,
            spacing=self.spacing,  # Probably need to adjust based on which y_index it is
            delta=self.delta,  # Probably need to adjust based on which y_index it is
            subset=self.variances.get_slice(
                self.task, self.noise_type
            ),  # Need to adjust based on which noise type it is
            n_columns=self.variances.n_columns,
            marker_size=50,
            y_padding=0.15,
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if self.noise_type not in self.variances.get_data(self.task):
            return

        self.rained_histogram.step(i)
        hit_the_ground = self.rained_histogram.get()
        rain_std(self.variances.get_bar(self.task, self.noise_type), hit_the_ground)

    def leave(self):
        self(self.n_frames, None, None, None)


class VarianceSum:
    def __init__(self, n_frames, variances):
        self.n_frames = n_frames
        self.variances = variances

        self.base = "bootstrapping_seed"
        self.labels = [
            "init_seed",
            "sampler_seed",
            "transform_seed",
            "random_search",
        ]

        self.initialized = False

    def get_label_index(self, label):
        return self.variances.label_order.index(label)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.highlight = VariancesHighlight(
            self.n_frames,
            self.variances,
            noise_types=["everything"] + [self.base] + self.labels,
            vbars=[],
        )
        self.highlight.initialize(fig, ax, last_animation)

        n_variances = 9

        self.bars = self.variances.bar_axes["vgg"].barh(
            [self.get_label_index(self.base)] * len(self.labels),
            [0] * len(self.labels),
            left=0,
            align="edge",
            color=[
                variances_colors(get_variance_color(label)) for label in self.labels
            ],
            clip_on=False,
            zorder=zorder.get(),
        )

        base = self.variances.bars["vgg"][self.get_label_index(self.base)]
        bbox = base.get_bbox()
        self.base_width = bbox.x1
        self.total_width = bbox.x1
        for bar, label in zip(self.bars, self.labels):
            original_bar = self.variances.bars["vgg"][self.get_label_index(label)]
            width = original_bar.get_width()
            self.total_width += width

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        self.highlight(i, fig, ax, last_animation)

        if i < FPS:
            return

        j = i - FPS

        width = translate(
            self.base_width, self.total_width, j, self.n_frames / 2, saturation=5
        )

        base = self.variances.bars["vgg"][self.get_label_index(self.base)]
        bbox = base.get_bbox()
        x = bbox.x1
        y = bbox.y0
        total = bbox.x1
        for k, [bar, label] in enumerate(zip(self.bars, self.labels)):
            original_bar = self.variances.bars["vgg"][self.get_label_index(label)]
            bar_width = original_bar.get_width()
            current_width = max(min(bar_width + total, width) - total, 0)
            if current_width + x > 0.007:
                correction = current_width + x - 0.007
                # This will break if first bar already overflow
                for p_bar in [base] + list(self.bars[: k - 1]):
                    bbox = p_bar.get_bbox()
                    p_bar.set_xy((bbox.x0 - correction, y))
                x -= correction

            bar.set_xy((x, y))
            bar.set_width(current_width)
            x += current_width
            total += current_width

        if i < self.n_frames * 2 / 3:
            return

        j = i - self.n_frames * 2 / 3

        base = self.variances.bars["vgg"][self.get_label_index(self.base)]
        bbox = base.get_bbox()

        if getattr(self, "base_offset", None) is None:
            self.base_offset = bbox.x0

        base_offset = translate(
            self.base_offset,
            0,
            j,
            self.n_frames / 4,
            saturation=10,
        )

        delta = bbox.x0 - base_offset
        base.set_xy((base_offset, bbox.y0))

        prop = translate(1, 0, j, self.n_frames / 15, saturation=10)

        for bar, label in zip(self.bars, self.labels):
            original_bar = self.variances.bars["vgg"][self.get_label_index(label)]
            bar.set_width(prop * original_bar.get_width())
            # bbox = bar.get_bbox()
            # bar.set_xy((bbox.x0 + delta, bbox.y0))

    def leave(self):
        pass


class VariancesHighlight:
    def __init__(
        self, n_frames, variances, noise_types, vbars, ratio_in=0.05, ratio_out=0.95
    ):
        self.n_frames = n_frames
        self.n_frames_close_in = int(n_frames * ratio_in)
        self.n_frames_close_out = int(n_frames * ratio_out)
        self.variances = variances
        self.noise_types = noise_types
        self.vbars = vbars
        self.initialized = False

    @property
    def hist_y(self):
        return self.variances.get_weights_y(self.noise_type)

    def get_label_y(self, label):
        return self.variances.get_label_y(label) - self.variances.get_label_height() / 2

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return
        self.grey_patches = {}
        for label in self.variances.label_order:
            if label in self.noise_types:
                continue
            self.grey_patches[label] = patches.Rectangle(
                (0, self.get_label_y(label)),
                0,
                self.variances.get_label_height(),
                fill=True,
                color="black",
                alpha=0.5,
                zorder=zorder.get(),
                transform=fig.transFigure,
                linewidth=0,
            )
            fig.patches.append(self.grey_patches[label])
            # self.variances.bar_axes["vgg"].add_patch(self.grey_patches[label])
        # self.black_patch.set_width(5)
        # self.black_patch.set_height(5)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if self.n_frames_close_in < i < self.n_frames_close_out:
            return

        if i <= self.n_frames_close_in:
            new_width = translate(0, 1, i, self.n_frames_close_in, saturation=10)
        else:
            new_width = translate(
                1,
                0,
                i - self.n_frames_close_out,
                self.n_frames - self.n_frames_close_out,
                saturation=10,
            )

        for rectangle in self.grey_patches.values():
            rectangle.set_width(new_width)

    def leave(self):
        self(self.n_frames, None, None, None)


class NormalHighlight:
    def __init__(self, n_frames, variances):
        self.n_frames = n_frames
        self.n_frames_close_in = n_frames * 0.05
        self.n_frames_close_out = n_frames * 0.95
        self.variances = variances
        self.initialized = False

    @property
    def hist_y(self):
        return self.variances.get_weights_y(self.noise_type)

    def get_label_y(self, label):
        return self.variances.get_label_y(label) - self.variances.get_label_height() / 2

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return
        self.grey_patches = {}
        for task in self.variances.keys:
            bbox = self.variances.bar_axes[task].get_position()
            self.grey_patches[task] = patches.Rectangle(
                (bbox.x0, bbox.y0),
                bbox.width,
                0,
                self.variances.get_label_height(),
                fill=True,
                color="black",
                alpha=0.3,
                zorder=zorder.get(),
                transform=fig.transFigure,
                linewidth=0,
            )
            fig.patches.append(self.grey_patches[task])
            # self.variances.bar_axes["vgg"].add_patch(self.grey_patches[label])
        # self.black_patch.set_width(5)
        # self.black_patch.set_height(5)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if self.n_frames_close_in < i < self.n_frames_close_out:
            return

        bbox = self.variances.bar_axes["vgg"].get_position()

        if i < self.n_frames_close_in:
            new_height = translate(
                0, bbox.height, i, self.n_frames_close_in, saturation=10
            )
        else:
            new_height = translate(
                bbox.height,
                0,
                i - self.n_frames_close_out,
                self.n_frames - self.n_frames_close_out,
                saturation=10,
            )

        for rectangle in self.grey_patches.values():
            rectangle.set_height(new_height)

    def leave(self):
        self(self.n_frames, None, None, None)


class VariancesFlushHist:
    def __init__(self, n_frames, variances):
        self.n_frames = n_frames
        self.variances = variances
        self.n_tasks = len(variances.keys)
        self.past_ax_padding = self.variances.ax_max_padding / ((self.n_tasks) * 2)
        self.new_ax_padding = self.variances.ax_max_padding / ((self.n_tasks) * 2)
        remaining_space = (
            1 - self.variances.label_width - self.past_ax_padding * self.n_tasks
        )
        self.ax_past_width = remaining_space / (self.n_tasks * 2)
        self.ax_past_width = min(self.ax_past_width, self.variances.ax_max_width)
        remaining_space = (
            1 - self.variances.label_width - self.new_ax_padding * self.n_tasks
        )
        self.ax_new_width = remaining_space / (self.n_tasks)
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_std_x = self.variances.standard_deviation_label.get_position()[0]

        for scatter in self.variances.scatters.values():
            scatter.set_clip_on(True)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for ith_task in range(self.n_tasks):
            task = self.variances.keys[ith_task]
            # Do your work
            new_bar_width = 0.95 * translate(
                self.ax_past_width,
                self.ax_new_width,
                i,
                self.n_frames / 2,
            )
            new_hist_width = 0.95 * translate(
                self.ax_past_width,
                0,
                i,
                self.n_frames / 2,
            )
            new_padding = 0.95 * translate(
                self.past_ax_padding,
                self.new_ax_padding,
                i,
                self.n_frames / 2,
            )

            self.variances.hist_axes[task].set_position(
                [
                    self.variances.label_width
                    + ith_task * (new_bar_width + new_hist_width + 2 * new_padding),
                    self.variances.ax_bottom_padding,
                    new_hist_width,
                    self.variances.ax_height,
                ]
            )
            self.variances.bar_axes[task].set_position(
                [
                    self.variances.label_width
                    + ith_task * (new_bar_width + new_hist_width + 2 * new_padding)
                    + new_hist_width
                    + new_padding,
                    self.variances.ax_bottom_padding,
                    new_bar_width,
                    self.variances.ax_height,
                ]
            )

            # self.variances.task_label_objects[task].set_position(
            #     (
            #         self.variances.label_width
            #         + (2 * ith_task) * (new_width + new_padding)
            #         + new_width
            #         + new_padding / 2,
            #         self.variances.title_y,
            #     )
            # )

        # new_y =

        (x, y) = self.variances.performances_label.get_position()
        new_y = translate(
            self.variances.x_axis_label_y, -0.1, i, self.n_frames, saturation=3
        )
        self.variances.performances_label.set_position((x, new_y))

        (_, y) = self.variances.standard_deviation_label.get_position()
        new_x = translate(self.old_std_x, x, i, self.n_frames, saturation=3)
        self.variances.standard_deviation_label.set_position((new_x, y))

    def leave(self):
        self(self.n_frames, None, None, None)


class SqueezeTask:
    def __init__(self, n_frames, variances, task):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task

    @property
    def hist_y(self):
        return self.variances.get_weights_y(self.noise_type)

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        self.variances.bar_axes[self.task].set_position(
            [0.2, 0.1, translate(0.2, 0.08, i, self.n_frames), 0.8]
        )
        self.variances.hist_axes[self.task].set_position(
            [
                translate(0.2, 0.2 + 0.08, i, self.n_frames),
                0.1,
                translate(0.2, 0.08, i, self.n_frames),
                0.8,
            ]
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class VarianceTask:
    def __init__(self, n_frames, variances, task):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.variance_sources = []
        self.nth_task = variances.keys.index(task) + 1
        self.past_ax_padding = self.variances.ax_max_padding / ((self.nth_task) * 2)
        self.new_ax_padding = self.variances.ax_max_padding / ((self.nth_task + 1) * 2)
        remaining_space = (
            1 - self.variances.label_width - self.past_ax_padding * self.nth_task
        )
        self.ax_past_width = remaining_space / ((self.nth_task - 1) * 2)
        self.ax_past_width = min(self.ax_past_width, self.variances.ax_max_width)
        remaining_space = (
            1 - self.variances.label_width - self.new_ax_padding * (self.nth_task + 1)
        )
        self.ax_new_width = remaining_space / ((self.nth_task) * 2)
        for source in variances.label_order:
            self.variance_sources.append(
                VarianceSource(
                    n_frames, variances, task, source, delta=50, with_label=False
                )
            )

        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for source in self.variance_sources:
            source.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for ith_task in range(self.nth_task):
            task = self.variances.keys[ith_task]
            # Do your work
            new_width = 0.95 * translate(
                self.ax_past_width,
                self.ax_new_width,
                i,
                self.n_frames / 2,
            )
            new_padding = 0.95 * translate(
                self.past_ax_padding,
                self.new_ax_padding,
                i,
                self.n_frames / 2,
            )

            self.variances.hist_axes[task].set_position(
                [
                    self.variances.label_width
                    + ith_task * 2 * (new_width + new_padding),
                    self.variances.ax_bottom_padding,
                    new_width,
                    self.variances.ax_height,
                ]
            )
            self.variances.bar_axes[task].set_position(
                [
                    self.variances.label_width
                    + (ith_task * 2 + 1) * (new_width + new_padding),
                    self.variances.ax_bottom_padding,
                    new_width,
                    self.variances.ax_height,
                ]
            )

            self.variances.task_label_objects[task].set_position(
                (
                    self.variances.label_width
                    + (2 * ith_task) * (new_width + new_padding)
                    + new_width
                    + new_padding / 2,
                    self.variances.title_y,
                )
            )

            if task == self.task:
                break

        self.variances.performances_label.set_position(
            (
                self.variances.label_width + new_width / 2,
                self.variances.x_axis_label_y,
            )
        )

        self.variances.standard_deviation_label.set_position(
            (
                self.variances.label_width + new_width * 3 / 2 + new_padding,
                self.variances.x_axis_label_y,
            )
        )

        if self.nth_task > 1:
            self.variances.standard_deviation_label.set_text("STD")
        if self.nth_task > 2:
            self.variances.performances_label.set_text("Perf.")

        for source in self.variance_sources:
            source(max(i - int(self.n_frames / FPS), 0), fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class Algorithms:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        fig.clear()
        # Use a specific ax for comments otherwise text are in cmap='grey' and the rendering is
        # terrible.
        self.comment_ax = fig.add_axes([0, 0, 0, 0], zorder=zorder())
        self.ideal_ax = fig.add_axes([0.15, 0.15, 0.4, 0.8])
        self.ideal_img = mpimg.imread("algorithms_ideal.png")
        self.ideal_ax.imshow(self.ideal_img, cmap="gray", interpolation="none")

        self.biased_ax = fig.add_axes([1.2, 0.125, 0.4, 0.82])
        self.biased_img = mpimg.imread("algorithms_biased.png")
        self.biased_ax.imshow(self.biased_img, cmap="gray", interpolation="none")

        for axis in [self.ideal_ax, self.biased_ax]:
            for side in ["top", "right", "bottom", "left"]:
                axis.spines[side].set_visible(False)
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

        self.top_white_box = patches.Rectangle(
            (0, 0.84),
            1,
            0,
            fill=True,
            color="black",
            alpha=0.6,
            zorder=zorder(),
            transform=fig.transFigure,
            linewidth=0,
        )
        fig.patches.append(self.top_white_box)
        self.bottom_white_box = patches.Rectangle(
            (0, 0),
            1,
            -1,
            fill=True,
            color="black",
            alpha=0.6,
            zorder=zorder.get(),
            transform=fig.transFigure,
            linewidth=0,
        )
        fig.patches.append(self.bottom_white_box)

        self.fade_out = reverse(FadeOut(FADE_OUT / 2))
        self.fade_out.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.fade_out(i, fig, ax, last_animation)

    def leave(self):
        pass


class CodeHighlight:
    def __init__(self, n_frames, lines, comment=None, comment_side=None):
        self.n_frames = n_frames
        self.lines = lines
        self.comment = comment
        self.comment_side = comment_side
        self.comment_x = 0.25 if comment_side == "left" else 0.8
        self.line_height = 0.043
        self.n_lines = 18
        self.padding = 0.005
        self.initialized = False

        self.new_height = -(self.lines[0] - 1) * self.line_height
        if self.new_height:
            self.new_height += self.padding
        n_lines = self.lines[-1] - self.lines[0] + 1
        self.new_gap = n_lines * self.line_height + self.padding

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.comment_ax = last_animation.comment_ax
        self.top_white_box = last_animation.top_white_box
        self.bottom_white_box = last_animation.bottom_white_box

        self.old_height = self.top_white_box.get_height()

        top_bbox = self.top_white_box.get_bbox()
        bottom_bbox = self.bottom_white_box.get_bbox()
        self.old_gap = top_bbox.y1 - bottom_bbox.y0

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        new_height = translate(
            self.old_height, self.new_height, i, self.n_frames / 10, saturation=10
        )

        # self.top_white_box.set_xy((0, 0.8))
        self.top_white_box.set_height(new_height)

        new_gap = translate(
            self.old_gap, self.new_gap, i, self.n_frames / 10, saturation=10
        )

        bbox = self.top_white_box.get_bbox()
        self.bottom_white_box.set_xy((0, bbox.y1 - new_gap))

        if self.comment and i > self.n_frames / 10:
            self.comment_ax.text(
                self.comment_x,
                bbox.y1 - new_gap + self.padding,
                self.comment,
                va="bottom",
                ha="right" if self.comment_side == "left" else "left",
                fontsize=24,
                transform=fig.transFigure,
                zorder=zorder.get(),
            )

    def leave(self):
        pass


class BringInBiasedEstimator:
    def __init__(self, n_frames, algorithms):
        self.n_frames = n_frames
        self.initialized = False
        self.algorithms = algorithms

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_x = self.algorithms.biased_ax.get_position().x0

        # To provide for the next CodeHighlight
        self.comment_ax = last_animation.comment_ax
        self.top_white_box = last_animation.top_white_box
        self.bottom_white_box = last_animation.bottom_white_box

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        new_x = translate(self.old_x, 0.535, i, self.n_frames / 5, saturation=10)

        bbox = self.algorithms.biased_ax.get_position()
        self.algorithms.biased_ax.set_position(
            [new_x, bbox.y0, bbox.width, bbox.height]
        )

    def leave(self):
        self(self.n_frames, None, None, None)


# TODO: Decouple moving and linear scale
#       Make one animation to group them and run them at the same time.
#       Reuse linear scale in the BulletPoint


class SlidingAxWhiteBackground:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False
        self.ax_height = 0.55
        self.white_box_height = 0.835

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.white_box = patches.Rectangle(
            (0.15, 0),
            1,
            0,
            fill=True,
            color="white",
            alpha=1,
            zorder=zorder.get() - 1,
            transform=fig.transFigure,
            linewidth=0,
        )
        fig.patches.append(self.white_box)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.white_box.set_height(
            translate(0, self.white_box_height, i, self.n_frames / 10, saturation=10)
        )
        bbox = self.ax.get_position()
        self.ax.set_position(
            [
                bbox.x0,
                bbox.y0,
                bbox.width,
                translate(0, self.ax_height, i, self.n_frames / 5, saturation=10),
            ]
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class SlidingSimpleLinearScale:
    def __init__(self, n_frames, ax=None):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.linear_scale = SimpleLinearScale(self.n_frames)
        self.linear_scale.initialize(fig, ax, last_animation)
        self.sliding_movement = SlidingAxWhiteBackground(
            self.n_frames, self.linear_scale.simple_curve_ax
        )
        self.sliding_movement.initialize(fig, ax, last_animation)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.linear_scale(i, fig, ax, last_animation)
        self.sliding_movement(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class SimpleLinearScale:
    def __init__(
        self,
        n_frames,
        legend_fontsize=18,
        title_fontsize=38,
        axis_label_fontsize=32,
        axis_tick_fontsize=16,
        ax=None,
    ):
        self.n_frames = n_frames
        self.simple_curve_ax = ax
        self.initialized = False
        self.title_fontsize = title_fontsize
        self.axis_label_fontsize = axis_label_fontsize
        self.axis_tick_fontsize = axis_tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.T = 100
        self.x_max = 200

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.simple_curve_ax is None:
            self.simple_curve_ax = fig.add_axes([0.3, 0.2, 0.4, 0], zorder=zorder(2))
        self.ideal_curve = self.simple_curve_ax.plot(
            [],
            [],
            linewidth=5,
            label="IdealEst($k$)",
            color=EST_COLORS["IdealEst($k$)"],
        )[0]
        self.biased_curve = self.simple_curve_ax.plot(
            [],
            [],
            linewidth=5,
            label="FixHOptEst($k$)",
            color=EST_COLORS["FixHOptEst($k$, Init)"],
        )[0]
        self.simple_curve_ax.set_xlim((0, self.x_max))
        self.simple_curve_ax.set_ylim((0, self.x_max * self.T))
        self.legend = self.simple_curve_ax.legend(
            fontsize=self.legend_fontsize, loc="upper left"
        )
        self.x_label = "Sample size"
        self.y_label = "Number of trainings"
        self.simple_curve_ax.set_xlabel(self.x_label, fontsize=self.axis_label_fontsize)
        self.simple_curve_ax.set_ylabel(self.y_label, fontsize=self.axis_label_fontsize)
        self.simple_curve_ax.spines["top"].set_visible(False)
        self.simple_curve_ax.spines["right"].set_visible(False)

        self.simple_curve_ax.tick_params(
            axis="both", which="major", labelsize=self.axis_tick_fontsize
        )

        # self.variances.bar_axes["vgg"].add_patch(self.grey_patches[label])
        # self.black_patch.set_width(5)
        # self.black_patch.set_height(5)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        x = numpy.linspace(0, i / self.n_frames * self.x_max, 2)
        self.ideal_curve.set_xdata(x)
        self.ideal_curve.set_ydata(self.T * x)
        self.biased_curve.set_xdata(x)
        self.biased_curve.set_ydata(self.T + x)

    def leave(self):
        self(self.n_frames, None, None, None)


class MoveSimpleLinearScale:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

        self.new_x = 0.05
        self.new_y = 0.4
        self.new_width = 0.15
        self.new_height = 0.3
        self.new_label_font_size = 10
        self.new_legend_font_size = 10

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        bbox = last_animation.linear_scale.simple_curve_ax.get_position()
        self.old_x = bbox.x0
        self.old_y = bbox.y0
        self.old_width = bbox.width
        self.old_height = bbox.height

        self.x_label = last_animation.linear_scale.x_label
        self.y_label = last_animation.linear_scale.y_label
        self.simple_curve_ax = last_animation.linear_scale.simple_curve_ax
        self.legend = last_animation.linear_scale.legend
        self.ideal_curve = last_animation.linear_scale.ideal_curve
        self.biased_curve = last_animation.linear_scale.biased_curve

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        new_x = translate(self.old_x, self.new_x, i, self.n_frames, saturation=5)
        new_y = translate(self.old_y, self.new_y, i, self.n_frames, saturation=5)
        new_width = translate(self.old_width, self.new_width, i, self.n_frames)
        new_height = translate(self.old_height, self.new_height, i, self.n_frames / 2)
        self.simple_curve_ax.set_position(
            [
                new_x,
                new_y,
                new_width,
                new_height,
            ]
        )
        for text in self.legend.get_texts():
            text.set_fontsize(self.new_legend_font_size)

        new_line_width = translate(5, 2, i, self.n_frames, saturation=5)

        for line in self.legend.get_lines():
            line.set_linewidth(new_line_width)
        for curve in [self.ideal_curve, self.biased_curve]:
            curve.set_linewidth(new_line_width)

        new_xtick_fontsize = translate(16, 4, i, self.n_frames, saturation=5)
        self.simple_curve_ax.tick_params(
            axis="x", which="major", labelsize=new_xtick_fontsize
        )
        new_ytick_fontsize = translate(16, 0, i, self.n_frames, saturation=5)
        self.simple_curve_ax.tick_params(
            axis="y", which="major", labelsize=new_ytick_fontsize
        )

        new_label_fontsize = translate(
            32, self.new_label_font_size, i, self.n_frames, saturation=5
        )
        self.simple_curve_ax.set_xlabel(self.x_label, fontsize=new_label_fontsize)
        self.simple_curve_ax.set_ylabel(self.y_label, fontsize=new_label_fontsize)

    def leave(self):
        self(self.n_frames, None, None, None)


class VarianceEquations:
    def __init__(self):
        self.n_frames = 0
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ideal_ax = fig.add_axes([0.26, 0.625, 0.2, 0.1], zorder=zorder())
        self.ideal_img = mpimg.imread("ideal_var.png")
        self.ideal_ax.imshow(self.ideal_img, cmap="gray", interpolation="none")

        self.biased_ax = fig.add_axes([0.53, 0.53, 0.42, 0.3], zorder=zorder.get())
        self.biased_img = mpimg.imread("biased_var.png")
        self.biased_ax.imshow(self.biased_img, cmap="gray", interpolation="none")

        for axis in [self.ideal_ax, self.biased_ax]:
            for side in ["top", "right", "bottom", "left"]:
                axis.spines[side].set_visible(False)
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class EstimatorSimulation:
    def __init__(self):
        self.n_frames = 0
        self.initialized = False
        self.n_rows = 20
        self.min_k = 3
        self.max_k = 100
        self.k_text_template = "$k$={k}"
        self.rho_text_template = "$\\rho={rho:0.2f}\\sigma^2$"

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.axes = {}
        self.axes["left"] = fig.add_axes([0.235, 0.025, 0.23, 0.5], zorder=zorder())
        self.axes["right"] = fig.add_axes(
            [0.535, 0.025, 0.23, 0.5], zorder=zorder.get()
        )

        for ax in self.axes.values():
            ax.set_xlim((-5, 5))
            ax.set_ylim((-1, 22))
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        self.estimators = {}
        for key in self.axes.keys():
            self.estimators[key] = EstimatorBubbles(
                self.axes[key],
                n_rows=self.n_rows,
                std=1,
                rho=0,
                color=EST_COLORS[
                    "IdealEst($k$)" if key == "left" else "FixHOptEst($k$, Init)"
                ],
            )

        self.k_text = self.axes["left"].text(
            0.5,
            0.25,
            self.k_text_template.format(k=self.min_k),
            transform=fig.transFigure,
            va="center",
            ha="center",
            fontsize=18,
        )
        self.rho_text = self.axes["left"].text(
            0.78,
            0.25,
            self.rho_text_template.format(rho=0),
            transform=fig.transFigure,
            va="center",
            ha="left",
            fontsize=18,
        )

        # Sync them
        self.estimators["right"].data = self.estimators["left"].data

        for i in range(self.min_k):
            self.estimators["left"].increase_k()
        self.estimators["right"].refresh()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class EstimatorIncreaseK:
    def __init__(self, n_frames, estimators):
        self.n_frames = n_frames
        self.estimators = estimators
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        k = linear(self.estimators.min_k, self.estimators.max_k, i, self.n_frames)

        while self.estimators.estimators["left"].get_k() < k:
            self.estimators.estimators["left"].increase_k()
            self.estimators.estimators["right"].refresh()

        self.estimators.k_text.set_text(
            self.estimators.k_text_template.format(k=int(k))
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class EstimatorAdjustRho:
    def __init__(self, n_frames, estimators, new_rho):
        self.n_frames = n_frames
        self.estimators = estimators
        self.new_rho = new_rho
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_rho = self.estimators.estimators["right"].rho

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        rho = linear(self.old_rho, self.new_rho, i, self.n_frames)

        self.estimators.estimators["right"].adjust_rho(rho)

        self.estimators.rho_text.set_text(
            self.estimators.rho_text_template.format(rho=rho)
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class EstimatorTask:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.min_budgets = 2
        self.max_budgets = 100
        self.task = "bert-rte"
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.estimators = EstimatorsPlot()
        self.estimators.load()
        x = 1 if self.n_frames > 0 else 0.3
        self.ax = fig.add_axes([x, 0.575, 0.4, 0.2], zorder=zorder(2))
        self.ax.set_xlim((0, self.max_budgets))
        max_y = []
        for source in ["Init", "Data", "All"]:
            estimator = "FixHOptEst($k$, {var})".format(var=source)
            y, err = self.estimators.get_stat(
                self.task,
                estimator,
                numpy.arange(self.max_budgets) + 1,
                stat="std",
                standardized=False,
            )
            max_y.append(max(y))
        self.ax.set_ylim((0, max(max_y) * 1.1))
        sns.despine(ax=self.ax, bottom=False, left=False)
        self.estimators.plot(
            self.ax,
            self.task,
            # budgets=numpy.arange(self.max_budgets) + 1
            budgets=numpy.arange(self.min_budgets) + 1,
        )

        self.ax.legend(
            bbox_to_anchor=(1.05, 0, 0.5, 1),
            loc="lower left",
            ncol=1,
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
            fontsize=16,
        )

        self.ax.set_xlabel("Number of samples\nfor the estimator ($k$)", fontsize=12)
        self.ax.set_ylabel("Standard deviation\nof estimators\n(Accuracy)", fontsize=12)
        self.ax.text(
            50,
            max(max_y) * 0.8,
            LABELS[self.task],
            va="bottom",
            ha="center",
            fontsize=16,
        )

        self.ax.tick_params(axis="x", which="major", labelsize=12)
        self.ax.tick_params(axis="y", which="major", labelsize=12)

        self.white_patch = patches.Rectangle(
            (1, 0.575),
            1,
            0.2,
            fill=True,
            color="white",
            alpha=1,
            zorder=zorder.get() - 1,
            transform=fig.transFigure,
            linewidth=0,
        )
        fig.patches.append(self.white_patch)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if self.n_frames == 0:
            return
        x = translate(1, 0.3, i, self.n_frames / 10, saturation=10)
        bbox = self.white_patch.get_bbox()
        self.white_patch.set_xy((x * 0.95, bbox.y0))
        x = translate(1, 0.3, i, self.n_frames / 2, saturation=10)
        bbox = self.ax.get_position()
        self.ax.set_position((x, bbox.y0, bbox.width, bbox.height))

    def leave(self):
        self(self.n_frames, None, None, None)


class EstimatorShow:
    def __init__(self, n_frames, estimators, estimator_task, estimator):
        self.n_frames = n_frames
        self.estimators = estimators
        self.estimator_task = estimator_task
        self.estimator = estimator
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_rho = self.estimators.estimators["right"].rho

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        max_budgets = int(
            linear(
                self.estimator_task.min_budgets,
                self.estimator_task.max_budgets,
                i,
                self.n_frames,
            )
        )
        budgets = numpy.arange(max_budgets) + 1
        self.estimator_task.estimators.update(
            self.estimator_task.ax,
            self.estimator_task.task,
            budgets,
            estimators=[self.estimator],
        )

    def leave(self):
        self(self.n_frames, None, None, None)
        if self.estimator != "IdealEst($k$)":
            self.estimators.estimators["right"].estimator = self.estimator


class ComparisonMethod:
    def __init__(
        self,
        n_frames,
        method,
        x_padding,
        width=0.4,
        y_margin=0.5,
        height=0.2,
        despine=True,
    ):
        self.n_frames = n_frames
        self.method = method
        self.x_padding = x_padding
        self.initialized = False
        self.models = []
        self.width = width
        self.y_margin = y_margin
        self.height = height
        self.xlim = (-10, 10)
        self.despine = despine

    def redraw(self):
        if hasattr(self, "sample_size_panel"):
            self.sample_size_panel.redraw()

        if hasattr(self, "method_object"):
            self.method_object.redraw()

        if hasattr(self, "labels"):
            for label in self.labels.values():
                label.redraw()

        for model in self.models:
            model.redraw()

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.method == "Average":
            fig.clear()

        self.ax = fig.add_axes(
            [self.x_padding, self.y_margin, self.width, self.height], zorder=zorder()
        )
        self.ax.text(
            self.x_padding + self.width / 2,
            0.8,
            self.method,
            ha="center",
            transform=fig.transFigure,
            fontsize=32,
        )

        if self.despine:
            despine(self.ax)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class AddModel:
    def __init__(
        self,
        n_frames,
        comparison,
        name,
        mean=-1,
        std=1,
        min_x=-10,
        max_x=10,
        scale=1,
        color=None,
        fontsize=16,
        clip_on=True,
    ):
        self.n_frames = n_frames
        self.comparison = comparison
        self.name = name
        self.mean = mean
        self.std = std
        self.min_x = min_x
        self.max_x = max_x
        self.scale = scale
        self.color = color
        self.fontsize = fontsize
        self.clip_on = clip_on
        self.comparison.models.append(self)
        self.initialized = False

    def redraw(self):
        x = numpy.linspace(self.min_x, self.max_x, 1000)
        y = scipy.stats.norm.pdf(x, self.mean, self.std)
        y /= max(self.max_y, max(y))
        self.y = y * self.scale
        self.line.set_ydata(self.y)
        self.name_label.set_position((self.mean, max(self.y)))

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        x = numpy.linspace(self.min_x, self.max_x, 1000)
        y = scipy.stats.norm.pdf(x, self.mean, self.std)
        self.max_y = max(y)
        y /= max(self.max_y, max(y))
        self.y = y * self.scale
        self.line = self.comparison.ax.plot(x, self.y, color=self.color)[0]

        self.name_label = self.comparison.ax.text(
            self.mean,
            -1,
            self.name,
            ha="center",
            va="bottom",
            fontsize=self.fontsize,
            clip_on=self.clip_on,
        )
        self.comparison.ax.set_ylim(0, 1)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        scale = translate(0, 1, i, self.n_frames)
        self.line.set_ydata(self.y * scale)
        self.name_label.set_position((self.mean, max(self.y) * scale))

    def leave(self):
        self(self.n_frames, None, None, None)


class ComputeAverages:
    def __init__(self, n_frames, comparison):
        self.n_frames = n_frames
        self.comparison = comparison
        self.comparison.method_object = self
        self.delta = 2
        self.initialized = False
        self.whisker_width = 0.2
        self.delta_padding = 0.2

    def redraw(self):
        A, B = self.comparison.models
        diff = A.mean - B.mean

        adjust_h_moustachos(
            self.comparison.avg_plot,
            x=B.mean + diff / 2,
            y=0,
            whisker_width=self.whisker_width,
            whisker_length=diff / 2,
            center_width=0,
        )

        adjust_h_moustachos(
            self.delta_plot,
            x=B.mean + self.delta / 2,
            y=-1,
            whisker_width=self.whisker_width,
            whisker_length=self.delta / 2,
            center_width=0,
        )

        self.delta_label.set_position((B.mean - self.delta_padding, -1))

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return
        self.comparison.avg_axe = fig.add_axes(
            [self.comparison.x_padding, 0.35, self.comparison.width, 0.1],
            zorder=zorder(),
        )
        despine(self.comparison.avg_axe)
        self.comparison.avg_axe.set_xlim(self.comparison.xlim)
        self.comparison.avg_axe.set_ylim((-1, 1))
        A, B = self.comparison.models
        diff = A.mean - B.mean
        self.comparison.avg_plot = h_moustachos(
            self.comparison.avg_axe,
            x=B.mean + diff / 2,
            y=0,
            whisker_width=self.whisker_width * 0.01,
            whisker_length=diff / 2 * 0.01,
            center_width=0,
            clip_on=False,
        )

        self.delta_plot = h_moustachos(
            self.comparison.avg_axe,
            x=B.mean + self.delta / 2,
            y=-1,
            whisker_width=self.whisker_width * 0.0000001,
            whisker_length=diff / 2 * 0.000001,
            center_width=0,
            clip_on=False,
        )

        self.delta_label = self.comparison.avg_axe.text(
            B.mean - self.delta_padding, -1, "", ha="right", va="center", fontsize=16
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        A, B = self.comparison.models
        diff = A.mean - B.mean

        whisker_width = translate(
            self.whisker_width * 0.01,
            self.whisker_width,
            i,
            self.n_frames / 10,
            saturation=5,
        )

        if i > self.n_frames / 10:
            y = translate(
                2,
                0,
                i - self.n_frames / 10,
                self.n_frames / 10,
                saturation=5,
            )
        else:
            y = 2

        adjust_h_moustachos(
            self.comparison.avg_plot,
            x=B.mean + diff / 2,
            y=y,
            whisker_width=whisker_width,
            whisker_length=diff / 2,
            center_width=0,
        )

        if i > self.n_frames / 10 * 3:
            self.delta_label.set_text("$\delta$")
            whisker_width = translate(
                self.whisker_width * 0.01,
                self.whisker_width,
                i - self.n_frames / 10 * 3,
                self.n_frames / 10,
                saturation=5,
            )
        else:
            whisker_width = 1e-10

        if i > self.n_frames / 10 * 4:
            whisker_length = translate(
                0, self.delta / 2, i - self.n_frames / 10 * 4, self.n_frames
            )
            x = B.mean + whisker_length
        else:
            x = B.mean
            whisker_length = 1e-10

        adjust_h_moustachos(
            self.delta_plot,
            x=x,
            y=-1,
            whisker_width=whisker_width,
            whisker_length=whisker_length,
            center_width=0,
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class CompAddPAB:
    def __init__(self, n_frames, comparison):
        self.n_frames = n_frames
        self.comparison = comparison
        self.comparison.method_object = self
        self.initialized = False
        self.whisker_width = 0.2
        self.ax_width = 0.2

    def redraw(self):
        A, B = self.comparison.models
        diff = B.mean - A.mean

        lower, pab, upper = self.compute_pab()
        adjust_h_moustachos(
            self.comparison.pab_plot,
            x=pab,
            y=0,
            whisker_width=self.whisker_width,  # * 0.01,
            whisker_length=(pab - lower, upper - pab),  #  * 0.01,
            center_width=self.whisker_width * 1.5,
        )

        self.pab_label.set_position((pab, PAB))

    def get_standardized_data(self):

        standardized_data = {}
        for model in self.comparison.models:
            data = self.data[model.name]
            standardized_data[model.name] = data * model.std + model.mean

        return standardized_data

    def compute_pab(self):
        data = self.get_standardized_data()
        pab_center = pab(data["A"], data["B"])
        ci = normal_ci(data["A"], data["B"], sample_size=50)
        lower = max(pab_center - ci, 0)
        upper = min(pab_center + ci, 1)
        # lowers = []
        # uppers = []
        # idx_choices = numpy.arange(data["A"].shape[0])
        # for i in range(100):
        #     idx = numpy.random.choice(idx_choices, size=50)
        #     lower, upper = percentile_bootstrap(
        #         data["A"][idx], data["B"][idx], alpha=0.05, bootstraps=100
        #     )
        #     lowers.append(lower)
        #     uppers.append(upper)
        # lower = numpy.array(lowers).mean()
        # upper = numpy.array(uppers).mean()
        return lower, pab_center, upper

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.data = {}
        for model in self.comparison.models:
            self.data[model.name] = numpy.random.normal(0, 1, size=10000)

        center = self.comparison.x_padding + self.comparison.width / 2
        x = center - self.ax_width / 2

        self.ax_start_position = [x + self.ax_width / 2, 0.35, 0, 0.1]
        self.ax_position = [x, 0.35, self.ax_width, 0.1]

        self.comparison.pab_axe = fig.add_axes(
            self.ax_position,
            zorder=zorder(),
        )

        sns.despine(ax=self.comparison.pab_axe)
        # self.comparison.pab_axe.get_xaxis().set_smart_bounds(True)
        # despine(self.comparison.avg_axe)
        self.comparison.pab_axe.spines["left"].set_visible(False)
        self.comparison.pab_axe.get_yaxis().set_visible(False)
        self.comparison.pab_axe.set_xlim((0, 1))
        self.comparison.pab_axe.set_ylim((-1, 1))
        lower, pab, upper = self.compute_pab()
        self.comparison.pab_plot = h_moustachos(
            self.comparison.pab_axe,
            x=pab,
            y=0,
            whisker_width=self.whisker_width * 1e-10,
            whisker_length=1e-10,
            center_width=self.whisker_width * 1.5 * 1e-10,
            clip_on=False,
        )
        self.saved_ci = (lower, pab, upper)

        self.pab_label = self.comparison.pab_axe.text(
            pab, 0.75, "", ha="center", va="bottom", fontsize=16, clip_on=False
        )

        self.gamma_tick = self.comparison.pab_axe.plot(
            [pab, pab], [-1000, -1000], color="black", clip_on=False
        )[0]

        self.gamma_label = self.comparison.pab_axe.text(
            pab, -2, "", ha="center", va="bottom", fontsize=16, clip_on=False
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        # Make axe appear smoothly
        # Add center of pab first
        # Add whiskers

        x = translate(
            self.ax_start_position[0], self.ax_position[0], i, self.n_frames / 10
        )
        width = translate(
            self.ax_start_position[2], self.ax_position[2], i, self.n_frames / 10
        )

        self.comparison.pab_axe.set_position(
            [x, self.ax_position[1], width, self.ax_position[3]]
        )

        if i > self.n_frames / 10 * 3.5:
            self.pab_label.set_text("$P(A>B)$")

        if i > self.n_frames / 10 * 3:
            center_width = translate(
                self.whisker_width * 1.5 * 1e-10,
                self.whisker_width * 1.5,
                i - self.n_frames / 10 * 3,
                self.n_frames / 10,
            )
        else:
            center_width = self.whisker_width * 1.5 * 1e-10

        whisker_width = self.whisker_width * 1e-10
        whisker_length = 1e-10

        adjust_h_moustachos(
            self.comparison.pab_plot,
            x=pab,
            y=0,
            whisker_width=whisker_width,
            whisker_length=whisker_length,
            center_width=center_width,
        )

    def leave(self):
        pass


class ComputePAB:
    def __init__(self, n_frames, comparison):
        self.n_frames = n_frames
        self.comparison = comparison
        self.comparison.method_object = self
        self.initialized = False
        self.whisker_width = 0.2
        self.sample_size = 22
        self.gamma = 0.75
        self.y_offset = 0.15

    def redraw(self, pab=None):
        if pab is None:
            A, B = self.comparison.models
            diff = B.mean - A.mean

            lower, pab, upper = self.compute_pab()
        else:
            lower, pab, upper = pab

        self.pab = (lower, pab, upper)

        adjust_h_moustachos(
            self.comparison.pab_plot,
            x=pab,
            y=0,
            whisker_width=self.whisker_width,  # * 0.01,
            whisker_length=(pab - lower, upper - pab),  #  * 0.01,
            center_width=self.whisker_width * 1.5,
        )

        self.pab_label.set_position((pab, PAB))
        self.gamma_tick.set_xdata([self.gamma, self.gamma])
        x, y = self.gamma_label.get_position()
        self.gamma_label.set_position((self.gamma, y))

    def get_standardized_data(self):

        standardized_data = {}
        for model in self.comparison.models:
            data = self.data[model.name]
            standardized_data[model.name] = data * model.std + model.mean

        return standardized_data

    def compute_pab(self):
        data = self.get_standardized_data()
        pab_center = pab(data["A"], data["B"])
        ci = normal_ci(data["A"], data["B"], sample_size=self.sample_size)
        lower = max(pab_center - ci, 0)
        upper = min(pab_center + ci, 1)
        # lowers = []
        # uppers = []
        # idx_choices = numpy.arange(data["A"].shape[0])
        # for i in range(100):
        #     idx = numpy.random.choice(idx_choices, size=50)
        #     lower, upper = percentile_bootstrap(
        #         data["A"][idx], data["B"][idx], alpha=0.05, bootstraps=100
        #     )
        #     lowers.append(lower)
        #     uppers.append(upper)
        # lower = numpy.array(lowers).mean()
        # upper = numpy.array(uppers).mean()
        return lower, pab_center, upper

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.data = {}
        for model in self.comparison.models:
            self.data[model.name] = numpy.random.normal(0, 1, size=10000)

        self.ax_width = self.comparison.width / 2

        center = self.comparison.x_padding + self.comparison.width / 2
        x = center - self.ax_width / 2

        y = self.comparison.y_margin - self.y_offset

        self.ax_start_position = [x + self.ax_width / 2, y, 0, 0.1]
        self.ax_position = [x, y, self.ax_width, 0.1]

        self.comparison.pab_axe = fig.add_axes(
            self.ax_position,
            zorder=zorder() - 2,
        )

        self.comparison.pab_axe.xaxis.set_major_locator(plt.MaxNLocator(3))
        self.comparison.pab_axe.xaxis.set_ticks([0, 0.5, 1])

        sns.despine(ax=self.comparison.pab_axe)
        # self.comparison.pab_axe.get_xaxis().set_smart_bounds(True)
        # despine(self.comparison.avg_axe)
        self.comparison.pab_axe.spines["left"].set_visible(False)
        self.comparison.pab_axe.get_yaxis().set_visible(False)
        self.comparison.pab_axe.set_xlim((0, 1))
        self.comparison.pab_axe.set_ylim((-1, 1))
        lower, pab, upper = self.compute_pab()
        self.pab = (lower, pab, upper)
        self.comparison.pab_plot = h_moustachos(
            self.comparison.pab_axe,
            x=pab,
            y=0,
            whisker_width=self.whisker_width * 1e-10,
            whisker_length=1e-10,
            center_width=self.whisker_width * 1.5 * 1e-10,
            clip_on=False,
        )
        self.saved_ci = (lower, pab, upper)

        self.pab_label = self.comparison.pab_axe.text(
            pab, 0.75, "", ha="center", va="bottom", fontsize=16, clip_on=False
        )

        self.gamma_tick = self.comparison.pab_axe.plot(
            [self.gamma, self.gamma], [-1000, -1000], color="black", clip_on=False
        )[0]

        self.gamma_label = self.comparison.pab_axe.text(
            pab, -2, "", ha="center", va="bottom", fontsize=16, clip_on=False
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        # Make axe appear smoothly
        # Add center of pab first
        # Add whiskers

        x = translate(
            self.ax_start_position[0], self.ax_position[0], i, self.n_frames / 10
        )
        width = translate(
            self.ax_start_position[2], self.ax_position[2], i, self.n_frames / 10
        )

        self.comparison.pab_axe.set_position(
            [x, self.ax_position[1], width, self.ax_position[3]]
        )

        if i > self.n_frames / 10 * 3.5:
            self.pab_label.set_text("$P(A>B)$")

        if i > self.n_frames / 10 * 3:
            center_width = translate(
                self.whisker_width * 1.5 * 1e-10,
                self.whisker_width * 1.5,
                i - self.n_frames / 10 * 3,
                self.n_frames / 10,
            )
        else:
            center_width = self.whisker_width * 1.5 * 1e-10

        # if i > self.n_frames / 10 * 5.5:
        #     self.gamma_label.set_text("$\gamma$")

        # if i > self.n_frames / 10 * 5:
        #     gamma_tick_y = translate(
        #         0,
        #         0.25,
        #         i - self.n_frames / 10 * 5,
        #         self.n_frames / 10,
        #     )
        # else:
        #     gamma_tick_y = 0

        # self.gamma_tick.set_ydata([-1 + gamma_tick_y, -1 - gamma_tick_y])

        lower, pab, upper = self.saved_ci
        # if i > self.n_frames / 10 * 8:
        #     whisker_width = self.whisker_width
        #     whisker_length = translate(
        #         1e-10,
        #         pab - lower,
        #         i - self.n_frames / 10 * 8,
        #         self.n_frames / 10,
        #     )
        # else:
        #     whisker_width = self.whisker_width * 1e-10
        #     whisker_length = 1e-10
        whisker_width = self.whisker_width * 1e-10
        whisker_length = 1e-10

        adjust_h_moustachos(
            self.comparison.pab_plot,
            x=pab,
            y=0,
            whisker_width=whisker_width,
            whisker_length=whisker_length,
            center_width=center_width,
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class ChangeDists:
    def __init__(self, n_frames, comparisons, foo):
        self.n_frames = n_frames
        self.comparisons = comparisons
        self.foo = foo
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old = {}
        for comparison in self.comparisons:
            self.old[comparison.method] = {}
            for model in comparison.models:
                self.old[comparison.method][model.name] = {
                    "mean": model.mean,
                    "std": model.std,
                }

        self.new = {}
        for comparison in self.comparisons:
            new_models = self.foo(*comparison.models)
            self.new[comparison.method] = {}
            for i, model in enumerate(comparison.models):
                self.new[comparison.method][model.name] = {
                    "mean": new_models["mean"][i],
                    "std": new_models["std"][i],
                }

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for comparison in self.comparisons:
            for j, model in enumerate(comparison.models):
                ith_stats = {}
                for stat in ["mean", "std"]:
                    ith_stats[stat] = linear(
                        self.old[comparison.method][model.name][stat],
                        self.new[comparison.method][model.name][stat],
                        i,
                        self.n_frames,
                    )

                model.mean = ith_stats["mean"]
                model.std = ith_stats["std"]

            comparison.redraw()

    def leave(self):
        self(self.n_frames, None, None, None)


class PABDists:
    def __init__(self, ax, simulation, scale):
        self.ax = ax
        self.simulation = simulation
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_ylim((0, 1))
        self.ax.plot([0, 0], [1, 1.4], clip_on=False, linewidth=0.75, color="black")
        self.models = []
        self.scale = scale

    def set_pab(self, pab):
        self.simulation.set_pab(pab)
        for i, model in enumerate(self.models):
            model.mean = self.simulation.means[i]
            model.redraw()


class PABScatter:
    def __init__(self, ax, simulation, n_rows):
        self.ax = ax
        self.n_rows = n_rows
        self.simulation = simulation

        ax.set_xlim(-simulation.stds[0] * 5, simulation.stds[0] * 5)
        ax.set_ylim(-0.5, 20.5)
        despine(self.ax)
        self.estimator_name = "IdealEst"
        self.y_label = "{estimator} simulations"
        self.y_label_object = ax.set_ylabel(
            "",
            fontsize=18,
            horizontalalignment="right",
            y=0.9,
        )

        self.scatter = SimulationScatter(
            ax,
            simulation,
            n_rows=self.n_rows,
            colors=dict(A=variances_colors(0), B=variances_colors(1)),
        )
        self.scatter.redraw()

    def open(self, i, n_frames):
        y_label = self.y_label.format(estimator=self.estimator_name)
        i = int(numpy.round(linear(0, len(y_label), i, n_frames)))
        self.y_label_object.set_text(y_label[:i])

    def simulate(self, i, n_frames, model, lines):
        self.scatter.simulate(i, n_frames, model, lines)

    def redraw(self):
        self.scatter.redraw()

    def set_pab(self, pab):
        self.simulation.set_pab(pab)
        self.redraw()


class PABComparison:
    def __init__(self, ax, simulation, n_rows):
        self.ax = ax
        self.simulation = simulation
        self.n_rows = n_rows
        self.test_viz = None
        despine(self.ax)

    def set_test(self, test):
        # Trying to set the same test
        if self.test_viz is not None and self.test_viz.test is test:
            return

        # Test will be changed, clear axis from previous one
        if self.test_viz is not None:
            self.ax.clear()
            despine(self.ax)

        if isinstance(test, AverageTest):
            self.test_viz = AverageTestViz(
                self.ax, self.simulation, n_rows=self.n_rows, test=test
            )
            self.ax.set_xlim(0, self.simulation.stds[0] * 5 * 2)
            self.ax.set_ylim(-0.5, 20.5)
        elif isinstance(test, PABTest):
            self.test_viz = PABTestViz(
                self.ax, self.simulation, n_rows=self.n_rows, test=test
            )
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-0.5, 20.5)
        else:
            self.test_viz = None

    def set_pab(self, pab):
        if self.test_viz is not None:
            self.simulation.set_pab(pab)
            self.test_viz.redraw()


class PABPanel:
    def __init__(self, fig, ax, simulation, n_rows=20):
        self.ax = ax
        self.dists = PABDists(
            # fig.add_axes([0.1, 0.35, 0.2, 0.12], zorder=zorder()), simulation, scale=0.7
            fig.add_axes([0, 0, 0, 0], zorder=zorder()),
            simulation,
            scale=0.7,
        )
        self.scatter = PABScatter(
            fig.add_axes([0.1, 0.0, 0.2, 0.45], zorder=zorder.get() - 1),
            simulation,
            n_rows,
        )
        self.comparison = PABComparison(
            fig.add_axes([0.32, 0.0, 0.2, 0.45], zorder=zorder.get() - 2),
            simulation,
            n_rows,
        )

    def open(self, i, n_frames):
        figure_coords = self.get_figure_coords(self.pab)

        initial_x = figure_coords[0]
        initial_y = figure_coords[1]
        initial_width = 0
        initial_height = 0
        end_height = 0.12
        end_width = 0.2
        end_y = initial_y - 0.05 - end_height

        # Pull line down
        if i < n_frames / 2:
            x = initial_x
            width = 0
            y = linear(initial_y, end_y, i, n_frames / 2)
            height = linear(initial_height, end_height, i, n_frames / 2)
        # Open
        else:
            y = end_y
            height = end_height
            # x = linear(initial_x, end_x, i, n_frames / 2)
            width = linear(initial_width, end_width, i - n_frames / 2, n_frames / 2)
            x = initial_x - width / 2

        self.dists.ax.set_position((x, y, width, height))

        bbox = self.scatter.ax.get_position()
        self.scatter.ax.set_position((x, bbox.y0, bbox.width, bbox.height))

    def set_simulation(self, simulation):
        self.scatter.simulation = simulation
        self.comparison.simulation = simulation

    def set_test(self, test):
        self.comparison.set_test(test)

    def set_sample_size(self, sample_size):
        self.scatter.scatter.sample_size = sample_size
        self.scatter.simulate(1, 1, "A", [0, -1])
        self.scatter.simulate(1, 1, "B", [0, -1])

    def get_figure_coords(self, pab):
        display_coords = self.ax.transData.transform((pab, 0))
        return self.ax.figure.transFigure.inverted().transform(display_coords)

    def set_pab(self, pab):
        self.pab = pab
        figure_coords = self.get_figure_coords(pab)

        center = figure_coords[0]
        top = figure_coords[1]
        bbox = self.dists.ax.get_position()
        x = center - bbox.width / 2
        # x = center
        y = top - 0.05 - bbox.height
        self.dists.ax.set_position((x, y, bbox.width, bbox.height))
        self.dists.set_pab(pab)
        bbox = self.scatter.ax.get_position()
        self.scatter.set_pab(pab)
        self.scatter.ax.set_position((x, bbox.y0, bbox.width, bbox.height))
        scatter_width = bbox.width
        bbox = self.comparison.ax.get_position()
        self.comparison.set_pab(pab)
        self.comparison.ax.set_position(
            (x + scatter_width + 0.02, bbox.y0, bbox.width, bbox.height)
        )


class Indicator:
    def __init__(self, ax):
        self.line = ax.plot(
            [0], [0], color="grey", linestyle="--", clip_on=False, zorder=zorder()
        )[0]
        self.text = ax.text(
            0,
            0,
            "",
            va="center",
            ha="left",
            clip_on=False,
            fontsize=14,
            zorder=zorder.get(),
        )
        self.x_max = 1.01
        self.show = False

    def redraw(self):
        if self.show:
            x = self.pab
            y_idx = numpy.searchsorted(self.curve.pabs, self.pab)
            y = self.curve.rates[y_idx]
            self.line.set_xdata([x, self.x_max])
            self.line.set_ydata([y, y])
            self.text.set_text("{y: 3d}%".format(y=int(y)))
            self.text.set_position((self.x_max, y))
        else:
            self.line.set_xdata([0])
            self.line.set_ydata([0])
            self.text.set_text("")
            self.text.set_position((0, 0))

    def set_pab(self, pab):
        self.pab = pab
        self.redraw()

    def set_curve(self, curve):
        self.show = True
        self.curve = curve


class SimulationAnimation:
    def __init__(self, plot, ax=None, indicator=True, panel=True, legend=True):
        self.n_frames = 0
        self.ax = ax
        self.plot = plot
        self.ax_width = 0.45
        self.current_pab = 0.7
        self.with_indicator = indicator
        self.with_panel = panel
        self.with_legend = legend
        self.initialized = False

    def set_test(self, test):
        if self.with_panel:
            self.viz.set_test(test)

    def set_pab(self, pab):
        self.current_pab = pab
        if self.with_indicator:
            self.indicator.set_pab(pab)
        if self.with_panel:
            self.viz.set_pab(pab)

    def get_gamma(self):
        return self.plot.gamma

    def set_gamma(self, gamma):
        self.plot.set_gamma(gamma)

    def set_indicator(self, curve):
        if self.with_indicator:
            self.indicator.set_curve(curve)

    def get_sample_size(self):
        return self.viz.scatter.scatter.sample_size

    def set_sample_size(self, sample_size):
        self.plot.set_sample_size(sample_size)
        self.plot.redraw()
        self.ideal_simulation.sample_size = sample_size
        self.biased_simulation.sample_size = sample_size
        self.viz.set_sample_size(sample_size)

    def initialize(self, fig, ax, last_animation):

        if self.initialized:
            return

        if self.ax is None:
            self.ax = fig.add_axes([0.1, 0.6, self.ax_width, 0.3])

        self.plot.build_curves(self.ax)
        self.plot.format_ax(self.ax)

        if self.with_legend:
            self.plot.add_legend(self.ax)

        max_sample_size = (
            self.plot.simulations["ideal"].get_task("bert-rte").mu_a.shape[0]
        )
        current_sample_size = self.plot.sample_size
        self.ideal_simulation = self.plot.simulation_builder.create_simulations(
            ["bert-rte"],
            "ideal",
            sample_size=max_sample_size,
            pab=self.current_pab,
            simuls=21,
        ).get_task("bert-rte")
        self.ideal_simulation.sample_size = current_sample_size
        self.biased_simulation = self.plot.simulation_builder.create_simulations(
            ["bert-rte"],
            "biased",
            sample_size=max_sample_size,
            pab=self.current_pab,
            simuls=21,
        ).get_task("bert-rte")
        self.biased_simulation.sample_size = current_sample_size

        # self.plot.add_h0(self.ax)
        # self.plot.add_h01(self.ax)
        # self.plot.add_h1(self.ax)

        # for curve in self.plot.curves.values():
        #     curve.set_pab(1)
        # self.plot.curves["biased-avg"].set_pab(1)
        # self.plot.curves["oracle"].set_pab(1)
        # self.plot.curves["ideal-pab"].set_pab(1)
        # self.plot.curves["biased-pab"].set_pab(1)

        if self.with_panel:
            self.viz = PABPanel(fig, self.ax, self.ideal_simulation)
            self.viz.set_pab(self.current_pab)
        # self.viz.set_test(self.plot.curves["biased-avg"].comparison_method)

        if self.with_indicator:
            self.indicator = Indicator(self.ax)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class ShowPAB:
    def __init__(self, n_frames, animation, pab):
        self.n_frames = n_frames
        self.animation = animation
        self.pab = pab
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.animation.ideal_simulation.set_pab(0.7)
        self.animation.biased_simulation.set_pab(0.7)
        self.models = []
        for i, label in enumerate("AB"):
            model = AddModel(
                self.n_frames,
                self.animation.viz.dists,
                label,
                mean=self.animation.biased_simulation.means[i],
                std=self.animation.biased_simulation.stds[i],
                min_x=-self.animation.biased_simulation.stds[i] * 5,
                max_x=self.animation.biased_simulation.stds[i] * 5,
                scale=self.animation.viz.dists.scale,
                color=variances_colors(i),
            )
            model.initialize(fig, ax, last_animation)
            self.models.append(model)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for model in self.models:
            model(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class OpenPAB:
    def __init__(self, n_frames, animation):
        self.n_frames = n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.pab = self.animation.current_pab

        self.show = ShowPAB(self.n_frames / 2, self.animation, self.pab)
        self.show.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.animation.viz.open(i, self.n_frames)
        if i > self.n_frames / 2:
            self.show(i - self.n_frames / 2, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class OpenScatter:
    def __init__(self, n_frames, animation):
        self.n_frames = n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.animation.viz.scatter.open(i, self.n_frames)

    def leave(self):
        self(self.n_frames, None, None, None)


class MovePAB:
    def __init__(self, n_frames, animation, pab):
        self.n_frames = n_frames
        self.animation = animation
        self.pab = pab
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_pab = self.animation.current_pab

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pab = linear(self.old_pab, self.pab, i, self.n_frames)
        self.animation.set_pab(pab)

    def leave(self):
        self(self.n_frames, None, None, None)


class SetTest:
    def __init__(self, n_frames, animation, plot, names):
        self.n_frames = n_frames
        self.animation = animation
        self.plot = plot
        self.names = names
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.animation.viz.set_test(self.plot.curves[self.names[-1]].comparison_method)
        # Reset indicator and comparison panel
        self.animation.set_pab(self.animation.current_pab)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class ShowCurve:
    def __init__(self, n_frames, animation, plot, names, pab):
        self.n_frames = n_frames
        self.animation = animation
        self.plot = plot
        self.names = names
        self.pab = pab
        self.move = MovePAB(n_frames, animation, pab)
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.move.initialize(fig, ax, last_animation)
        self.curves = [self.plot.curves[name] for name in self.names]
        self.animation.set_indicator(self.curves[-1])
        self.animation.set_test(self.curves[-1].comparison_method)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.move(i, fig, ax, last_animation)
        for curve in self.curves:
            curve.set_pab(self.animation.current_pab)

        if self.names[0] == "oracle" and self.animation.current_pab > 0.65:
            self.animation.plot.add_oracle_annotation(
                self.animation.ax, self.curves[0].pabs, self.curves[0].rates
            )

    def leave(self):
        self(self.n_frames, None, None, None)


class ShowH0:
    def __init__(self, n_frames, animation):
        self.n_frames = n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self.animation.plot.add_h0(self.animation.ax)


class ShowH01:
    def __init__(self, n_frames, animation):
        self.n_frames = n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self.animation.plot.add_h01(self.animation.ax)


class ShowH1:
    def __init__(self, n_frames, animation):
        self.n_frames = n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self.animation.plot.add_h1(self.animation.ax)


# TODO: Add Acc, AUC and other metrics in labels for variances


class AdjustAverage:
    def __init__(self, n_frames, animation):
        self.n_frames = n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_gamma = self.animation.plot.curves[
            "biased-avg"
        ].comparison_method.gamma

        def cost(gamma):
            self.animation.plot.curves["biased-avg"].comparison_method.gamma = gamma
            self.animation.plot.curves["biased-avg"].compute()
            # Compare err with biased-pab
            avg_rate = self.animation.plot.curves["biased-avg"].rates  # [index]
            pab_rate = self.animation.plot.curves["biased-pab"].rates  # [index]
            return ((numpy.array(avg_rate) - numpy.array(pab_rate)) ** 2).sum()

        result = scipy.optimize.minimize_scalar(cost, bounds=(0.5, 1), method="bounded")
        # print(result.x)
        self.new_gamma = result.x

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        gamma = translate(
            self.old_gamma, self.new_gamma, i, self.n_frames, saturation=5
        )
        self.animation.plot.curves["biased-avg"].comparison_method.gamma = gamma
        self.animation.plot.curves["biased-avg"].compute()
        self.animation.plot.curves["biased-avg"].redraw()

        # TODO: Set simulations with Average tests and adjust delta based on new gamma

    def leave(self):
        self(self.n_frames, None, None, None)


class SwitchSimulation:
    def __init__(self, n_frames, animation, names):
        self.n_frames = n_frames
        self.animation = animation
        self.names = names
        self.estimator_name = "IdealEst"
        self.initialized = False
        self.ideal_names = ["oracle", "single"]

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_estimator_name = self.estimator_name

        if self.names[-1] in self.ideal_names:
            new_simulation = self.animation.ideal_simulation
            self.estimator_name = "IdealEst"
        else:
            new_simulation = self.animation.biased_simulation
            self.estimator_name = "FixHOptEst"

        old_simulation = self.animation.viz.scatter.simulation

        # Already current simulation, do nothing
        if new_simulation is old_simulation:
            self.n_frames = 0
            self.initialized = True
            return

        self.old_simulation = copy.deepcopy(old_simulation)
        self.new_simulation = copy.deepcopy(new_simulation)
        self.new_simulation.set_pab(self.old_simulation.pab)
        # Sort so that transition occurs are nearby data points
        # NOTE: Bring back if using the switch simulation...
        # for simul in [self.old_simulation, self.new_simulation]:
        #     simul.mu_a = numpy.sort(simul.mu_a, axis=0)
        #     simul.mu_b = numpy.sort(simul.mu_b, axis=0)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        simulation = self.animation.viz.scatter.simulation
        simulation.mu_a = translate(
            self.old_simulation.mu_a,
            self.new_simulation.mu_a,
            i,
            self.n_frames,
            saturation=4,
        )
        simulation.mu_b = translate(
            self.old_simulation.mu_b,
            self.new_simulation.mu_b,
            i,
            self.n_frames,
            saturation=4,
        )
        self.animation.viz.scatter.redraw()

        p = translate(0, 1, i, self.n_frames)
        estimator_name = ""

        for i in range(max(len(self.estimator_name), len(self.old_estimator_name))):
            switch = numpy.random.random() < p
            if switch and i < len(self.estimator_name):
                estimator_name += self.estimator_name[::-1][i]
            elif i < len(self.old_estimator_name):
                estimator_name += self.old_estimator_name[::-1][i]

        label = self.animation.viz.scatter.y_label.format(
            estimator=estimator_name[::-1]
        )
        self.animation.viz.scatter.y_label_object.set_text(label)

    def leave(self):
        self(self.n_frames, None, None, None)


def create_curve_section(
    title, names, animation, plot, times, switch_simulation=[], opacity=0.8
):

    # single will drop simulation

    # avg will simulate back to 50, and turn to biased

    #         SwitchSimulation(
    #             FPS * 2,
    #             FPS * 2,
    #             animation,
    #             ["oracle"],
    #         ),

    # TODO: Insert MovePAB() back to 0.4 during SectionTitle, during the still.

    section = Section(
        [
            SectionTitle(FPS * times["title"], title, opacity=opacity, fade_ratio=0.25),
            MovePAB(FPS * times["move"], animation, pab=0.4),
        ]
        + switch_simulation
        + [
            SetTest(FPS * 2, animation, plot, names),
            Still(FPS * 2),
            ShowCurve(
                FPS * times["pab_05"][0],
                animation,
                plot,
                names,
                pab=0.5,
            ),
            Still(FPS * times["pab_05"][1]),
            ShowCurve(
                FPS * times["pab_075"][0],
                animation,
                plot,
                names,
                pab=PAB,
            ),
            Still(FPS * times["pab_075"][1]),
            ShowCurve(
                FPS * times["pab_1"][0],
                animation,
                plot,
                names,
                pab=1,
            ),
            Still(FPS * times["pab_1"][1]),
            DropTest(animation),
        ]
    )
    return section


class Simulate:
    def __init__(self, n_frames, animation, models, rows, sample_size=None):
        self.n_frames = n_frames
        self.animation = animation
        self.models = models
        self.sample_size = sample_size
        self.rows = rows
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.sample_size:
            self.animation.viz.scatter.scatter.sample_size = self.sample_size

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for model in self.models:
            self.animation.viz.scatter.simulate(i, self.n_frames, model, self.rows)

    def leave(self):
        self(self.n_frames, None, None, None)


class DropSampleSize:
    def __init__(self, n_frames, animation, models, rows, sample_size):
        self.n_frames = n_frames
        self.animation = animation
        self.models = models
        self.rows = rows
        self.new_sample_size = sample_size
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for model in self.models:
            self.animation.viz.scatter.scatter.decrease_sample_size(
                self.new_sample_size, i, self.n_frames, model, self.rows
            )

    def leave(self):
        self(self.n_frames, None, None, None)
        self.animation.viz.scatter.scatter.sample_size = self.new_sample_size


class DropTest:
    def __init__(self, animation):
        self.n_frames = 0
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)
        self.animation.indicator.show = False
        self.animation.viz.set_test(None)


class AdjustSampleSize:
    def __init__(self, n_frames, animation, sample_size):
        self.n_frames = n_frames
        self.animation = animation
        self.sample_size = sample_size
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_sample_size = self.animation.get_sample_size()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        sample_size = int(
            numpy.round(
                numpy.exp(
                    linear(
                        numpy.log(self.old_sample_size),
                        numpy.log(self.sample_size),
                        i,
                        self.n_frames,
                    )
                )
            )
        )
        self.animation.set_sample_size(sample_size)

    def leave(self):
        self(self.n_frames, None, None, None)


class AdjustGamma:
    def __init__(self, n_frames, animation, gamma):
        self.n_frames = n_frames
        self.animation = animation
        self.gamma = gamma
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_gamma = self.animation.get_gamma()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        gamma = linear(
            self.old_gamma,
            self.gamma,
            i,
            self.n_frames,
        )
        self.animation.set_gamma(gamma)

    def leave(self):
        self(self.n_frames, None, None, None)


class ChapterTitle:
    def __init__(self, n_frames, number, title, animation_builder=None):
        self.n_frames = n_frames
        self.number = number
        self.title = f"{number}.\n{title}"
        self.animation_builder = animation_builder
        self.fontsize = 38
        self.x_padding = 0.1
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.fig = fig

        if self.animation_builder:
            x, y = 0.5, 0
        else:
            x, y = -1, -1

        self.ax = fig.add_axes([x, y, 0.5, 1])

        if self.animation_builder:
            self.animation = self.animation_builder(self.n_frames, self.ax)
            self.animation.initialize(fig, ax, last_animation)

        self.text_object = self.ax.text(
            self.x_padding,
            0.6,
            self.title,
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        self.fade_out = FadeOut(FADE_OUT, self.ax)
        self.fade_out.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        show_text(
            self.text_object,
            self.title,
            i,
            self.n_frames / 10,
            min_i=int(numpy.log10(self.number)) + 1,
        )
        if self.animation_builder:
            self.animation(i, fig, ax, last_animation)

        if i > self.n_frames - self.fade_out.n_frames:
            self.fade_out(
                i - (self.n_frames - self.fade_out.n_frames), fig, ax, last_animation
            )

    def leave(self):
        self.fig.clear()


def reverse(animation):
    if isinstance(animation, Section):
        return Section([Reverse(anim) for anim in animation.plots[::-1]])
    elif isinstance(animation, Parallel):
        return Parallel([Reverse(anim) for anim in animation.animations[::-1]])

    return Reverse(animation)


class Cascade:
    def __init__(self, n_frames, animations):
        self.n_frames_cascade = n_frames
        self.animations = animations
        self.n_frames_per_animation = int(self.n_frames_cascade / len(animations))
        self.initialized = False

    @property
    def n_frames(self):
        return self.n_frames_cascade + self.animations[-1].n_frames

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for anim in self.animations:
            anim.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n = int(linear(0, len(self.animations), i, self.n_frames_cascade))
        for j, anim in enumerate(self.animations[:n]):
            assert i >= j * self.n_frames_per_animation
            anim(i - j * self.n_frames_per_animation, fig, ax, last_animation)

    def leave(self):
        for anim in self.animations:
            anim.leave()


class Parallel:
    def __init__(self, animations):
        self.animations = animations

    @property
    def n_frames(self):
        return max(anim.n_frames for anim in animations)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for anim in self.animations:
            anim.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for anim in self.animations:
            anim(i, fig, ax, last_animation)

    def leave(self):
        for anim in self.animations:
            anim.leave()


class Reverse:
    def __init__(self, animation):
        self.n_frames = animation.n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.animation.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.animation(self.n_frames - i, fig, ax, last_animation)

    def leave(self):
        self.animation(0, None, None, None)


class SetHBarWidth:
    def __init__(self, n_frames, hbar, value):
        self.n_frames = n_frames
        self.hbar = hbar
        self.old_value = hbar.get_width()
        self.new_value = value
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        width = linear(self.old_value, self.new_value, i, self.n_frames)
        self.hbar.set_width(width)

    def leave(self):
        self(self.n_frames, None, None, None)


class WriteText:
    def __init__(self, text, text_object, min_i=0, fill=True):
        self.n_frames = int(FPS * len(text) / TEXT_SPEED)
        self.text = text
        self.text_object = text_object
        self.min_i = min_i
        self.fill = fill
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        show_text(
            self.text_object,
            self.text,
            i,
            self.n_frames,
            min_i=self.min_i,
            fill=self.fill,
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class SlideTitle:
    def __init__(
        self, n_frames, position, text, x_padding=0.02, y_padding=0.05, fontsize=34
    ):
        self.n_frames = n_frames
        self.number = position
        self.text = f"{position}. {text}"
        self.x_padding = x_padding
        self.y_padding = y_padding
        self.fontsize = fontsize
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes([-1, -1, 0.1, 0.1], zorder=zorder())

        self.text_object = self.ax.text(
            self.x_padding,
            1 - self.y_padding,
            "",
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        self.text = WriteText(
            self.text,
            self.text_object,
            min_i=int(numpy.log10(self.number)) + 1,
            fill=False,
        )
        self.text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class SectionTitle:
    def __init__(self, n_frames, title, fade_ratio=0.5, opacity=0.8):
        self.n_frames = n_frames
        self.title = title
        self.x_padding = 0.1
        self.fontsize = 34
        self.fade_ratio = fade_ratio
        self.opacity = opacity
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes([-1, -1, 0.1, 0.1], zorder=zorder())

        # self.text_object = self.ax.text(
        #     self.x_padding,
        #     0.6,
        #     "",
        #     va="top",
        #     ha="left",
        #     zorder=zorder(2),
        #     transform=fig.transFigure,
        #     fontsize=self.fontsize,
        # )

        # self.text = WriteText(self.title, self.text_object)

        # self.fade_out = FadeOut(
        #     self.text.n_frames * 2,
        #     self.ax,
        #     opacity=self.opacity,
        #     pause=False,
        #     zorder_pad=-1,
        # )
        # self.fade_out.initialize(fig, ax, last_animation)

        # fade_in = Parallel(
        #     [
        #         self.fade_out,
        #         self.text,
        #     ]  # Section([Still(self.text.n_frames), self.text])]
        # )

        self.fade_out = FadeOut(
            int(FADE_OUT / 2 * self.fade_ratio),
            self.ax,
            opacity=self.opacity,
            pause=False,
        )
        self.fade_out.initialize(fig, ax, last_animation)

        self.text_object = self.ax.text(
            self.x_padding,
            0.6,
            "",
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        self.text = WriteText(self.title, self.text_object)

        fade_in = Parallel([self.fade_out, self.text])

        fade_out = reverse(fade_in)

        n_frames_still = max(self.n_frames - (fade_in.n_frames + fade_out.n_frames), 0)
        self.section = Section([fade_in, Still(n_frames_still), fade_out])
        self.section.initialize(fig, ax, last_animation)
        assert (
            self.section.n_frames == self.n_frames
        ), f"{self.section.n_frames} != {self.n_frames}"

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.section(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class BulletPoint:
    def __init__(self, n_frames, text, animation_builder, position, total, fontsize=24):
        self.n_frames = n_frames
        self.text = f"{position}.\n{text}"
        self.fontsize = 32
        self.x_padding = 0.025
        self.animation_builder = animation_builder
        self.position = position
        self.total = total
        self.width = 1 / total
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes(
            [(self.position - 1) * self.width, 0, self.width, 1], zorder=zorder()
        )
        self.animation = self.animation_builder(self.n_frames, self.ax)
        self.animation.initialize(fig, ax, last_animation)
        self.text_object = self.ax.text(
            self.x_padding + (self.position - 1) * self.width,
            0.85,
            f"{self.position}. " + self.text,
            va="top",
            ha="left",
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        show_text(self.text_object, self.text, i, self.n_frames / 10, min_i=3)
        self.animation(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


def build_noisy_moustacho(
    n_frames,
    ratio,
    ax,
    x,
    y,
    whisker_length,
    whisker_width,
    center_width=0,
    center_noise={"stability": 0.05, "standard_deviation": 0.02},
    length_noise={"stability": 0.05, "standard_deviation": 0.02},
):
    n_frames_in = int(n_frames * ratio)
    n_frames_noise = n_frames - n_frames_in
    moustacho = Moustacho(ax, x, y, center_width, whisker_length, whisker_width)
    sections = [
        GrowMoustacho(n_frames_in, moustacho),
        NoisyMoustacho(n_frames_noise, moustacho, center_noise, length_noise),
    ]

    return Section(sections)


class Moustacho:
    def __init__(self, ax, x, y, center_width, whisker_length, whisker_width):
        self.ax = ax
        self.x = x
        self.y = y
        self.center_width = center_width
        self.whisker_length = whisker_length
        self.whisker_width = whisker_width

        for key in ["x", "y", "center_width", "whisker_length", "whisker_width"]:
            setattr(self, f"current_{key}", getattr(self, key))

    def _update_kwargs(self, kwargs):
        for key in ["x", "y", "center_width", "whisker_length", "whisker_width"]:
            kwargs.setdefault(key, getattr(self, f"current_{key}"))
            setattr(self, f"current_{key}", kwargs[key])

    def draw(self, **kwargs):
        self._update_kwargs(kwargs)
        self.moustacho = moustachos(self.ax, **kwargs)

    def update(self, **kwargs):
        self._update_kwargs(kwargs)
        adjust_moustachos(self.moustacho, **kwargs)


class GrowMoustacho:
    def __init__(self, n_frames, moustacho):
        self.n_frames = n_frames
        self.moustacho = moustacho
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        # bbox = self.moustacho.ax.get_position()
        # width = bbox.width / 2
        # height = bbox.height / 3
        # x = (bbox.x0 + bbox.x1) / 2 - width * 2 / 3
        # y = (bbox.y0 + bbox.y1) / 2 - height / 2
        # self.moustacho.ax.set_position([x, y, width, height])
        # self.moustacho.ax.set_xlim(-1, 1)
        # self.moustacho.ax.set_ylim(-2, 2)
        # despine(self.moustacho.ax)

        self.moustacho.draw()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        kwargs = {}
        for key in ["whisker_length", "whisker_width", "center_width"]:
            end_value = getattr(self.moustacho, key)
            kwargs[key] = translate(
                end_value * 0.01,
                end_value,
                i,
                self.n_frames,
                saturation=5,
            )

        self.moustacho.update(**kwargs)

    def leave(self):
        self(self.n_frames, None, None, None)


class NoisyMoustacho:
    def __init__(self, n_frames, moustacho, center, length):
        self.n_frames = n_frames
        self.moustacho = moustacho
        self.y = center
        self.whisker_length = length
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        kwargs = {}
        for key in ["y", "whisker_length"]:
            current_value = getattr(self.moustacho, f"current_{key}")
            center_value = getattr(self.moustacho, key)
            process_value = current_value - center_value
            process_kwargs = getattr(self, key)
            process_value += ornstein_uhlenbeck_step(0, process_value, **process_kwargs)
            kwargs[key] = center_value + process_value

        self.moustacho.update(**kwargs)

    def leave(self):
        self(self.n_frames, None, None, None)


class MiniMoustacho:
    def __init__(
        self,
        n_frames,
        ax,
        center_noise,
        length_noise,
        x=0,
        y=0,
        whisker_length=1,
        whisker_width=0.2,
        center_width=0,
        resize=True,
    ):
        self.n_frames = n_frames
        self.initialized = False
        self.ax = ax

        self.x = x
        self.y = y
        self.whisker_length = whisker_length
        self.whisker_width = whisker_width
        self.center_width = center_width

        self.center_noise = center_noise
        self.length_noise = length_noise
        self.resize = resize

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.resize:
            bbox = self.ax.get_position()
            width = bbox.width / 2
            height = bbox.height / 3
            x = (bbox.x0 + bbox.x1) / 2 - width * 2 / 3
            y = (bbox.y0 + bbox.y1) / 2 - height / 2
            self.ax.set_position([x, y, width, height])
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-2, 2)
            despine(self.ax)

        self.moustacho_anim = build_noisy_moustacho(
            self.n_frames,
            ratio=1 / 10,
            ax=self.ax,
            x=self.x,
            y=self.y,
            whisker_length=self.whisker_length,
            whisker_width=self.whisker_width,
            center_width=self.center_width,
            center_noise=self.center_noise,
            length_noise=self.length_noise,
        )
        self.moustacho_anim.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.moustacho_anim(i, fig, ax, last_animation)

    def leave(self):
        pass


class MiniVariance:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.anim = MiniMoustacho(
            self.n_frames,
            self.ax,
            center_noise={"stability": 0.00, "standard_deviation": 0.0},
            length_noise={"stability": 0.05, "standard_deviation": 0.02},
        )
        self.anim.initialize(fig, ax, last_animation)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.anim(i, fig, ax, last_animation)

    def leave(self):
        self.anim.leave()


class MiniEstimator:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.anim = MiniMoustacho(
            self.n_frames,
            self.ax,
            center_width=0.2,
            whisker_width=0.1,
            center_noise={"stability": 0.01, "standard_deviation": 0.05},
            length_noise={"stability": 0.0, "standard_deviation": 0.0},
        )
        self.anim.initialize(fig, ax, last_animation)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.anim(i, fig, ax, last_animation)

    def leave(self):
        self.anim.leave()


class MiniComparison:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.anims = []
        for i in range(2):
            self.anims.append(
                MiniMoustacho(
                    self.n_frames,
                    self.ax,
                    x=-0.5 + i,
                    center_width=0.2,
                    whisker_width=0.1,
                    center_noise={"stability": 0.01, "standard_deviation": 0.05},
                    length_noise={"stability": 0.0, "standard_deviation": 0.0},
                    resize=i == 0,
                )
            )
            self.anims[i].initialize(fig, ax, last_animation)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for anim in self.anims:
            anim(i, fig, ax, last_animation)

    def leave(self):
        self.anim.leave()


class MiniPapersWithCode:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.fade_in = reverse(
            FadeOut(int(self.n_frames / 10), ax=self.ax, zorder_pad=10)
        )
        self.fade_in.initialize(fig, ax, last_animation)

        # NOTE: perhaps PapersWithCode should create a new ax instead of using the
        #       previous one.
        self.papers_with_code = build_papers_with_code(
            self.n_frames,
            title_fontsize=8,
            axis_label_fontsize=8,
            axis_tick_fontsize=6,
            ax=self.ax,
        )
        bbox = self.ax.get_position()
        width = bbox.width * 2 / 3
        x = (bbox.x0 + bbox.x1) / 2 - width / 2
        self.ax.set_position([x, 0.25, width, 0.2])

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.fade_in(i, fig, ax, last_animation)
        self.papers_with_code(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


# For 3.
#
class MiniVarianceBarPlot:
    def __init__(self, n_frames, ax, variances, sources):
        self.n_frames = n_frames
        self.ax = ax
        self.variances = variances
        self.sources = sources[::-1]
        self.task = "vgg"
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        bbox = self.ax.get_position()
        width = bbox.width / 3
        x = (bbox.x0 + bbox.x1) / 2 + width / 2
        self.ax.set_position([x, 0.15, width, 0.4])

        self.bars = self.ax.barh(
            range(len(self.sources)),
            [
                self.variances.get_data(self.task)[noise_type].std()
                for noise_type in self.sources
            ],
            height=0.6,
            align="edge",
            clip_on=False,
            color=[
                variances_colors(get_variance_color(label)) for label in self.sources
            ],
            zorder=zorder(),
        )

        self.ax.set_xlim((0, self.variances._get_max_std(self.task) * 1.05))
        self.ax.set_ylim((0, len(self.sources)))

        despine(self.ax)

        sections = []
        self.labels_objects = {}
        for i, label in enumerate(self.sources):
            self.labels_objects[label] = self.ax.text(
                -0.1 * self.variances._get_max_std(self.task),  # TODO
                i + 0.25,  # TODO
                "",
                # transform=self.ax.transFigure,
                va="center",
                ha="right",
                fontsize=18,
                clip_on=False,
            )

            write_text = WriteText(
                Variances.labels[label], self.labels_objects[label], fill=False
            )

            hbar = self.bars[i]
            value = hbar.get_width()
            hbar.set_width(0)
            var_increase = SetHBarWidth(write_text.n_frames, hbar, value)

            sections.append(Parallel([write_text, var_increase]))

        self.text_animations = Section(sections)
        self.text_animations.initialize(fig, ax, last_animation)

        self.ax.set_xlabel("STD", fontsize=18)
        self.ax.xaxis.set_label_coords(0.16, -0.035)
        # self.standard_deviation_label = self.ax.text(
        #     self.variances._get_max_std(self.task) / 2,  # TODO
        #     0,  # TODO
        #     "STD",
        #     transform=fig.transFigure,
        #     va="center",
        #     ha="center",
        #     fontsize=18,
        #     clip_on=False,
        # )

        # self.variances.get_data(self.task)[noise_type]

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text_animations(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class MiniSimpleLinearScale:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.linear_scale = SimpleLinearScale(
            self.n_frames,
            ax=self.ax,
            title_fontsize=8,
            axis_label_fontsize=12,
            axis_tick_fontsize=8,
            legend_fontsize=14,
        )
        self.linear_scale.initialize(fig, ax, last_animation)

        bbox = self.ax.get_position()
        width = bbox.width * 2 / 3
        x = (bbox.x0 + bbox.x1) / 2 - width / 2
        self.ax.set_position([x, 0.3, width, 0.2])

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.linear_scale(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class TextBox:
    def __init__(
        self,
        texts,
        ax,
        x,
        y,
        height,
        min_alpha=0.1,
        fontsize=16,
        minfontsize=8,
        colors=None,
        **kwargs,
    ):
        self.texts = texts
        self.current_text = texts[0]
        self.create_objects(
            ax, x, y, height, fontsize=fontsize, colors=colors, **kwargs
        )
        self.x = x
        self.y = y
        self.min_alpha = min_alpha
        self.minfontsize = minfontsize
        self.fontsize = fontsize
        self.max_delta = height * (len(texts) - 1)
        self.initiate_move(texts[0])
        self.set_text(texts[0])

    def create_objects(self, ax, x, y, height, colors, **kwargs):
        self.objects = []
        for i, text in enumerate(self.texts):
            self.objects.append(
                ax.text(
                    x,
                    y - i * height,
                    text,
                    color=colors[i] if colors else None,
                    **kwargs,
                )
            )

    def get_i(self, text):
        return self.texts.index(text)

    def get_snapshot(self):
        return [text_object.get_position() for text_object in self.objects]

    def initiate_move(self, text):
        self.positions = self.get_snapshot()
        self.moving_to_text = text

    def move_text(self, frac, text):
        assert text == self.moving_to_text, f"Movement to {text} not initiated"
        i = self.get_i(text)
        x, y = self.positions[i]
        diff = (self.y - y) * frac
        for position, text_object in zip(self.positions, self.objects):
            x, y = position
            new_y = y + diff
            text_object.set_position((x, new_y))

            delta = numpy.abs(self.y - new_y)

            delta_ratio = delta / self.max_delta

            opacity = delta_ratio * self.min_alpha + (1 - delta_ratio)
            text_object.set_alpha(opacity)

            fontsize = (
                delta_ratio * self.minfontsize + (1 - delta_ratio) * self.fontsize
            )

            text_object.set_fontsize(fontsize)

        # TODO: Set alpha based on how far the label is from focus

    def set_text(self, text):
        self.move_text(1, text)
        self.current_text = text


def build_estimator_change_chain(n_frames, ax, plot, estimators, repetitions=3):

    bbox = ax.get_position()
    width = bbox.width * 2 / 3
    x = (bbox.x0 + bbox.x1) / 2 - width / 2
    ax.set_position([x, 0.05, width, 0.4])
    despine(ax)

    ax.set_xlim((-5, 5))
    ax.set_ylim((-1, 22))

    if len(estimators) == 2:
        labels = dict(zip(estimators, estimators))
        text_box = TextBox(
            estimators,
            ax,
            x=-5,
            y=27,
            height=2.5,
            clip_on=False,  #  transform=ax.transAxes,
            ha="left",
            minfontsize=8,
            fontsize=16,
            colors=[EST_COLORS[estimator] for estimator in estimators],
        )
        objects = text_box.objects
        texts = estimators
    else:
        labels = dict(
            [
                (estimator, estimator.split(" ")[-1].replace(")", ""))
                for estimator in estimators
            ]
        )
        x = 3
        y = 25
        text_box = TextBox(
            [labels[e] for e in estimators],
            ax,
            x=x,
            y=y,
            height=2.5,
            clip_on=False,  #  transform=ax.transAxes,
            ha="left",
            minfontsize=8,
            fontsize=16,
            colors=[EST_COLORS[estimator] for estimator in estimators],
        )

        label = "FixHOptEST($k$,         )"
        label_object = ax.text(
            x - 5.65, y, label, ha="left", fontsize=16, clip_on=False
        )

        objects = text_box.objects + [label_object]
        texts = text_box.texts + [label]

    for text_object in objects:
        text_object.set_text("")

    n_frames_per_change = n_frames / (len(estimators) * repetitions)
    scatter = EstimatorBubbles(ax, n_rows=20, std=1, rho=0)

    fade_in = Parallel(
        [EstimatorFadeIn(int(n_frames_per_change / 2), scatter, plot, estimators[0])]
        + [WriteText(text, text_object) for (text_object, text) in zip(objects, texts)]
    )

    sections = [
        fade_in,
        Still(int(n_frames_per_change / 2)),
    ]
    for repetition in range(repetitions):
        # Skip first estimator on first round
        for estimator in estimators[1 * int(repetition == 0) :]:
            n_frames_move = int(n_frames_per_change / 2)
            estimator_swap = EstimatorChange(n_frames_move, scatter, plot, estimator)
            text_roll = MovingTextBox(n_frames_move, text_box, labels[estimator])
            sections.extend(
                [
                    Parallel([estimator_swap, text_roll]),
                    Still(int(n_frames_per_change / 2)),
                ]
            )

    return Section(sections)


class EstimatorFadeIn:
    def __init__(self, n_frames, scatter, plot, estimator, min_k=3, max_k=100):
        self.n_frames = n_frames
        self.scatter = scatter
        self.plot = plot
        self.estimator = estimator
        self.task = "bert-rte"
        self.min_k = min_k
        self.max_k = max_k
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        while self.scatter.get_k() < self.min_k:
            self.scatter.increase_k()

        self.old_k = self.scatter.get_k()

        if self.estimator == "IdealEst($k$)":
            rho = 0
        else:
            rho = self.plot.get_stat(
                self.task,
                self.estimator,
                self.max_k,
                stat="rho_var",
            )
            self.scatter.adjust_rho(rho)

        self.scatter.scatter.set_color(EST_COLORS[self.estimator])

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        k = linear(self.old_k, self.max_k, i, self.n_frames)
        while self.scatter.get_k() < k:
            self.scatter.increase_k()

        #     EST_COLORS[self.old_estimator],
        #     EST_COLORS[self.estimator],
        # ]
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        #     "Custom",
        #     colors,
        #     N=self.n_frames,
        # )

    def leave(self):
        self(self.n_frames, None, None, None)
        self.scatter.estimator = self.estimator


class MovingTextBox:
    def __init__(self, n_frames, text_box, target_text):
        self.n_frames = n_frames
        self.text_box = text_box
        self.target_text = target_text
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.text_box.initiate_move(self.target_text)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text_box.move_text(i / self.n_frames, self.target_text)

    def leave(self):
        self.text_box.set_text(self.target_text)


class EstimatorChange:
    def __init__(self, n_frames, scatter, plot, estimator, max_budgets=100):
        self.n_frames = n_frames
        self.scatter = scatter
        self.plot = plot
        self.estimator = estimator
        self.task = "bert-rte"
        self.max_budgets = max_budgets
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.old_rho = self.scatter.rho
        self.old_estimator = self.scatter.estimator

        if self.estimator == "IdealEst($k$)":
            self.new_rho = 0
        else:
            self.new_rho = self.plot.get_stat(
                self.task,
                self.estimator,
                self.max_budgets,
                stat="rho_var",
            )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        new_rho = linear(self.old_rho, self.new_rho, i, self.n_frames)
        self.scatter.adjust_rho(new_rho)
        # TODO: Adjust label text
        # self.scatter.rho_text.set_text(
        #     self.estimators.rho_text_template.format(rho=new_rho)
        # )

        colors = [
            EST_COLORS[self.old_estimator],
            EST_COLORS[self.estimator],
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Custom",
            colors,
            N=self.n_frames,
        )
        self.scatter.scatter.set_color(cmap(i))

    def leave(self):
        self(self.n_frames, None, None, None)
        self.scatter.estimator = self.estimator


class Parallel:
    def __init__(self, animations):
        self.animations = animations
        self.initialized = False

    @property
    def n_frames(self):
        return max(animation.n_frames for animation in self.animations)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for animation in self.animations:
            animation.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for animation in self.animations:
            if i <= animation.n_frames:
                animation(i, fig, ax, last_animation)

    def leave(self):
        for animation in self.animations:
            animation.leave()


class FadeOut:
    def __init__(
        self,
        n_frames,
        ax=None,
        opacity=1,
        pause=True,
        zorder_pad=0,
        transform=None,
        x=0,
        y=0,
        width=1,
        height=1,
    ):
        self.n_frames = n_frames
        self.ax = ax
        self.opacity = opacity
        self.pause = 2 if pause else 1
        self.zorder_pad = zorder_pad
        self.transform = transform
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.ax is None:
            self.ax = fig.add_axes([-1, -1, 1, 1], zorder=zorder() + self.zorder_pad)

        # TODO: Add white rectangle

        self.patch = patches.Rectangle(
            (self.x, self.y),
            self.width,
            self.height,
            fill=True,
            color="white",
            zorder=zorder() + self.zorder_pad,
            transform=fig.transFigure if self.transform is None else self.transform,
            clip_on=False,
            alpha=0.0,
        )
        self.ax.add_patch(self.patch)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        alpha = min(
            linear(0, self.opacity, i, int(self.n_frames / self.pause)), self.opacity
        )
        self.patch.set_alpha(alpha)

    def leave(self):
        self(self.n_frames, None, None, None)


class MiniSimulation:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        bbox = self.ax.get_position()
        width = bbox.width * 2 / 3
        x = (bbox.x0 + bbox.x1) / 2 - width / 2
        self.ax.set_position([x, 0.3, width, 0.2])

        fade_out = FadeOut(FADE_OUT / 2, ax=self.ax, zorder_pad=10)
        self.fade_in = reverse(fade_out)
        fade_out.initialize(fig, ax, last_animation)
        fade_out.patch.set_xy((bbox.x0, 0))
        fade_out.patch.set_width(0.5)
        fade_out.patch.set_height(0.6)
        # fade_out.patch.set_color("black")
        # fade_out.patch.set_alpha(0.5)

        self.plot = build_simulation_plot()
        self.plot.add_h0(self.ax, subtitle="")
        # self.plot.add_h01(self.ax)
        self.plot.add_h1(self.ax, subtitle="")
        self.animation = SimulationAnimation(
            self.plot, self.ax, indicator=False, panel=False, legend=False
        )
        self.animation.initialize(fig, ax, last_animation)
        self.curves = []
        for names in [
            ["ideal-avg", "biased-avg"],
            ["ideal-pab", "biased-pab"],
            ["single"],
        ]:
            self.curves.append(
                ShowCurve(
                    self.n_frames / 2,
                    self.animation,
                    self.plot,
                    names,
                    pab=1,
                )
            )
            self.curves[-1].initialize(fig, ax, last_animation)

        self.ax.set_ylabel("Rate of Detections", fontsize=16)
        self.ax.set_xlabel("$P(A > B)$", fontsize=16)
        self.ax.xaxis.set_label_coords(0.5, -0.25)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.fade_in(i, fig, ax, last_animation)
        for curve in self.curves:
            curve(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class MiniH1Change:
    def __init__(self, n_frames, ax):
        self.n_frames = n_frames
        self.ax = ax
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.minisimulation = MiniSimulation(self.n_frames / 5, self.ax)
        self.minisimulation.initialize(fig, ax, last_animation)

        self.animation = Section(
            [
                Still(self.minisimulation.curves[-1].n_frames),
                AdjustGamma(self.n_frames / 2, self.minisimulation.animation, 0.9),
                AdjustGamma(self.n_frames / 2, self.minisimulation.animation, 0.6),
            ]
        )
        self.animation.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.minisimulation(i, fig, ax, last_animation)
        self.animation(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class ClipOn:
    def __init__(self, variances):
        self.n_frames = 0
        self.variances = variances
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for scatter in self.variances.scatters.values():
            scatter.set_clip_on(True)

    def leave(self):
        self(self.n_frames, None, None, None)


class DrawBarAxis:
    def __init__(self, n_frames, variances, task, noise_type):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.noise_type = noise_type
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        bar_ax = self.variances.bar_axes[self.task]
        self.draw_axis = DrawAxis(
            self.n_frames,
            bar_ax,
            "Standard Deviation",
            self.variances.standard_deviation_label,
            xlim=(0, self.variances._get_max_std(self.task) * 1.05 * 100),
        )

        self.draw_axis.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.draw_axis(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class DrawHistAxis:
    def __init__(self, n_frames, variances, task, noise_type):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.noise_type = noise_type
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        data = self.variances.get_data(self.task)[self.noise_type]
        data = (1 - data) * 100

        scatter_ax = self.variances.hist_axes[self.task]
        self.draw_axis = DrawAxis(
            self.n_frames,
            scatter_ax,
            "Performances",
            self.variances.performances_label,
            xlim=(min(data), max(data)),
        )

        self.draw_axis.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.draw_axis(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class DropAxis:
    def __init__(self, axes):
        self.axes = axes
        self.reverse = reverse(Parallel(axes))
        self.n_frames = self.reverse.n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.reverse.initialize(fig, ax, last_animation)
        self.axes[0].draw_axis.write_label.min_i = 4
        assert self.axes[0].draw_axis.write_label.text == "Performances"

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.reverse(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)
        for ax in self.axes:
            ax.draw_axis.ax.remove()


class DrawAxis:
    def __init__(self, n_frames, ax, text, text_object, xlim, flush=False):
        self.n_frames = n_frames
        self.ax_ref = ax
        self.text = text
        self.text_object = text_object
        self.xlim = xlim
        self.flush = flush
        self.initialized = False

    def get_position(self, i, n_frames):
        bbox = self.ax_ref.get_position()
        x_center = (bbox.x1 + bbox.x0) / 2 + 0.005
        max_width = bbox.width * 0.9
        width = linear(0, max_width, i, n_frames)
        x0 = x_center - width / 2
        y0 = bbox.y0 + 0.05
        return [x0, y0, width, 0.001]

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        position = self.get_position(1, 1)
        self.ax = fig.add_axes(position, zorder=zorder.get())
        self.line = self.ax.plot([], [])[0]
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(3))

        for side in ["top", "right", "left"]:
            self.ax.spines[side].set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.ax.set_xlim(self.xlim)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(3))

        self.write_label = WriteText(self.text, self.text_object, fill=False)
        self.write_label.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        position = self.get_position(i, self.write_label.n_frames)
        self.ax.set_position(position)
        self.write_label(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class DrawHistAxis2:
    def __init__(self, n_frames, variances, task, noise_type):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.noise_type = noise_type
        self.initialized = False

    def get_position(self, i, n_frames):
        scatter_ax = self.variances.hist_axes[self.task]
        bbox = scatter_ax.get_position()
        x_center = (bbox.x1 + bbox.x0) / 2 + 0.005
        max_width = bbox.width * 0.9
        width = linear(0, max_width, i, n_frames)
        x0 = x_center - width / 2
        y0 = bbox.y0 + 0.05
        return [x0, y0, width, 0.001]

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        position = self.get_position(1, 1)
        self.ax = fig.add_axes(position, zorder=zorder.get())
        self.line = self.ax.plot([], [])[0]
        data = self.variances.get_data(self.task)[self.noise_type]
        data = (1 - data) * 100
        self.ax.set_xlim(min(data), max(data))
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        # self.ax.xaxis.set_ticks([89.75, 90.25, 90.75])

        for side in ["top", "right", "left"]:
            self.ax.spines[side].set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.write_label = WriteText(
            "Performances", self.variances.performances_label, fill=False
        )
        self.write_label.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        position = self.get_position(i, self.write_label.n_frames)
        self.ax.set_position(position)
        self.write_label(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class AddTaskTitle:
    def __init__(self, n_frames, variances, task):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.write_text = WriteText(
            self.variances.titles[self.task],
            self.variances.task_label_objects[self.task],
        )
        self.write_text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.write_text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class SamplePoints:
    def __init__(self, comparison):
        self.n_frames = 0
        self.comparison = comparison
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        assert self.comparison.simulation.contest == []
        for model in self.comparison.models:
            point = numpy.random.normal(model.mean, model.std)
            self.comparison.simulation.contest.append(point)

        self.comparison.simulation.redraw()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class DropWorst:
    def __init__(self, comparison):
        self.n_frames = 0
        self.comparison = comparison
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.comparison.simulation.evaluate_contest()
        self.comparison.simulation.redraw()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class CountPoint:
    def __init__(self, comparison, n_frames_per_move):
        self.comparison = comparison
        self.n_frames_per_move = n_frames_per_move
        self.last_i = 0
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        # points = []
        # for model in self.comparison.models:
        #     offsets = self.comparison.simulation.scatters[model.name].get_offsets()
        #     points.append(offsets[-1][0])

        # model_index = numpy.argmin(points)
        # drop_model = self.comparison.models[model_index].name
        # offsets = self.comparison.simulation.scatters[drop_model].get_offsets()
        # self.comparison.simulation.scatters[drop_model].set_offsets(offsets[:-1])
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n_steps = i - self.last_i
        for j, model in enumerate(self.comparison.models):
            for point in self.comparison.simulation.points[model.name]:
                for _ in range(n_steps):
                    point.drop(self.n_frames_per_move)

        self.last_i = i

        self.comparison.simulation.redraw()

    def leave(self):
        self(self.n_frames, None, None, None)


class Point:
    def __init__(self, x, y, end_x=0, end_y=-0.5):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.end_x = end_x
        self.end_y = end_y

        self.xlim = (self.start_x, self.end_x)
        if self.start_x > self.end_x:
            self.xlim = self.xlim[::-1]

        self.ylim = (self.start_y, self.end_y)
        if self.start_y > self.end_y:
            self.ylim = self.ylim[::-1]

    def drop(self, n_frames_per_move):
        x_speed = (self.end_x - self.start_x) / n_frames_per_move
        self.x = numpy.clip(self.x + x_speed, a_min=self.xlim[0], a_max=self.xlim[1])

        y_speed = (self.end_y - self.start_y) / n_frames_per_move
        # self.y = max(self.y + y_speed, self.end_y)
        self.y = numpy.clip(self.y + y_speed, a_min=self.ylim[0], a_max=self.ylim[1])

    def inert(self):
        return self.x == self.end_x and self.y == self.end_y


class ToyPABSimulation:
    def __init__(
        self,
        comparison,
        end_y=-0.5,
        start_time=FPS * 5 / 10,
        n_slow=2,
        n_speedup=10,
        end_time=FPS * 1 / 10,
        custom_time=None,
    ):
        self.comparison = comparison
        self.comparison.simulation = self
        self.contest = []
        self.points = {}
        self.start_y = 0
        self.end_y = end_y
        self.num = 22
        self.n_frames_per_move = FPS * 0.5
        self.time = numpy.zeros(self.num)
        self.time[:n_slow] = start_time
        self.time[n_slow : n_slow + n_speedup] = numpy.exp(
            numpy.linspace(numpy.log(start_time), numpy.log(end_time), n_speedup)
        )
        self.time[n_slow + n_speedup :] = end_time
        if custom_time:
            self.time[: len(custom_time)] = custom_time
        self.time = self.time.astype(int)
        self.n_frames = sum(self.time) * 2 + int(numpy.ceil(self.n_frames_per_move))
        self.initialized = False

    def evaluate_contest(self):
        model_index = numpy.argmax(self.contest)
        model_name = self.comparison.models[model_index].name
        # Move to left
        if model_index == 0:
            end_x = -5.5 + len(self.points[model_name]) * 0.5
        else:
            end_x = 5.25 - len(self.points[model_name]) * 0.5
        self.points[model_name].append(
            Point(
                self.contest[model_index], self.start_y, end_x=end_x, end_y=self.end_y
            )
        )
        self.contest = []

    def redraw(self):
        for i, model in enumerate(self.comparison.models):
            points = [(p.x, p.y) for p in self.points[model.name]]
            if self.contest:
                points.append((self.contest[i], 0))
            if points:
                self.scatters[model.name].set_offsets(points)
            else:
                self.scatters[model.name].set_offsets([(-20, -20)])

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.scatters = {}
        for model in self.comparison.models:
            self.scatters[model.name] = self.comparison.ax.scatter(
                [], [], color=model.color, clip_on=False, marker="s", zorder=zorder(5)
            )
            self.points[model.name] = []

        animation = []

        # y_speed = (self.end_y - self.start_y) / (FPS * 0.5)

        # TODO: increase speed gradually
        # TODO: Move x

        self.count = CountPoint(self.comparison, self.n_frames_per_move)
        # for i in range(2):
        #     animation.extend(
        #         [
        #             SamplePoints(self.comparison),
        #             Still(FPS * 1),
        #             DropWorst(self.comparison),
        #             Still(FPS * 1),
        #         ]
        #     )
        for time in self.time:
            animation.extend(
                [
                    SamplePoints(self.comparison),
                    Still(time),  # int(FPS * (num - i) / num)),
                    DropWorst(self.comparison),
                    Still(time),  #  int(FPS * (num - i) / num)),
                ]
            )

        self.animation = Section(animation)
        self.old_pab = self.comparison.method_object.compute_pab()[1]
        self.new_pab = self.old_pab
        self.pab_point = Point(self.old_pab, 0, end_x=self.new_pab, end_y=0)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        # Add one block per model
        # SamplePoints()
        # wait x seconds
        # drop worse one block
        # DropWorst()
        # move to PAB axis
        # CountPoint()
        names = [model.name for model in self.comparison.models]
        points = [len(self.points[name]) for name in names]
        if sum(points) > 0:
            pab = points[0] / sum(points)
            if pab != self.new_pab:
                self.old_pab = self.comparison.method_object.pab[1]
                self.new_pab = pab
                self.pab_point = Point(self.old_pab, 0, end_x=pab, end_y=0)
            else:
                self.pab_point.drop(FPS * 0.25)
            pab = self.pab_point.x
            self.comparison.method_object.redraw((pab - 1e-10, pab, pab + 1e-10))
        self.animation(i, fig, ax, last_animation)
        self.count(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class AddPABWhiskers:
    def __init__(self, n_frames, comparison):
        self.n_frames = n_frames
        self.comparison = comparison
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        lower, pab, upper = self.comparison.method_object.pab
        alpha = 0.05
        sample_size = 22  # TODO: Should be total number of points in simulation (22)
        ci = scipy.stats.norm.isf(alpha / 2) * numpy.sqrt(pab * (1 - pab) / sample_size)
        lower = max(pab - ci, 0)
        upper = min(pab + ci, 1)
        self.saved_ci = (lower, pab, upper)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        lower, pab, upper = self.saved_ci

        whisker_width = self.comparison.method_object.whisker_width
        whisker_length = translate(
            1e-10,
            pab - lower,
            i,
            self.n_frames,
        )

        adjust_h_moustachos(
            self.comparison.pab_plot,
            x=pab,
            y=0,
            whisker_width=whisker_width,
            whisker_length=whisker_length,
            center_width=self.comparison.method_object.whisker_width * 1.5,
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class AddPABGamma:
    def __init__(self, n_frames, comparison):
        self.n_frames = n_frames
        self.comparison = comparison
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if i > self.n_frames * 9 / 10:
            self.comparison.method_object.gamma_label.set_text("$\gamma$")

        gamma_tick_y = translate(
            0,
            0.25,
            i,
            self.n_frames,
        )

        self.comparison.method_object.gamma_tick.set_ydata(
            [-1 + gamma_tick_y, -1 - gamma_tick_y]
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class RemoveBlocks:
    def __init__(self, n_frames, comparison):
        self.n_frames = n_frames
        self.comparison = comparison
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        # names = [model.name for model in self.comparison.models]
        # points = [self.comparison.simulation.points[name] for name in names]
        # for name, model_points in zip(names, points):
        #     steps = numpy.linspace(1, self.n_frames, len(model_points))
        #     alpha = linear(1, 0, numpy.ones(len(model_points)) * i, steps)
        #     # TODO: I'm here
        #     # print(alpha)
        #     # self.comparison.simulation.scatters[name].set_alpha(alpha)

        names = [model.name for model in self.comparison.models]
        points = [self.comparison.simulation.points[name] for name in names]
        points = sum(points, [])
        for point in points[:i]:
            point.y = -1000
        self.comparison.simulation.redraw()

    def leave(self):
        self(self.n_frames, None, None, None)


class SetEstimatorRho:
    def __init__(self, n_frames, estimators, estimator_task, estimator):
        self.n_frames = n_frames
        self.estimators = estimators
        self.estimator_task = estimator_task
        self.estimator = estimator
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        rho = self.estimator_task.estimators.get_stat(
            self.estimator_task.task,
            self.estimator,
            self.estimator_task.max_budgets,
            stat="rho_var",
        )

        self.adjust = EstimatorAdjustRho(self.n_frames, self.estimators, rho)
        self.adjust.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.adjust(i, fig, ax, last_animation)

        right_estimator = self.estimators.estimators["right"]
        if hasattr(self.estimators.estimators["right"], "estimator"):
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "Custom",
                [
                    EST_COLORS[right_estimator.estimator],
                    EST_COLORS[self.estimator],
                ],
                N=self.n_frames,
            )
            right_estimator.scatter.set_color(cmap(i))

    def leave(self):
        self(self.n_frames, None, None, None)


class Template:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class Cover:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.title = plt.text(
            0.5,
            0.7,
            ("Accounting for Variance\n" "in Machine Learning Benchmarks"),
            fontsize=30,
            horizontalalignment="center",
            transform=plt.gcf().transFigure,
            verticalalignment="center",
        )

        authors = """
        Xavier Bouthillier$^{1,2}$, Pierre Delaunay$^3$, Mirko Bronzi$^2$, Assya Trofimov$^{1,2,4}$,
        Brennan Nichyporuk$^{2,5,6}$, Justin Szeto$^{2,5,6}$, Naz Sepah$^{2,5,6}$,
        Edward Raff$^{7,8}$, Kanika Madan$^{1,2}$, Vikram Voleti$^{1,2}$,
        Samira Ebrahimi Kahou$^{2,6}$, Vincent Michalski$^{1,2}$, Dmitriy Serdyuk$^{1,2}$,
        Tal Arbel$^{2,5,6,10}$, Christopher Pal$^{2,9,10,11}$, Gaël Varoquaux$^{2,6,12}$, Pascal Vincent$^{1,2,10}$"""

        affiliations = """
        $^1$Université de Montréal, $^2$Mila, $^3$Independant, $^4$IRIC, $^5$Center for Intelligent Machines,
        $^6$McGill University, $^7$Booz Allen Hamilton, $^8$University of Maryland, Baltimore County,
        $^9$Polytechnique, $^{10}$CIFAR, $^{11}$Element AI, $^{12}$Inria
        """

        self.authors_text = plt.text(
            0.5,
            0.45,
            authors,
            fontsize=16,
            horizontalalignment="center",
            transform=plt.gcf().transFigure,
            verticalalignment="center",
        )

        self.affiliations_text = plt.text(
            0.5,
            0.2,
            affiliations,
            fontsize=12,
            horizontalalignment="center",
            transform=plt.gcf().transFigure,
            verticalalignment="center",
        )
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(ax)

    def leave(self):
        self.title.set_position((-1, -1))
        self.authors_text.set_position((-1, -1))
        self.affiliations_text.set_position((-1, -1))


def build_papers_with_code(
    n_frames, title_fontsize=38, axis_label_fontsize=32, axis_tick_fontsize=16, ax=None
):
    # sections = [PapersWithCode(FPS * 5)]
    # for i in range(1, 76):
    #     sections.append(NoisyPapersWithCode(max(int(FPS * (2 / i)), 1)))

    # for i in range(76, 1 - 1):
    #     sections.append(NoisyPapersWithCode(max(int(FPS * (2 / i)), 1)))

    total = 1414
    n_intro_frames = int(n_frames * FPS * 5 / total)
    n_moving_frames = int((n_frames - n_intro_frames) / 2)

    papers_with_code = PapersWithCode(
        n_intro_frames,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        axis_tick_fontsize=axis_tick_fontsize,
        ax=ax,
    )

    sections = [papers_with_code]
    noisy_frames_counter = 0
    n_noisy_frames_sequence = []
    i = 1
    while noisy_frames_counter < n_moving_frames:
        # for i in range(1, 76):
        n_noisy_frames = max(int(n_frames * (FPS * 2 / i / total)), 2)
        noisy_frames_counter += n_noisy_frames
        sections.append(NoisyPapersWithCode(n_noisy_frames, papers_with_code))
        n_noisy_frames_sequence.append(n_noisy_frames)
        i += 1

    # for i in range(76, 1 - 1):
    for n_noisy_frames in n_noisy_frames_sequence[::-1]:
        sections.append(NoisyPapersWithCode(n_noisy_frames, papers_with_code))

    return Section(sections)


def build_intro(data_folder):
    sections = [
        Cover(FPS * 30),
        Black(FPS * 1),
        # Intro
        build_papers_with_code(FPS * 24),
        Still(FPS * 2),
        VarianceLabel(FPS * 10),
        EstimatorLabel(FPS * 10),
        ComparisonLabel(FPS * 20),
        # Zoom(FPS * 2),
        # Still(FPS * 5),
    ]

    return Chapter("Intro", sections, pbar_position=0)


def build_variances_chapter_title():
    return ChapterTitle(FPS * 5, 1, "Variance in\nML Benchmarks", MiniVariance)


def build_variances(data_folder):

    # TODO: Zoom on variance estimator, (average estimator and comparison)
    # Black(FPS * 0.25),
    # 1. Variances section

    variances = Variances(data_folder)

    draw_hist_axis = DrawHistAxis(FPS * 5, variances, "vgg", "init_seed")
    draw_bar_axis = DrawBarAxis(FPS * 5, variances, "vgg", "init_seed")

    sections = [
        build_variances_chapter_title(),
        variances,
        # TODO: Make label appear 1 or 2 seconds before it starts raining.
        #       Especially for the first one, weight inits.
        # TODO: Add STD and Performances labels at the bottom.
        VarianceSourceLabel(FPS * 5, variances, "init_seed"),
        AddTaskTitle(FPS * 5, variances, "vgg"),
        draw_hist_axis,
        draw_bar_axis,
        SpeedUpVarianceSource(
            FPS * 20,
            FPS * 5,
            FPS * 5,
            variances,
            "vgg",
            "init_seed",
            with_label=False,
            start_spacing=700,
            end_spacing=10,
            start_delta=10,
            end_delta=20,
        ),
        # VarianceSource(
        #     FPS * 10,
        #     variances,
        #     "vgg",
        #     "init_seed",
        #     with_label=False,
        #     spacing=10,
        #     delta=20,
        # ),
        # TODO: Maybe highlight median seed that is fixed for other noise types
        VarianceSourceLabel(FPS * 5, variances, "sampler_seed"),
        # VarianceSource(FPS * 5, variances, "vgg", "sampler_seed", with_label=False),
        SpeedUpVarianceSource(
            FPS * 10,
            FPS * 3,
            FPS * 5,
            variances,
            "vgg",
            "sampler_seed",
            with_label=False,
            start_spacing=500,
            end_spacing=10,
            start_delta=10,
            end_delta=20,
            n_slow=3,
            n_speedup=20,
        ),
        VarianceSourceLabel(FPS * 5, variances, "transform_seed"),
        VarianceSource(FPS * 5, variances, "vgg", "transform_seed", with_label=False),
        VarianceSourceLabel(FPS * 10, variances, "bootstrapping_seed"),
        VarianceSource(
            FPS * 10, variances, "vgg", "bootstrapping_seed", with_label=False
        ),
        # VarianceSource(FPS * 10, variances, "vgg", "bootstrapping_seed"),
        # VarianceSource(FPS * 2, variances, "vgg", "global_seed"),
        VarianceSourceLabel(FPS * 2, variances, "global_seed"),
        VarianceSourceLabel(FPS * 2, variances, "reference"),
        VarianceSourceLabel(FPS * 5, variances, "random_search"),
        VarianceSource(
            FPS * 10, variances, "vgg", "random_search", spacing=20, delta=20
        ),
        VarianceSourceLabel(FPS * 5, variances, "noisy_grid_search"),
        VarianceSource(
            FPS * 20, variances, "vgg", "noisy_grid_search", spacing=50, delta=20
        ),
        VarianceSourceLabel(FPS * 5, variances, "bayesopt"),
        VarianceSource(FPS * 20, variances, "vgg", "bayesopt", spacing=20, delta=50),
        VarianceSourceLabel(FPS * 10, variances, "everything"),
        VarianceSource(FPS * 20, variances, "vgg", "everything"),
        # TODO:
        VarianceSum(FPS * 15, variances),
        VariancesHighlight(FPS * 10, variances, ["init_seed"], vbars=[]),
        VariancesHighlight(FPS * 10, variances, ["bootstrapping_seed"], vbars=[]),
        VariancesHighlight(
            FPS * 10, variances, ["init_seed", "random_search"], vbars=[]
        ),
        Still(FPS * 5),
        Parallel(
            [
                DropAxis([draw_hist_axis, draw_bar_axis]),
                VarianceTask(FPS * 5, variances, "segmentation"),
            ]
        ),
        VarianceTask(FPS * 5, variances, "bio-task2"),
        VarianceTask(FPS * 5, variances, "bert-sst2"),
        VarianceTask(FPS * 5, variances, "bert-rte"),
        # NormalHighlight(FPS * 5, variances),
        # ClipOn(variances),
        Still(FPS * 1),
        VariancesFlushHist(FPS * 1, variances),
        Still(FPS * 5),
        VariancesHighlight(FPS * 10, variances, ["init_seed"], vbars=[]),
        VariancesHighlight(FPS * 10, variances, ["bootstrapping_seed"], vbars=[]),
        VariancesHighlight(
            FPS * 10,
            variances,
            ["init_seed", "random_search"],
            vbars=[],
            ratio_out=2,  # Don't remove highligth, just fade out.
        ),
        FadeOut(FADE_OUT / 2),
        build_variance_recap(data_folder),
        FadeOut(FADE_OUT / 2),
    ]

    return Chapter("Variances", sections, pbar_position=1)


def build_estimators_chapter_title():
    return ChapterTitle(FPS * 5, 2, "Estimating\nMean Performance", MiniEstimator)


def build_estimators(data_folder):

    OPACITY = 0.95

    algorithms = Algorithms(FPS * 10)

    estimators = EstimatorSimulation()

    estimator_task = EstimatorTask(0)

    # TODO: Add paperswithcode transition, showing comparison moustachos
    # 3. Comparison section

    sections = [
        build_estimators_chapter_title(),
        # TODO: Add paperswithcode transition, showing estimation moustacho
        # 2. Estimator section
        # TODO: Come back to average estimator (average estimator and comparison)
        algorithms,
        # Move estimator to the left
        # Add Ideal Algo1 (sweep highligh line above each line, and add O(T) on left
        # when at it)
        CodeHighlight(FPS * 5, lines=[1, 3]),
        CodeHighlight(FPS * 2, lines=[5, 5]),
        CodeHighlight(FPS * 2, lines=[6, 7]),
        CodeHighlight(FPS * 2, lines=[8, 8]),
        CodeHighlight(FPS * 2, lines=[10, 10], comment="O(T)", comment_side="left"),
        CodeHighlight(FPS * 2, lines=[11, 11], comment="O(1)", comment_side="left"),
        CodeHighlight(FPS * 2, lines=[12, 12]),
        CodeHighlight(FPS * 2, lines=[14, 15], comment="O(k*T)", comment_side="left"),
        CodeHighlight(FPS * 2, lines=[1, 20]),
        BringInBiasedEstimator(FPS * 5, algorithms),
        CodeHighlight(FPS * 2, lines=[1, 3]),
        CodeHighlight(FPS * 2, lines=[5, 7]),
        CodeHighlight(FPS * 3, lines=[8, 8], comment="O(T)", comment_side="right"),
        CodeHighlight(FPS * 2, lines=[9, 9]),
        CodeHighlight(FPS * 2, lines=[10, 11]),
        CodeHighlight(FPS * 2, lines=[12, 12], comment="O(1)", comment_side="right"),
        CodeHighlight(FPS * 2, lines=[13, 13]),
        CodeHighlight(FPS * 2, lines=[15, 16], comment="O(k+T)", comment_side="right"),
        CodeHighlight(FPS * 2, lines=[1, 20]),
        # Add basic linear plot k* 100 vs k + 100
        # Flush algos
        # SimpleLinearScale(FPS * 5),
        SlidingSimpleLinearScale(FPS * 5),
        # TODO: Remove plot instead of moving on left
        FadeOut(FPS * 2, x=0, y=0, width=1, height=0.83),
        # MoveSimpleLinearScale(FPS * 2),
        VarianceEquations(),
        reverse(FadeOut(FPS * 2, x=0, y=0, width=1, height=0.83)),
        Still(FPS * 10),
        estimators,
        reverse(FadeOut(FPS * 2, x=0, y=0, width=1, height=0.6)),
        Still(FPS * 10),
        EstimatorIncreaseK(FPS * 10, estimators),
        Still(FPS * 5),
        EstimatorAdjustRho(FPS * 10, estimators, new_rho=2),
        Still(FPS * 5),
        FadeOut(FPS * 1, x=0, y=0.6, width=1, height=0.23),
        estimator_task,
        reverse(
            Parallel(
                [
                    FadeOut(FPS * 1, x=0, y=0.54, width=1, height=0.26),
                    FadeOut(FPS * 1, x=0.42, y=0.45, width=0.15, height=0.1),
                ]
            )
        ),
        Still(FPS * 5),
        SectionTitle(
            FPS * 5, "Randomizing everything", opacity=OPACITY, fade_ratio=0.25
        ),
        EstimatorShow(FPS * 1, estimators, estimator_task, "IdealEst($k$)"),
        Still(FPS * 5),
        SectionTitle(
            FPS * 5,
            "Randomizing weight initialization only",
            opacity=OPACITY,
            fade_ratio=0.25,
        ),
        SetEstimatorRho(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, Init)"),
        EstimatorShow(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, Init)"),
        Still(FPS * 5),
        SectionTitle(
            FPS * 5, "Randomizing data splits only", opacity=OPACITY, fade_ratio=0.25
        ),
        SetEstimatorRho(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, Data)"),
        EstimatorShow(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, Data)"),
        Still(FPS * 5),
        SectionTitle(
            FPS * 5,
            "Randomizing everything except HPO",
            opacity=OPACITY,
            fade_ratio=0.25,
        ),
        SetEstimatorRho(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, All)"),
        EstimatorShow(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, All)"),
        Still(FPS * 10),
        FadeOut(FADE_OUT / 2),
        build_estimators_recap(data_folder),
        FadeOut(FADE_OUT / 2),
    ]

    return Chapter("Estimators", sections, pbar_position=2)


def build_comparison_chapter_title():
    return ChapterTitle(FPS * 5, 3, "Comparing\nAlgorithms", MiniComparison)


def build_comparisons(data_folder):

    average_comparison = ComparisonMethod(FPS * 1, "Average", 0.1)
    pab_comparison = ComparisonMethod(FPS * 1, "Probability of outperforming", 0.5)

    sections = [
        build_comparison_chapter_title(),
        average_comparison,
        AddModel(FPS * 1, average_comparison, "A", mean=1, std=2, scale=0.85),
        AddModel(FPS * 1, average_comparison, "B", mean=-1, std=2, scale=0.85),
        ComputeAverages(FPS * 5, average_comparison),
        pab_comparison,
        AddModel(FPS * 1, pab_comparison, "A", mean=1, std=2, scale=0.85),
        AddModel(FPS * 1, pab_comparison, "B", mean=-1, std=2, scale=0.85),
        Still(FPS * 5),
        ComputePAB(FPS * 5, pab_comparison),
        ToyPABSimulation(pab_comparison),
        Still(FPS * 5),
        AddPABWhiskers(FPS * 5, pab_comparison),
        Still(FPS * 2),
        RemoveBlocks(FPS * 5, pab_comparison),
        Still(FPS * 1),
        AddPABGamma(FPS * 2, pab_comparison),
        Still(FPS * 5),
    ]

    def change_model(a, b, modif=(0, 0), scale=(1, 1)):
        return {
            "mean": (a.mean + modif[0], b.mean + modif[1]),
            "std": (a.std * scale[0], b.std * scale[(1)]),
        }

    modifs = [(0, 2), (0, -6), (-4, 0), (4, 4), (5, -5), (-5, 5)]

    for modif in modifs:

        sections.append(
            ChangeDists(
                FPS * 1,
                [average_comparison, pab_comparison],
                foo=functools.partial(change_model, modif=modif),
            )
        )

    sections.append(Still(FPS * 5))

    modifs = [(10, 2), (1 / 10, 1 / 2), (1 / 10, 1 / 10)]

    for modif in modifs:

        sections.append(
            ChangeDists(
                FPS * 1,
                [average_comparison, pab_comparison],
                foo=functools.partial(change_model, scale=modif),
            )
        )

    sections += [
        Still(FPS * 5),
        SectionTitle(FPS * 5, "Simulations", opacity=1, fade_ratio=0.5),
        FadeOut(0),
    ]

    return Chapter("Test methods", sections, pbar_position=3)


SAMPLE_SIZE = 50


def build_simulation_plot():

    MAX_SAMPLE_SIZE = 1000
    N_POINTS = 100
    SIMULS = 1000  # 10000

    simulation_plot = SimulationPlot(
        # gamma=PAB, sample_size=50, n_points=100, simuls=1000
        gamma=PAB,
        sample_size=MAX_SAMPLE_SIZE,
        n_points=N_POINTS,
        simuls=SIMULS,
        pab_kwargs={"ci_type": "normal"},
    )
    simulation_plot.build_simulations()
    simulation_plot.set_sample_size(SAMPLE_SIZE)

    return simulation_plot


def build_simulations(data_folder):

    OPACITY = 0.95

    simulation_plot = build_simulation_plot()

    simulation_animation = SimulationAnimation(simulation_plot)

    sections = [
        # FadeOut(0),
        simulation_animation,
        reverse(FadeOut(FADE_OUT)),
        Still(FPS * 5),
        OpenPAB(FPS * 2, simulation_animation),
        Still(FPS * 1),
        OpenScatter(FPS * 0.5, simulation_animation),
        Simulate(FPS * 2, simulation_animation, models=["B"], rows=[0]),
        Still(FPS * 1),
        Simulate(FPS * 1, simulation_animation, models=["A"], rows=[0]),
        Still(FPS * 1),
        Simulate(FPS * 1, simulation_animation, models=["B", "A"], rows=[1]),
    ]

    for i in range(1, 10):
        sections.extend(
            [
                Simulate(
                    int(FPS * 1 / i),
                    simulation_animation,
                    models=["B", "A"],
                    rows=[i * 2],
                ),
                Simulate(
                    int(FPS * 1 / i),
                    simulation_animation,
                    models=["B", "A"],
                    rows=[i * 2 + 1],
                ),
            ]
        )

    sections += [
        Still(FPS * 5),
        MovePAB(FPS * 2, simulation_animation, pab=0.4),
        MovePAB(FPS * 2, simulation_animation, pab=1),
        MovePAB(FPS * 2, simulation_animation, pab=0.4),
        ShowH0(FPS * 5, simulation_animation),
        MovePAB(FPS * 2, simulation_animation, pab=0.5),
        ShowH01(FPS * 5, simulation_animation),
        MovePAB(FPS * 2, simulation_animation, pab=PAB),
        ShowH1(FPS * 5, simulation_animation),
        MovePAB(FPS * 2, simulation_animation, pab=1),
        Still(FPS * 2),
        create_curve_section(
            "The cheapest optimal",
            ["oracle"],
            simulation_animation,
            simulation_plot,
            times={
                "title": 5,
                "move": 0,
                "pab_05": (1, 5),
                "pab_075": (1, 5),
                "pab_1": (1, 5),
            },
            opacity=OPACITY,
        ),
        create_curve_section(
            "Single point comparison",
            ["single"],
            simulation_animation,
            simulation_plot,
            times={
                "title": 5,
                "move": 0,
                "pab_05": (1, 5),
                "pab_075": (1, 5),
                "pab_1": (1, 5),
            },
            switch_simulation=[
                Still(FPS * 2),
                DropSampleSize(
                    FPS * 2,
                    simulation_animation,
                    models=["B", "A"],
                    rows=[0, -1],
                    sample_size=1,
                ),
                Still(FPS * 2),
            ],
            opacity=OPACITY,
        ),
        create_curve_section(
            "Average comparisons",
            ["ideal-avg", "biased-avg"],
            simulation_animation,
            simulation_plot,
            times={
                "title": 5,
                "move": 0,
                "pab_05": (1, 5),
                "pab_075": (1, 5),
                "pab_1": (1, 5),
            },
            switch_simulation=[
                Still(FPS * 2),
                SwitchSimulation(1, simulation_animation, ["biased-avg"]),
                Simulate(
                    FPS * 2,
                    simulation_animation,
                    models=["A", "B"],
                    rows=[0, -1],
                    sample_size=SAMPLE_SIZE,
                ),
            ],
            opacity=OPACITY,
        ),
        create_curve_section(
            "Probability of outperforming",
            ["ideal-pab", "biased-pab"],
            simulation_animation,
            simulation_plot,
            times={
                "title": 5,
                "move": 0,
                "pab_05": (1, 5),
                "pab_075": (1, 5),
                "pab_1": (1, 5),
            },
            opacity=OPACITY,
        ),
        MovePAB(FPS * 1, simulation_animation, pab=0.75),
        Still(FPS * 5),
        SectionTitle(
            FPS * 5, "Adjusting $H_1$ (with $\gamma$)", opacity=OPACITY, fade_ratio=0.25
        ),
        AdjustGamma(FPS * 5, simulation_animation, gamma=0.9),
        Still(FPS * 5),
        AdjustGamma(FPS * 1, simulation_animation, gamma=0.75),
        SectionTitle(
            FPS * 5, "Adjusting sample size", opacity=OPACITY, fade_ratio=0.25
        ),
        Still(FPS * 5),
        AdjustSampleSize(FPS * 5, simulation_animation, sample_size=1000),
        Still(FPS * 5),
        # SectionTitle(FPS * 5, "Adjusting both", opacity=OPACITY, fade_ratio=0.25),
        # AdjustGamma(FPS * 5, simulation_animation, gamma=0.9),
        # AdjustAverage(FPS * 5, simulation_animation),
        # Still(FPS * 5),
        FadeOut(FADE_OUT / 2),
        build_comparisons_recap(data_folder),
        FadeOut(FADE_OUT / 2),
    ]

    return Chapter("Simulations", sections, pbar_position=4)


# TODO: Create recaps, and insert them at end of each chapters
#       Add words to add emphasis on important points
#       Add intro to chapters
#       Add sample size scale axis at end of simulation chapter
#       Add last 'slide' on recommendations (add pointer to video code)
#
#       Give clear explanations of H0 and H1 and why we should define these
#
#       Should I add a reference to our survey somewhere?
#
# 5mins video
#       Recommendations with guidelines
def build_variance_recap(data_folder):
    # Bring back paperswithcode and tell that the variance used for the animation is the one
    # measured in our experiments. We can easily see that this variance is concernly large.
    # It affects substantially the rankings.
    #
    # 1 Variance is large in common ML tasks
    # 2 Variance due to HPO is important
    # 3 Bootstrap (data sampling) dominates all other sources
    #   of variance

    variances = Variances(data_folder, with_ax=False)
    variances.initialize(None, None, None)

    # TODO: Maybe add a title at top of slide?

    sections = [
        BulletPoint(
            FPS * 10,
            text="Variance is\nlarge enough\nto be concerning",
            animation_builder=MiniPapersWithCode,
            position=1,
            total=3,
        ),
        BulletPoint(
            FPS * 10,
            text="Variance due to\nrandom data\nsplits dominates",
            animation_builder=functools.partial(
                MiniVarianceBarPlot,
                variances=variances,
                sources=[
                    "bootstrapping_seed",
                    "sampler_seed",
                    "init_seed",
                    "random_search",
                ],
            ),
            position=2,
            total=3,
        ),
        BulletPoint(
            FPS * 10,
            text="HPO variance\nis important",
            animation_builder=functools.partial(
                MiniVarianceBarPlot,
                variances=variances,
                sources=["bayesopt", "noisy_grid_search", "random_search", "init_seed"],
            ),
            position=3,
            total=3,
        ),
    ]

    return Chapter("", sections)


def build_estimators_recap(data_folder):

    estimator_plot = EstimatorsPlot()
    estimator_plot.load()

    # 1. Ideal estimator is too costly in general
    # 2. Using the same hyperparameters for many samples reduces the quality of the average empirical risk estimation
    # 3. Randomizing many sources of variation attenuates the loss of quality
    sections = [
        BulletPoint(
            FPS * 10,
            text="Ideal estimator\nis expensive",
            animation_builder=MiniSimpleLinearScale,
            position=1,
            total=3,
        ),
        BulletPoint(
            FPS * 10,
            text="Ignoring HPO\nhurts",
            animation_builder=functools.partial(
                build_estimator_change_chain,
                plot=estimator_plot,
                estimators=["IdealEst($k$)", "FixHOptEst($k$, Init)"],
            ),
            position=2,
            total=3,
        ),
        BulletPoint(
            FPS * 10,
            text="Randomizing\nmany sources\nof variation helps",
            animation_builder=functools.partial(
                build_estimator_change_chain,
                plot=estimator_plot,
                estimators=[
                    "FixHOptEst($k$, {source})".format(source=source)
                    for source in ["Init", "Data", "All"]
                ],
            ),
            position=3,
            total=3,
        ),
    ]

    return Chapter("", sections)


def build_comparisons_recap(data_folder):
    # 1. PAB is easier to use to control well rates of false positives and false negatives
    # 2. Statistical tests based on biased estimators are a reasonably good and cheap

    sections = [
        # BulletPoint(
        #     FPS * 10,
        #     text="Single point\ncomparison is\nterrible",
        #     animation_builder=MiniPapersWithCode,
        #     position=1,
        #     total=2,
        # ),
        BulletPoint(
            FPS * 10,
            text="Biased estimator is a\nreasonably good and\ncheap approximation",
            animation_builder=MiniSimulation,
            position=1,
            total=2,
        ),
        BulletPoint(
            FPS * 10,
            text="False positives and\nnegatives are easily\ncontroled with $P(A>B)$",
            animation_builder=MiniH1Change,
            position=2,
            total=2,
        ),
    ]
    return Chapter("", sections)


def build_recap(data_folder):

    sections = [
        ChapterTitle(FPS * 5, 4, "Recap"),
        build_variances_chapter_title(),
        build_variance_recap(data_folder),
        build_estimators_chapter_title(),
        build_estimators_recap(data_folder),
        build_comparison_chapter_title(),
        build_comparisons_recap(data_folder),
    ]

    return Chapter("Recap", sections, pbar_position=5)


chapters = dict(
    intro=build_intro,
    variances=build_variances,
    estimators=build_estimators,
    comparisons=build_comparisons,
    simulations=build_simulations,
    recap=build_recap,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--fps", default=60, choices=[2, 5, 10, 30, 60], type=int)
    parser.add_argument("--chapters", nargs="*", choices=chapters.keys(), type=str)
    # parser.add_argument("--output", default="mlsys2021.mp4", type=str)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--data-folder", default="~/Dropbox/Olympus-Data", type=str)
    parser.add_argument("--parallel", action="store_true", default=False)

    options = parser.parse_args(argv)

    if options.chapters is None:
        options.chapters = list(chapters.keys())

    args = [(chapter, options) for chapter in options.chapters]

    if options.parallel:
        with Pool() as p:
            p.starmap(create_video, args)
    else:
        list(itertools.starmap(create_video, args))


def create_video(chapter, options):

    width = 1280 / options.dpi
    height = 720 / options.dpi

    animate = Animation(
        [chapters[chapter](data_folder=options.data_folder)],
        width,
        height,
        start=options.start * FPS,
        fps=options.fps,
    )

    total_time = int(animate.n_frames / FPS) + options.start

    end = total_time if options.end < 0 else options.end
    end = min(total_time, end)

    animate.end = end * FPS

    frames = (end - options.start) * options.fps

    anim = FuncAnimation(
        animate.fig,
        animate,
        frames=frames,
        interval=20,
        blit=True,
    )

    Writer = animation.writers["ffmpeg"]
    writer = Writer(
        fps=options.fps,
        metadata=dict(artist="Xavier Bouthillier"),
        # bitrate=bitrate
    )

    animate.fig.set_size_inches(width, height, True)
    anim.save(f"{chapter}.mp4", writer=writer, dpi=options.dpi)


if __name__ == "__main__":

    # import matplotlib.font_manager

    # print(
    #     "\n".join(
    #         sorted(
    #             matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    #         )
    #     )
    # )
    # import sys

    # sys.exit(0)

    plt.rcParams.update({"font.size": 8})
    plt.close("all")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    # plt.rc("font", family="Times New Roman")
    # rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
    #                                'Lucida Grande', 'Verdana']
    # plt.rc('text', usetex=True)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("axes", labelsize=32, titlesize=38)

    main()
    # cProfile.run("main()", sort="cumtime")
