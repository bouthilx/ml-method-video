import argparse
import functools

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


from moustachos import adjust_moustachos, moustachos, h_moustachos, adjust_h_moustachos
from rained_histogram import rained_histogram, rain_std
from utils import translate, linear, ZOrder, despine
from paperswithcode import PapersWithCodePlot, cum_argmax
from variances import VariancesPlot
from estimator_bubbles import EstimatorBubbles
from estimators import EstimatorsPlot, LABELS
from estimators import COLORS as EST_COLORS
from simulations import pab, percentile_bootstrap, normal_ci

END = 10
FPS = 60


zorder = ZOrder()


N_INTRO = 25


norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
variances_colors = matplotlib.cm.get_cmap("tab10")

# fig = plt.figure()  # figsize=(width, height))
# fig.tight_layout()
# ax = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1))
# plt.gca().set_position([0, 0, 1, 1])
# scatter_solid = ax.scatter([], [], alpha=1)


def ornstein_uhlenbeck_step(mean, past_position, stability, standard_deviation):
    return stability * (mean - past_position) + numpy.random.normal(
        0, standard_deviation
    )


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
        self.pbar = tqdm(total=self.n_frames)
        self.initialized = True
        total = 0
        for j, plot in enumerate(self.plots):
            if total + plot.n_frames > self.start:
                self.counter = int(self.start - total)
                self.j = j
                break
            else:
                plot.initialize(self.fig, self.ax, self.plots[j - 1] if j > 0 else None)
                plot.clear()
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
            return (self.scatter,)
        elif plot.n_frames <= self.counter:
            plot.clear()

            self.j += 1
            self.counter -= plot.n_frames
            plot = self.plots[self.j]
            plot.initialize(
                self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
            )
            while plot.n_frames <= self.counter:
                plot.clear()
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
        self.pbar.update(self.step)
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

    def clear(self):
        self.black_patch.set_width(0)
        self.black_patch.set_height(0)


class PapersWithCode:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.key = "sst2"
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        # ax.plot([0.6, 4], [0.1, 0.8])  #  = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1))
        paperswithcode = PapersWithCodePlot()
        paperswithcode.load()

        self.x = paperswithcode.data[self.key][:, 0]
        self.new_y = paperswithcode.data[self.key][:, 1]
        self.p = self.new_y
        self.scatter = ax.scatter(
            self.x, self.p, numpy.ones(len(self.p)) * 15, zorder=zorder.get()
        )
        cum_x, cum_y = cum_argmax(self.x, self.p)
        self.line = ax.plot(cum_x, cum_y)[0]

        ax.set_position([0.2, 0.2, 0.6, 0.6])
        if self.key == "sst2":
            ax.set_ylim(80, 100)
        elif self.key == "cifar10":
            ax.set_ylim(85, 100)
        ax.set_xlim(min(self.x), max(self.x))

        ax.set_xlabel("Year")
        ax.set_ylabel("Accuracy")
        ax.set_title("Sentiment Analysis on\nSST-2 Binary classification")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def clear(self):
        pass


class NoisyPapersWithCode:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

        self.sizes = {"rte": 277, "sst2": 872, "cifar10": 10000}

    def _generate_noise(self, y):
        dataset_size = self.sizes[self.key]
        std = scipy.stats.binom(n=dataset_size, p=y / 100).std() / dataset_size
        return numpy.random.normal(y / 100, std) * 100

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.key = last_animation.key
        self.x = last_animation.x
        self.p = last_animation.p
        self.old_y = last_animation.new_y
        self.new_y = self._generate_noise(self.p)
        self.old_cum_x, self.old_cum_y = cum_argmax(self.x, self.old_y)
        self.new_cum_x, self.new_cum_y = cum_argmax(self.x, self.new_y)

        self.scatter = last_animation.scatter
        self.line = ax.plot(self.old_cum_x, self.old_cum_y, color="#1f77ba")[0]
        # self.line = last_animation.line
        last_animation.line.set_alpha(0.2)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(fig, ax, last_animation)

        ratio = 0.5
        saturation = 7
        y = translate(
            self.old_y, self.new_y, i, self.n_frames * ratio, saturation=saturation
        )
        self.scatter.set_offsets(list(zip(self.x, y)))
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

    def clear(self):
        pass


class Still:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        pass

    def clear(self):
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
            2019, 99, "", va="top", ha="left", fontsize=32, zorder=zorder(2)
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

    def clear(self):
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
            2019, 92, "", va="top", ha="left", fontsize=32, zorder=zorder.get()
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

    def clear(self):
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
            2019, 84.5, "", va="top", ha="left", fontsize=32, zorder=zorder.get()
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

    def clear(self):
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

    def clear(self):
        pass


class Variances:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.n_frames = 1  # FPS * 1
        self.initialized = False
        self.keys = ["vgg", "segmentation", "bio-task2", "bert-sst2", "bert-rte"]
        self.titles = {
            "vgg": "CIFAR10\nVGG11",
            "segmentation": "PascalVOC\nResNet18",
            "bio-task2": "MHC\nMLP",
            "bert-sst2": "Glue-SST2\nBERT",
            "bert-rte": "Glue-RTE\nBERT",
        }
        self.label_order = [
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
        self.label_colors = {}
        i = 0
        for label in self.label_order:
            self.label_colors[label] = i
            if not label.startswith("empty_"):
                i += 1
        self.labels = {
            "empty_below_hpo": "",
            "empty_above_hpo": "",
            "bootstrapping_seed": "Data (bootstrap)",
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

        # TODO: adjust colors
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

        self.bar_axes = {}
        self.hist_axes = {}
        self.bars = {}
        self.scatters = {}
        for i, key in enumerate(self.keys):
            self.bar_axes[key] = fig.add_subplot(
                1,
                2 * len(self.keys),
                i * 2 + 1,
                facecolor=["red", "blue", "purple", "yellow", "orange"][i],
                frameon=False,
            )

            self.hist_axes[key] = fig.add_subplot(
                1,
                2 * len(self.keys),
                i * 2 + 2,
                facecolor=["red", "blue", "purple", "yellow", "orange"][i],
                frameon=False,
            )

            # TODO: Use as many bars as there is noise source (+ everything) and
            #       leave a bit more space around HPO instead of full empty bars.
            self.bars[key] = self.bar_axes[key].barh(
                range(len(self.label_order)),
                numpy.zeros(len(self.label_order)),
                align="edge",
                clip_on=False,
                color=[
                    variances_colors(self.label_colors[label])
                    for label in self.label_order
                ],
                zorder=zorder(),
            )

            self.bar_axes[key].set_xlim((0, self._get_max_std(key) * 1.05))
            self.bar_axes[key].set_ylim((0, 12))

            data = numpy.ones(self._get_n_points(key)) * -1000
            # self.scatters[key].set_offsets(list(zip(data, data)))

            arrays = self.get_data(key)
            colors = numpy.concatenate(
                [
                    numpy.ones(arrays[label].shape) * self.label_colors[label]
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
                self.titles[task],
                transform=fig.transFigure,
                va="center",
                ha="center",
                fontsize=24,
            )

        self.performances_label = self.bar_axes["vgg"].text(
            self.label_width + self.ax_max_width / 2,
            self.x_axis_label_y,
            "Performances",
            transform=fig.transFigure,
            va="center",
            ha="center",
            fontsize=24,
        )

        self.standard_deviation_label = self.bar_axes["vgg"].text(
            self.label_width + self.ax_max_width * 3 / 2 + self.ax_max_padding,
            self.x_axis_label_y,
            "Standard Deviation",
            transform=fig.transFigure,
            va="center",
            ha="center",
            fontsize=24,
        )

        # self.bar_axes[self.keys[-1]].set_position([0.2, 0.1, 0.6, 0.2])
        # self.hist_axes["vgg"].set_position([0.2, 0.2, 0.6, 0.6])
        # self.hist_axes[self.keys[-1]].set_position([0.2, 0.2, 0.6, 0.6])

        ax.remove()

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def clear(self):
        pass


class VarianceSource:
    def __init__(
        self, n_frames, variances, task, noise_type, delta=20, with_label=True
    ):
        self.n_frames = n_frames
        self.variances = variances
        self.task = task
        self.noise_type = noise_type
        self.with_label = with_label
        self.delta = delta

    @property
    def hist_y(self):
        return self.variances.get_weights_y(self.noise_type)

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        # TODO: Add label if with_label
        if self.with_label:
            text = self.variances.labels[self.noise_type]
            n = int(translate(3, len(text) + 1, i, self.n_frames / 15, saturation=5))
            self.variances.labels_objects[self.noise_type].set_text(
                text[:n]  #  + (" " * (len(text) - n))
            )

        if self.noise_type not in self.variances.get_data(self.task):
            return

        hit_the_ground = rained_histogram(
            self.variances.scatters[self.task],
            self.variances.get_data(self.task)[self.noise_type],
            y_min=self.hist_y,
            y_max=self.hist_y + self.variances.hist_height * 0.5,
            y_sky=self.variances.hist_heights,
            step=i,
            steps=self.n_frames,
            spacing=10,  # Probably need to adjust based on which y_index it is
            delta=self.delta,  # Probably need to adjust based on which y_index it is
            subset=self.variances.get_slice(
                self.task, self.noise_type
            ),  # Need to adjust based on which noise type it is
            n_columns=self.variances.n_columns,
            marker_size=50,
            y_padding=0.15,
        )
        rain_std(self.variances.get_bar(self.task, self.noise_type), hit_the_ground)

    def clear(self):
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
                variances_colors(self.variances.label_colors[label])
                for label in self.labels
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

    def clear(self):
        pass


class VariancesHighlight:
    def __init__(self, n_frames, variances, noise_types, vbars):
        self.n_frames = n_frames
        self.n_frames_close_in = int(n_frames * 0.05)
        self.n_frames_close_out = int(n_frames * 0.95)
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
        # TODO: Create new rectangles
        # For any label not in self.noise_types:
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

        if i < self.n_frames_close_in:
            new_width = translate(0, 1, i, self.n_frames_close_in, saturation=10)
        else:
            new_width = translate(
                1,
                0,
                i - self.n_frames_close_out,
                self.n_frames - self.n_frames_close_out,
                saturation=10,
            )

        # TODO: Slides the rectangles very fast ->
        for rectangle in self.grey_patches.values():
            rectangle.set_width(new_width)

    def clear(self):
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
        # TODO: Create new rectangles
        # For any label not in self.noise_types:
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

        # TODO: Slides the rectangles very fast ->
        for rectangle in self.grey_patches.values():
            rectangle.set_height(new_height)

    def clear(self):
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

    def initialize(self, fig, ax, last_animation):
        self.old_std_x = self.variances.standard_deviation_label.get_position()[0]
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        # TODO: Animate ax motion
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

    def clear(self):
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

    def clear(self):
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

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        # TODO: Animate ax motion
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

    def clear(self):
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
        self.ideal_ax.imshow(self.ideal_img, cmap="gray")

        self.biased_ax = fig.add_axes([1.2, 0.125, 0.4, 0.82])
        self.biased_img = mpimg.imread("algorithms_biased.png")
        self.biased_ax.imshow(self.biased_img, cmap="gray")

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

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def clear(self):
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

    def clear(self):
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

    def clear(self):
        self(self.n_frames, None, None, None)


class SimpleLinearScale:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False
        self.T = 100
        self.x_max = 200
        self.ax_height = 0.55
        self.white_box_height = 0.835

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        # TODO: Create new rectangles
        # For any label not in self.noise_types:
        self.white_box = patches.Rectangle(
            (0.15, 0),
            1,
            0,
            fill=True,
            color="white",
            alpha=1,
            zorder=zorder(),
            transform=fig.transFigure,
            linewidth=0,
        )
        fig.patches.append(self.white_box)

        self.simple_curve_ax = fig.add_axes([0.3, 0.2, 0.4, 0], zorder=zorder())
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
        self.legend = self.simple_curve_ax.legend(fontsize=18, loc="upper left")
        self.x_label = "Sample size"
        self.y_label = "Number of trainings"
        self.simple_curve_ax.set_xlabel(self.x_label)
        self.simple_curve_ax.set_ylabel(self.y_label)
        self.simple_curve_ax.spines["top"].set_visible(False)
        self.simple_curve_ax.spines["right"].set_visible(False)
        # self.variances.bar_axes["vgg"].add_patch(self.grey_patches[label])
        # self.black_patch.set_width(5)
        # self.black_patch.set_height(5)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.white_box.set_height(
            translate(0, self.white_box_height, i, self.n_frames / 10, saturation=10)
        )
        bbox = self.simple_curve_ax.get_position()
        self.simple_curve_ax.set_position(
            [
                bbox.x0,
                bbox.y0,
                bbox.width,
                translate(0, self.ax_height, i, self.n_frames / 5, saturation=10),
            ]
        )
        x = numpy.linspace(0, i / self.n_frames * self.x_max, 100)
        self.ideal_curve.set_xdata(x)
        self.ideal_curve.set_ydata(self.T * x)
        self.biased_curve.set_xdata(x)
        self.biased_curve.set_ydata(self.T + x)

    def clear(self):
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

        bbox = last_animation.simple_curve_ax.get_position()
        self.old_x = bbox.x0
        self.old_y = bbox.y0
        self.old_width = bbox.width
        self.old_height = bbox.height

        self.x_label = last_animation.x_label
        self.y_label = last_animation.y_label
        self.simple_curve_ax = last_animation.simple_curve_ax
        self.legend = last_animation.legend
        self.ideal_curve = last_animation.ideal_curve
        self.biased_curve = last_animation.biased_curve

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

    def clear(self):
        self(self.n_frames, None, None, None)


class VarianceEquations:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ideal_ax = fig.add_axes([0.26, 0.625, 0.2, 0.1], zorder=zorder())
        self.ideal_img = mpimg.imread("ideal_var.png")
        self.ideal_ax.imshow(self.ideal_img, cmap="gray")

        self.biased_ax = fig.add_axes([0.53, 0.53, 0.42, 0.3], zorder=zorder.get())
        self.biased_img = mpimg.imread("biased_var.png")
        self.biased_ax.imshow(self.biased_img, cmap="gray")

        for axis in [self.ideal_ax, self.biased_ax]:
            for side in ["top", "right", "bottom", "left"]:
                axis.spines[side].set_visible(False)
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def clear(self):
        pass


class EstimatorSimulation:
    def __init__(self):
        self.n_frames = 1
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

    def clear(self):
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

    def clear(self):
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

        self.old_rho = self.estimators.estimators["left"].rho

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):

        rho = linear(self.old_rho, self.new_rho, i, self.n_frames)

        self.estimators.estimators["right"].adjust_rho(rho)

        self.estimators.rho_text.set_text(
            self.estimators.rho_text_template.format(rho=rho)
        )

    def clear(self):
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
        self.ax = fig.add_axes([1, 0.575, 0.4, 0.2], zorder=zorder(2))
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
        x = translate(1, 0.3, i, self.n_frames / 10, saturation=10)
        bbox = self.white_patch.get_bbox()
        self.white_patch.set_xy((x * 0.95, bbox.y0))
        x = translate(1, 0.3, i, self.n_frames / 2, saturation=10)
        bbox = self.ax.get_position()
        self.ax.set_position((x, bbox.y0, bbox.width, bbox.height))

    def clear(self):
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
        if self.estimator != "IdealEst($k$)":
            rho_var = self.estimator_task.estimators.get_stat(
                self.estimator_task.task,
                self.estimator,
                self.estimator_task.max_budgets,
                stat="rho_var",
            )
            new_rho = linear(self.old_rho, rho_var, i, self.n_frames)
            self.estimators.estimators["right"].adjust_rho(new_rho)
            self.estimators.rho_text.set_text(
                self.estimators.rho_text_template.format(rho=new_rho)
            )

            if hasattr(self.estimators.estimators["right"], "estimator"):
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "Custom",
                    [
                        EST_COLORS[self.estimators.estimators["right"].estimator],
                        EST_COLORS[self.estimator],
                    ],
                    N=self.n_frames,
                )
                self.estimators.estimators["right"].scatter.set_color(cmap(i))

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

    def clear(self):
        self(self.n_frames, None, None, None)
        if self.estimator != "IdealEst($k$)":
            self.estimators.estimators["right"].estimator = self.estimator


class ComparisonMethod:
    def __init__(self, n_frames, method, x_padding):
        self.n_frames = n_frames
        self.method = method
        self.x_padding = x_padding
        self.initialized = False
        self.models = []
        self.width = 0.4
        self.xlim = (-10, 10)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.method == "Average":
            fig.clear()

        self.ax = fig.add_axes([self.x_padding, 0.5, self.width, 0.2], zorder=zorder())
        self.ax.text(
            self.x_padding + self.width / 2,
            0.8,
            self.method,
            ha="center",
            transform=fig.transFigure,
            fontsize=32,
        )

        despine(self.ax)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def clear(self):
        pass


class AddModel:
    def __init__(self, n_frames, comparison, name, mean=-1, std=1):
        self.n_frames = n_frames
        self.comparison = comparison
        self.name = name
        self.mean = mean
        self.std = std
        self.comparison.models.append(self)
        self.initialized = False

    def redraw(self):
        x = numpy.linspace(-10, 10, 1000)
        y = scipy.stats.norm.pdf(x, self.mean, self.std)
        y /= max(self.max_y, max(y))
        self.y = y
        self.line.set_ydata(y)
        self.name_label.set_position((self.mean, max(self.y)))

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        x = numpy.linspace(-10, 10, 1000)
        y = scipy.stats.norm.pdf(x, self.mean, self.std)
        self.max_y = max(y)
        y /= max(self.max_y, max(y))
        self.y = y
        self.line = self.comparison.ax.plot(x, y)[0]

        self.name_label = self.comparison.ax.text(
            self.mean, -1, self.name, ha="center", va="bottom", fontsize=16
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        scale = translate(0, 1, i, self.n_frames)
        self.line.set_ydata(self.y * scale)
        self.name_label.set_position((self.mean, max(self.y) * scale))

    def clear(self):
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

    def clear(self):
        self(self.n_frames, None, None, None)


class ComputePAB:
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

        self.pab_label.set_position((pab, 0.75))

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
            [pab, pab], [-0.75, -1.25], color="black", clip_on=False
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

        if i > self.n_frames / 10 * 5.5:
            self.gamma_label.set_text("$\gamma$")

        if i > self.n_frames / 10 * 5:
            gamma_tick_y = translate(
                0,
                0.25,
                i - self.n_frames / 10 * 5,
                self.n_frames / 10,
            )
        else:
            gamma_tick_y = 0

        self.gamma_tick.set_ydata([-1 + gamma_tick_y, -1 - gamma_tick_y])

        lower, pab, upper = self.saved_ci
        if i > self.n_frames / 10 * 8:
            whisker_width = self.whisker_width
            whisker_length = translate(
                1e-10,
                pab - lower,
                i - self.n_frames / 10 * 8,
                self.n_frames / 10,
            )
        else:
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

    def clear(self):
        pass


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
                model.redraw()

            comparison.method_object.redraw()

    def clear(self):
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

    def clear(self):
        pass


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
            (
                "Simulated Hyperparameter Optimization\n"
                "for Statistical Tests in Machine Learning Benchmarks"
            ),
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
        Tal Arbel$^{2,5,6,10}$, Christopher Pal$^{2,9,10,11}$, Gaël Varoquaux$^{2,6,12}$, Pascal Vincent$^{1,2,10,13}$"""

        affiliations = """
        $^1$Université de Montréal, $^2$Mila, $^3$Independant, $^4$IRIC, $^5$Center for Intelligent Machines,
        $^6$McGill University, $^7$Booz Allen Hamilton, $^8$University of Maryland, Baltimore County,
        $^9$Polytechnique, $^{10}$CIFAR, $^{11}$Element AI, $^{12}$Inria, $^{13}$FAIR
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

    def clear(self):
        self.title.set_position((-1, -1))
        self.authors_text.set_position((-1, -1))
        self.affiliations_text.set_position((-1, -1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--fps", default=60, choices=[2, 5, 10, 30, 60], type=int)
    parser.add_argument("--output", default="mlsys2021.mp4", type=str)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--data-folder", default="~/Dropbox/Olympus-Data", type=str)

    options = parser.parse_args(argv)

    width = 1280 / options.dpi
    height = 720 / options.dpi

    variances = Variances(options.data_folder)
    algorithms = Algorithms(FPS * 10)
    estimators = EstimatorSimulation()
    estimator_task = EstimatorTask(FPS * 2)
    average_comparison = ComparisonMethod(FPS * 1, "Average", 0.1)
    pab_comparison = ComparisonMethod(FPS * 1, "Probability of outperforming", 0.5)

    animate = Animation(
        [
            Cover(FPS * 30),
            Black(FPS * 1),
            # Intro
            PapersWithCode(FPS * 5),
        ]
        + [NoisyPapersWithCode(max(int(FPS * (2 / i)), 1)) for i in range(1, 76)]
        + [NoisyPapersWithCode(max(int(FPS * (2 / i)), 1)) for i in range(76, 1, -1)]
        + [
            Still(FPS * 2),
            VarianceLabel(FPS * 10),
            EstimatorLabel(FPS * 10),
            ComparisonLabel(FPS * 20),
            Zoom(FPS * 2),
            Still(FPS * 5),
            # TODO: Zoom on variance estimator, (average estimator and comparison)
            # Black(FPS * 0.25),
            # 1. Variances section
            variances,
            # TODO: Make label appear 1 or 2 seconds before it starts raining.
            #       Especially for the first one, weight inits.
            # TODO: Add STD and Performances labels at the bottom.
            VarianceSource(FPS * 10, variances, "vgg", "init_seed"),
            # TODO: Maybe highlight median seed that is fixed for other noise types
            VarianceSource(FPS * 5, variances, "vgg", "sampler_seed"),
            VarianceSource(FPS * 5, variances, "vgg", "transform_seed"),
            VarianceSource(FPS * 10, variances, "vgg", "bootstrapping_seed"),
            VarianceSource(FPS * 2, variances, "vgg", "global_seed"),
            VarianceSource(FPS * 2, variances, "vgg", "reference"),
            VarianceSource(FPS * 20, variances, "vgg", "random_search"),
            VarianceSource(FPS * 20, variances, "vgg", "noisy_grid_search"),
            VarianceSource(FPS * 20, variances, "vgg", "bayesopt"),
            VarianceSource(FPS * 20, variances, "vgg", "everything"),
            # TODO:
            VarianceSum(FPS * 15, variances),
            VariancesHighlight(FPS * 10, variances, ["init_seed"], vbars=[]),
            VariancesHighlight(FPS * 10, variances, ["bootstrapping_seed"], vbars=[]),
            VariancesHighlight(
                FPS * 10, variances, ["init_seed", "random_search"], vbars=[]
            ),
            Still(FPS * 5),
            VarianceTask(FPS * 5, variances, "segmentation"),
            VarianceTask(FPS * 5, variances, "bio-task2"),
            VarianceTask(FPS * 5, variances, "bert-sst2"),
            VarianceTask(FPS * 10, variances, "bert-rte"),
            # Still(FPS * 5),  # TODO: Replace with NormalHighlight before flushing them
            NormalHighlight(FPS * 5, variances),
            Still(FPS * 1),
            VariancesFlushHist(FPS * 1, variances),
            Still(FPS * 5),
            VariancesHighlight(FPS * 10, variances, ["init_seed"], vbars=[]),
            VariancesHighlight(FPS * 10, variances, ["bootstrapping_seed"], vbars=[]),
            VariancesHighlight(
                FPS * 10, variances, ["init_seed", "random_search"], vbars=[]
            ),
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
            CodeHighlight(
                FPS * 2, lines=[14, 15], comment="O(k*T)", comment_side="left"
            ),
            CodeHighlight(FPS * 2, lines=[1, 20]),
            # TODO: Add FixedHOptEst Algo1
            BringInBiasedEstimator(FPS * 5, algorithms),
            CodeHighlight(FPS * 2, lines=[1, 3]),
            CodeHighlight(FPS * 2, lines=[5, 7]),
            CodeHighlight(FPS * 3, lines=[8, 8], comment="O(T)", comment_side="right"),
            CodeHighlight(FPS * 2, lines=[9, 9]),
            CodeHighlight(FPS * 2, lines=[10, 11]),
            CodeHighlight(
                FPS * 2, lines=[12, 12], comment="O(1)", comment_side="right"
            ),
            CodeHighlight(FPS * 2, lines=[13, 13]),
            CodeHighlight(
                FPS * 2, lines=[15, 16], comment="O(k+T)", comment_side="right"
            ),
            CodeHighlight(FPS * 2, lines=[1, 20]),
            # Add basic linear plot k* 100 vs k + 100
            # Flush algos
            SimpleLinearScale(FPS * 5),
            MoveSimpleLinearScale(FPS * 2),
            VarianceEquations(FPS * 5),
            estimators,
            Still(FPS * 10),
            EstimatorIncreaseK(FPS * 10, estimators),
            Still(FPS * 5),
            EstimatorAdjustRho(FPS * 10, estimators, new_rho=2),
            Still(FPS * 5),
            estimator_task,
            EstimatorShow(FPS * 1, estimators, estimator_task, "IdealEst($k$)"),
            Still(FPS * 5),
            EstimatorShow(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, Init)"),
            Still(FPS * 5),
            EstimatorShow(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, Data)"),
            Still(FPS * 5),
            EstimatorShow(FPS * 1, estimators, estimator_task, "FixHOptEst($k$, All)"),
            Still(FPS * 10),
            # TODO: Add paperswithcode transition, showing comparison moustachos
            # 3. Comparison section
            average_comparison,
            AddModel(FPS * 1, average_comparison, "A", mean=1, std=2),
            AddModel(FPS * 1, average_comparison, "B", mean=-1, std=2),
            ComputeAverages(FPS * 5, average_comparison),
            pab_comparison,
            AddModel(FPS * 1, pab_comparison, "A", mean=1, std=2),
            AddModel(FPS * 1, pab_comparison, "B", mean=-1, std=2),
            Still(FPS * 5),
            ComputePAB(FPS * 5, pab_comparison),
        ]
        + [
            ChangeDists(FPS * 1, [average_comparison, pab_comparison], foo=foo)
            for foo in [
                lambda a, b: {"mean": (a.mean, b.mean + 2), "std": (a.std, b.std)},
                lambda a, b: {"mean": (a.mean, b.mean - 6), "std": (a.std, b.std)},
                lambda a, b: {"mean": (a.mean - 4, b.mean), "std": (a.std, b.std)},
                lambda a, b: {"mean": (a.mean + 4, b.mean + 4), "std": (a.std, b.std)},
                lambda a, b: {"mean": (a.mean + 5, b.mean - 5), "std": (a.std, b.std)},
                lambda a, b: {"mean": (a.mean - 5, b.mean + 5), "std": (a.std, b.std)},
            ]
        ]
        + [Still(FPS * 5)]
        + [
            ChangeDists(FPS * 1, [average_comparison, pab_comparison], foo=foo)
            for foo in [
                lambda a, b: {"mean": (a.mean, b.mean), "std": (a.std * 10, b.std * 2)},
                lambda a, b: {"mean": (a.mean, b.mean), "std": (a.std / 10, b.std / 2)},
                lambda a, b: {
                    "mean": (a.mean, b.mean),
                    "std": (a.std / 10, b.std / 10),
                },
            ]
        ]
        + [
            Still(FPS * 5),
            # ChangeVariances(FPS * 10),
            # Simulations
            # Add estimator bubble bottom left
            # Add empty plot center
            # Swipe dist example on Y axis to explain, explain at the same time the grey regions
            # 4. Summary with stat test example
        ],
        width,
        height,
        start=options.start * FPS,
        fps=options.fps,
    )
    total_time = int(animate.n_frames / FPS) + options.start

    end = total_time if options.end < 0 else options.end
    end = min(total_time, end)

    animate.end = end * FPS

    # TODO: Make it start from options.start
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
    anim.save(options.output, writer=writer, dpi=options.dpi)


if __name__ == "__main__":

    plt.rcParams.update({"font.size": 8})
    plt.close("all")
    # plt.rc('font', family='serif', serif='Times')
    plt.rc("font", family="Times New Roman")
    # plt.rc('text', usetex=True)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("axes", labelsize=32, titlesize=38)

    main()
