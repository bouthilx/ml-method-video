import argparse
import copy
import functools
from collections import defaultdict
from multiprocessing import Pool
import string
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

from moviepy.editor import VideoFileClip, concatenate_videoclips


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

from mlsys2021 import (
    Point,
    variances_colors,
    Animation,
    Cover,
    Chapter,
    ChapterTitle,
    Section,
    ComparisonMethod,
    AddModel,
    ChangeDists,
    FPS,
    Still,
    FadeOut,
    SectionTitle,
    WriteText,
    ToyPABSimulation,
    AddPABWhiskers,
    RemoveBlocks,
    ComputePAB,
)


zorder = ZOrder()
numbering = ZOrder()


class VarianceSource:
    def __init__(
        self, text, position, total=3, x_padding=0.05, x_margin=0.2, y_margin=0.2
    ):
        self.n_frames = 0
        self.text = text
        self.position = position
        self.total = total
        self.x_padding = x_padding
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.y_height = 0.5
        self.y_padding = 0.05
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        width = (1 - 2 * self.x_margin - (self.total - 1) * self.x_padding) / self.total
        x = self.x_margin + (self.position - 1) * (width + self.x_padding)
        self.ax = fig.add_axes([x, self.y_margin, width, self.y_height])
        despine(self.ax)
        self.label = self.ax.text(
            x + width / 2,
            self.y_height + self.y_margin + self.y_padding,
            "",
            transform=fig.transFigure,
            fontsize=28,
            ha="center",
            va="bottom",
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class WriteLabel:
    def __init__(self, variance_source):
        self.n_frames = WriteText(variance_source.text, None).n_frames
        self.variance_source = variance_source
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.write_text = WriteText(
            self.variance_source.text, self.variance_source.label
        )
        self.write_text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.write_text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class WriteOther:
    def __init__(self, variance_source, text, x, y):
        self.n_frames = WriteText(text, None).n_frames
        self.variance_source = variance_source
        self.text = text
        self.x = x
        self.y = y
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        text_object = self.variance_source.ax.text(
            self.x,
            self.y,
            "",
            fontsize=24,
            va="center",
            ha="left",
            transform=fig.transFigure,
            zorder=zorder(),
        )

        self.write_text = WriteText(self.text, text_object)
        self.write_text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.write_text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class RemoveOrderSplitRow:
    def __init__(self, n_frames, panel, row):
        self.n_frames = n_frames
        self.panel = panel
        self.row = row
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.text = self.panel.texts[self.row]
        self.label = self.panel.rows[self.row]

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        text = self.panel.texts[self.row]
        nth = int(linear(len(text), 0, i, self.n_frames))
        self.panel.rows[self.row].set_text(text[:nth])

    def leave(self):
        self(self.n_frames, None, None, None)


class AddOrderSplit:
    def __init__(
        self,
        n_frames,
        panel,
        blocks,
        row,
        new=False,
        choices=None,
        shuffle=True,
        seed=None,
        label=None,
    ):
        self.n_frames = n_frames
        self.panel = panel
        self.blocks = blocks
        self.row = row
        print(self.row)
        self.new = new
        self.n_rows = 6
        if choices is None:
            choices = list(string.ascii_lowercase)
        self.choices = choices
        self.shuffle = shuffle
        self.seed = seed
        self.label = label
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.seed:
            numpy.random.seed(self.seed)

        if self.new:
            text = self.choices[: sum(self.blocks)]
            if (self.row > 1 or self.seed) and self.shuffle:
                numpy.random.shuffle(text)
            i = self.blocks[0]
            tmp_text = copy.deepcopy(text[:i])
            for size in self.blocks[1:]:
                tmp_text += list(" | ") + copy.deepcopy(text[i : i + size])
                i += size
            text = "".join(tmp_text)
        else:
            text = self.choices[: self.blocks[0]]
            if (self.row > 1 or self.seed) and self.shuffle:
                numpy.random.shuffle(text)
            tmp_text = copy.deepcopy(text)
            for size in self.blocks[1:]:
                if self.shuffle:
                    numpy.random.shuffle(text)
                tmp_text += list(" | ") + copy.deepcopy(text)
            text = "".join(tmp_text)

        self.text = text.upper()

        if not hasattr(self.panel, "rows"):
            self.panel.rows = dict()
            self.panel.texts = dict()

        self.panel.texts[self.row] = self.text
        self.panel.rows[self.row] = self.panel.ax.text(
            0, self.row + 0.5, "", clip_on=False, fontsize=24, ha="left", va="center"
        )

        # self.panel.ax.set_xlim(-0.5, len(self.arch) - 0.5)
        self.panel.ax.set_ylim(self.n_rows, 1)

        if self.label:
            self.panel.ax.text(
                -0.25,
                self.row + 0.5,
                self.label,
                clip_on=False,
                ha="right",
                va="center",
                fontsize=32,
                fontweight="bold",
            )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        text = self.panel.texts[self.row]
        nth = int(linear(0, len(text), i, self.n_frames))
        print(text[:nth])
        self.panel.rows[self.row].set_text(text[:nth])

    def leave(self):
        self(self.n_frames, None, None, None)


class AddWeights:
    def __init__(self, n_frames, init, row, arch=[2, 4, 4, 2], seed=None, label=None):
        self.n_frames = n_frames
        self.init = init
        self.row = row
        self.n_rows = 6
        self.arch = arch
        self.seed = seed
        self.label = label
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.seed:
            numpy.random.seed(self.seed)

        self.nodes = []
        for i, n_neurons in enumerate(self.arch):
            width = 1 / n_neurons
            layer = numpy.arange(n_neurons).astype(float) * width + width / 2 + self.row
            self.nodes.append(list(zip([i] * n_neurons, layer)))

        self.vertices = []
        for i, layer_nodes in enumerate(self.nodes[:-1]):
            layer = []
            for j, neuron_in in enumerate(layer_nodes):
                for k, neuron_out in enumerate(self.nodes[i + 1]):
                    vertex = [neuron_in, neuron_out, numpy.random.random()]
                    layer.append(vertex)
            self.vertices.append(layer)

        for layer in self.nodes:
            numpy.random.shuffle(layer)

        for layer in self.vertices:
            numpy.random.shuffle(layer)

        if not hasattr(self.init, "scatter"):
            self.init.scatter = self.init.ax.scatter(
                [], [], clip_on=False, zorder=zorder(2)
            )
            self.init.ax.set_xlim(-0.5, len(self.arch) - 0.5)
            self.init.ax.set_ylim(self.n_rows, 1)
            self.init.lines = defaultdict(list)

        self.i = 0
        self.j = 0
        self.mode = "nodes"
        self.last_i = 0
        self.steps = 0
        self.total = 0
        for groups in [self.nodes, self.vertices]:
            self.total += sum(map(len, groups))

        if self.label:
            self.init.ax.text(
                -1,
                self.row + 0.5,
                self.label,
                clip_on=False,
                ha="right",
                va="center",
                fontsize=32,
                fontweight="bold",
            )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        steps = linear(0, self.total, i, self.n_frames)
        while self.steps <= steps:
            if self.j >= len(self.nodes):
                return

            if self.mode == "nodes":
                offsets = list(self.init.scatter.get_offsets())
                offsets.append(self.nodes[self.j][self.i])
                self.init.scatter.set_offsets(offsets)
                if self.i == len(self.nodes[self.j]) - 1 and self.j == 0:
                    self.j += 1
                    self.i = 0
                elif self.i == len(self.nodes[self.j]) - 1:
                    self.mode = "vertices"
                    self.i = 0
                else:
                    self.i += 1
            else:
                vertex = self.vertices[self.j - 1][self.i]
                x, y = zip(*vertex[:2])
                self.init.lines[self.row].append(
                    self.init.ax.plot(
                        x,
                        y,
                        alpha=vertex[-1],
                        color="black",
                        clip_on=False,
                        zorder=zorder.get() - 1,
                    )[0]
                )
                if self.i == len(self.vertices[self.j - 1]) - 1:
                    self.j += 1
                    self.i = 0
                    self.mode = "nodes"
                else:
                    self.i += 1

            self.steps += 1

    def leave(self):
        self(self.n_frames, None, None, None)


class RemoveWeights:
    def __init__(self, n_frames, init, row, arch):
        self.n_frames = n_frames
        self.init = init
        self.row = row
        self.arch = arch
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.n_vertices = len(self.init.lines[self.row])
        self.n_nodes = sum(self.arch)
        self.offsets = list(self.init.scatter.get_offsets())

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n_nodes = int(linear(self.n_nodes, 0, i, self.n_frames))
        n_vertices = int(linear(self.n_vertices, 0, i, self.n_frames))

        while len(self.init.lines[self.row]) > n_vertices:
            line = self.init.lines[self.row].pop(-1)
            line.remove()

        total = len(self.offsets)
        n = total - (self.n_nodes - n_nodes)
        self.offsets[:n]
        self.init.scatter.set_offsets(self.offsets[:n])

    def leave(self):
        self(self.n_frames, None, None, None)


class AddBlocks:
    def __init__(self, n_frames, pab, n, n_rows=12):
        self.n_frames = n_frames
        self.pab = pab
        self.n = n
        self.row = -1
        self.rows = []
        self.n_rows = n_rows
        self.x_margin = 0.1
        self.y_margin = 0.1
        self.width = 0.4
        self.height = 0.6
        self.p_padding = 0.05
        self.x_pab_head = int(self.n * 1.05)
        self.x_pab = int(self.n * 1.3)
        self.x_pab_sorted = int(self.n * 1.5)
        self.x_pab_bounds = int(self.n * 1.8)
        self.models = "AB"
        self.colors = dict(A=variances_colors(0), B=variances_colors(1))
        self.initialized = False

    def redraw(self):
        for row in self.points.keys():
            for model in self.models:
                offsets = [(point.x, point.y) for point in self.points[row][model]]
                if offsets:
                    self.scatters[row][model].set_offsets(offsets)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes([self.x_margin, self.y_margin, self.width, self.height])
        despine(self.ax)
        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(self.n_rows, 0)
        self.scatters = dict()
        self.points = dict()
        for row in range(-1, self.n_rows):
            self.scatters[row] = dict()
            self.points[row] = dict()
            for model in self.models:
                self.points[row][model] = []
                self.scatters[row][model] = self.ax.scatter(
                    [],
                    [],
                    s=100,
                    marker="s",
                    color=self.colors[model],
                    clip_on=False,
                )

        n_a = int(self.n * self.pab)

        self.data = [self.models[0]] * n_a + [self.models[1]] * (self.n - n_a)

        self.tmp_points = []
        for i, model in enumerate(self.data):
            point = Point(x=i, y=self.row, end_x=i, end_y=self.row)
            point.model = model
            self.tmp_points.append(point)

        self.data_pab = int(n_a / self.n * 100)
        text = f"$P(A>B)={self.data_pab:2d}$%"
        text_object = self.ax.text(
            self.x_pab_head,
            self.row,
            text,
            ha="left",
            va="center",
            fontsize=16,
            clip_on=False,
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n = int(linear(0, self.n, i, self.n_frames))
        points = defaultdict(list)
        for point in self.tmp_points[:n]:
            points[point.model].append(point)

        self.points[self.row] = points
        self.redraw()

        # offsets = defaultdict(list)
        # for j, point in enumerate(self.data[:n]):
        #     offsets[point].append(j)

        # for model in self.models:
        #     print(list(zip(offsets[model], [1] * len(offsets[model]))))
        #     if offsets[model]:
        #         self.scatters[model].set_offsets(
        #             list(zip(offsets[model], [1] * len(offsets[model])))
        #         )

    def leave(self):
        self(self.n_frames, None, None, None)


class SampleBlocks:
    def __init__(self, blocks, row, n_frames_per_sample, n_frames_per_move):
        if not isinstance(n_frames_per_sample, tuple):
            n_frames_per_sample = (n_frames_per_sample, n_frames_per_sample)

        if not isinstance(n_frames_per_move, tuple):
            n_frames_per_move = (n_frames_per_move, n_frames_per_move)

        self.n_frames = n_frames_per_sample[0] * blocks.n + n_frames_per_move[0]
        self.blocks = blocks
        self.row = row
        self.n_frames_per_sample = n_frames_per_sample
        self.n_frames_per_move = n_frames_per_move
        self.last_i = 0
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.idx = numpy.random.randint(0, self.blocks.n, size=self.blocks.n)
        count = defaultdict(int)

        self.points = []
        self.blocks.points[self.row] = defaultdict(list)
        for index in self.idx:
            model = self.blocks.data[index]
            if model == self.blocks.models[0]:
                end_x = count[model]
            else:
                end_x = self.blocks.n - count[model] - 1

            point = Point(x=index, y=self.blocks.row, end_x=end_x, end_y=self.row)
            point.model = model
            self.points.append(point)
            self.blocks.points[self.row][model].append(point)

            count[model] += 1

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n_frames_per_sample = linear(
            *self.n_frames_per_sample, step=i, steps=self.n_frames
        )

        n_frames_per_move = linear(*self.n_frames_per_move, step=i, steps=self.n_frames)

        n_frames_until_last_point_moves = len(self.points) * n_frames_per_sample
        n_points = int(linear(0, len(self.points), i, n_frames_until_last_point_moves))
        n_steps = i - self.last_i
        for j, model in enumerate(self.blocks.models):
            for point in self.points[:n_points]:
                for _ in range(n_steps):
                    point.drop(n_frames_per_move)

        self.last_i = i

        self.blocks.redraw()

    def leave(self):
        self(self.n_frames, None, None, None)


class ComputeBootstrapPAB:
    def __init__(self, blocks):
        self.n_frames = (blocks.n_rows - 1) * WriteText("$22$", None).n_frames
        self.blocks = blocks
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.blocks.pabs = []

        pabs = []
        for row in range(1, self.blocks.n_rows):
            pab = len(self.blocks.points[row][self.blocks.models[0]]) / self.blocks.n
            pabs.append(pab)

            text = f"${int(pab * 100):2d}$"
            text_object = self.blocks.ax.text(
                self.blocks.x_pab,
                row,
                "",
                ha="left",
                va="center",
                fontsize=16,
                clip_on=False,
            )
            self.blocks.pabs.append(WriteText(text, text_object))

        self.sections = Section(self.blocks.pabs)
        self.sections.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.sections(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class SortBootstrapedPAB:
    def __init__(self, blocks, n_frames_per_pab, n_frames_per_move):
        self.n_frames = (blocks.n_rows - 1) * n_frames_per_pab + n_frames_per_move
        self.blocks = blocks
        self.n_frames_per_pab = n_frames_per_pab
        self.n_frames_per_move = n_frames_per_move
        self.last_i = 0
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        pabs = [int(write_text.text.strip("$")) for write_text in self.blocks.pabs]
        idx = numpy.argsort(pabs)

        self.pabs = []
        self.blocks.sorted_pabs = []
        for i, index in enumerate(idx):
            point = Point(
                x=self.blocks.x_pab,
                y=index + 1,
                end_x=self.blocks.x_pab_sorted,
                end_y=i + 1,
            )
            pab = pabs[index]
            point.pab = pab
            self.blocks.sorted_pabs.append(point)
            self.pabs.append(
                self.blocks.ax.text(
                    self.blocks.x_pab,
                    index + 1,
                    f"${pab:2d}$",
                    ha="left",
                    va="center",
                    fontsize=16,
                    clip_on=False,
                )
            )

        # Copy texts, find where they should be (sorted_idx)
        # Move them using points

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n_frames_until_last_point_moves = len(self.pabs) * self.n_frames_per_pab
        n_points = int(linear(0, len(self.pabs), i, n_frames_until_last_point_moves))
        n_steps = i - self.last_i
        for text, point in zip(self.pabs, self.blocks.sorted_pabs[:n_points]):
            for _ in range(n_steps):
                point.drop(self.n_frames_per_move)
            text.set_position((point.x, point.y))

        self.last_i = i

        self.blocks.redraw()

    def leave(self):
        self(self.n_frames, None, None, None)


class SlideLine:
    def __init__(self, n_frames, ax, x0, x1, y0, y1):
        self.n_frames = n_frames
        self.ax = ax
        self.x = (x0, x1)
        self.y = (y0, y1)
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.line = self.ax.plot(
            (self.x[0], self.x[0]),
            (self.y[0], self.y[0]),
            color="black",
            linestyle="--",
            clip_on=False,
        )[0]

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        x1 = linear(self.x[0], self.x[1], i, self.n_frames)
        y1 = linear(self.y[0], self.y[1], i, self.n_frames)
        self.line.set_xdata((self.x[0], x1))
        self.line.set_ydata((self.y[0], y1))

    def leave(self):
        self(self.n_frames, None, None, None)


class AddLowerBound:
    def __init__(self, n_frames, blocks):
        self.n_frames = n_frames
        self.blocks = blocks
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        pabs = [point.pab for point in self.blocks.sorted_pabs]

        alpha = 0.05

        lower = numpy.percentile(pabs, alpha / 2 * 100)
        index = numpy.searchsorted(pabs, lower, side="right")

        self.blocks.lower = lower

        y = index + 1
        if lower < pabs[index]:
            y -= 0.5
        else:
            y += 0.5

        text_object = self.blocks.ax.text(
            self.blocks.x_pab_bounds,
            y,
            "",
            ha="left",
            va="center",
            fontsize=16,
            clip_on=False,
        )

        self.text = WriteText(f"${int(lower):2d}$", text_object)
        self.text.initialize(fig, ax, last_animation)

        self.line = SlideLine(
            self.n_frames - self.text.n_frames,
            self.blocks.ax,
            x0=self.blocks.x_pab_sorted,
            x1=self.blocks.x_pab_bounds,
            y0=y,
            y1=y,
        )
        self.line.initialize(fig, ax, last_animation)

        self.sections = Section([self.line, self.text])
        self.sections.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.sections(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class AddUpperBound:
    def __init__(self, n_frames, blocks):
        self.n_frames = n_frames
        self.blocks = blocks
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        pabs = [point.pab for point in self.blocks.sorted_pabs]

        alpha = 0.05

        upper = numpy.percentile(pabs, (1 - alpha / 2) * 100)

        index = numpy.searchsorted(pabs, upper, side="left")

        self.blocks.upper = upper

        y = index + 1
        if upper <= pabs[index]:
            y -= 0.5
        else:
            y += 0.5

        text_object = self.blocks.ax.text(
            self.blocks.x_pab_bounds,
            y,
            "",
            ha="left",
            va="center",
            fontsize=16,
            clip_on=False,
        )

        self.text = WriteText(f"${int(upper):2d}$", text_object)
        self.text.initialize(fig, ax, last_animation)

        self.line = SlideLine(
            self.n_frames - self.text.n_frames,
            self.blocks.ax,
            x0=self.blocks.x_pab_sorted,
            x1=self.blocks.x_pab_bounds,
            y0=y,
            y1=y,
        )
        self.line.initialize(fig, ax, last_animation)

        self.sections = Section([self.line, self.text])
        self.sections.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.sections(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class AddCI:
    def __init__(self, n_frames, blocks):
        self.n_frames = n_frames
        self.center_width = 0.5
        self.whisker_width = 0.25
        self.whisker_length = 5
        self.y = -3
        self.blocks = blocks
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        diff = self.blocks.upper - self.blocks.lower
        left_frac = (self.blocks.data_pab - self.blocks.lower) / diff
        right_frac = (self.blocks.upper - self.blocks.data_pab) / diff

        self.whisker_length = (
            numpy.array([left_frac, right_frac]) * self.whisker_length * 2
        )

        print(self.whisker_length.shape)
        print(list(self.whisker_length * 0.01))

        self.moustacho_plot = h_moustachos(
            self.blocks.ax,
            x=self.blocks.x_pab_sorted,
            y=self.y,
            whisker_width=self.whisker_width * 0.01,
            whisker_length=tuple(self.whisker_length * 0.01),
            center_width=self.center_width * 0.01,
            clip_on=False,
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        whisker_width = linear(
            self.whisker_width * 0.01, self.whisker_width, i, self.n_frames
        )
        whisker_length = linear(
            self.whisker_length * 0.01, self.whisker_length, i, self.n_frames
        )
        center_width = linear(
            self.center_width * 0.01, self.center_width, i, self.n_frames
        )

        adjust_h_moustachos(
            self.moustacho_plot,
            x=self.blocks.x_pab_sorted,
            y=self.y,
            whisker_width=whisker_width,
            whisker_length=tuple(whisker_length),
            center_width=center_width,
        )

        if i == self.n_frames:
            self.blocks.ax.text(
                self.blocks.x_pab_sorted - whisker_length[0],
                self.y - 1,
                f"${int(self.blocks.lower):2d}$",
                ha="center",
                va="center",
                fontsize=16,
                clip_on=False,
            )
            self.blocks.ax.text(
                self.blocks.x_pab_sorted,
                self.y - 1,
                f"${int(self.blocks.data_pab):2d}$",
                ha="center",
                va="center",
                fontsize=16,
                clip_on=False,
            )
            self.blocks.ax.text(
                self.blocks.x_pab_sorted + whisker_length[1],
                self.y - 1,
                f"${int(self.blocks.upper):2d}$",
                ha="center",
                va="center",
                fontsize=16,
                clip_on=False,
            )

    def leave(self):
        self(self.n_frames, None, None, None)


def build_intro(position=numbering()):

    comparison = ComparisonMethod(
        FPS * 1, "", 0.2, width=0.6, y_margin=0.35, height=0.3
    )
    # TODO: Add A and B labels, but wait before adding the curves
    sections = [
        Cover(FPS * 10),
        comparison,
        AddModel(FPS * 1, comparison, "A", mean=1, std=2, scale=0.85, fontsize=24),
        AddModel(FPS * 1, comparison, "B", mean=-1, std=2, scale=0.85, fontsize=24),
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
                [comparison],
                foo=functools.partial(change_model, modif=modif),
            )
        )

    sections.append(Still(FPS * 5))

    modifs = [(10, 2), (1 / 10, 1 / 2), (1 / 10, 1 / 10)]

    for modif in modifs:

        sections.append(
            ChangeDists(
                FPS * 1,
                [comparison],
                foo=functools.partial(change_model, scale=modif),
            )
        )

    sections += [
        Still(FPS * 5),
    ]

    return Chapter("Intro", sections, pbar_position=position)


def build_random(position=numbering()):

    v_pos = ZOrder(0)
    init = VarianceSource("Weights init", v_pos())
    order = VarianceSource("Data order", v_pos())
    splits = VarianceSource("Data splits", v_pos())

    weights_row = ZOrder(0)
    order_row = ZOrder(0)
    splits_row = ZOrder(0)

    sections = [
        ChapterTitle(FPS * 5, position, "Randomize, randomize, randomize"),
        init,
        WriteLabel(init),
        AddWeights(FPS * 1, init, row=weights_row()),
        Still(FPS * 1),
        AddWeights(FPS * 1, init, row=weights_row()),
        AddWeights(int(FPS * 0.5), init, row=weights_row()),
        AddWeights(int(FPS * 0.25), init, row=weights_row()),
        Still(FPS * 1),
        order,
        WriteLabel(order),
        AddOrderSplit(FPS * 5, order, row=order_row(), blocks=(4, 4)),
        AddOrderSplit(FPS * 1, order, row=order_row(), blocks=(4, 4)),
        AddOrderSplit(int(FPS * 0.5), order, row=order_row(), blocks=(4, 4)),
        AddOrderSplit(int(FPS * 0.25), order, row=order_row(), blocks=(4, 4)),
        Still(FPS * 1),
        splits,
        WriteLabel(splits),
        AddOrderSplit(FPS * 5, splits, row=splits_row(), blocks=(3, 2, 2), new=True),
        AddOrderSplit(FPS * 1, splits, row=splits_row(), blocks=(3, 2, 2), new=True),
        AddOrderSplit(
            int(FPS * 0.5), splits, row=splits_row(), blocks=(3, 2, 2), new=True
        ),
        AddOrderSplit(
            int(FPS * 0.25), splits, row=splits_row(), blocks=(3, 2, 2), new=True
        ),
        Still(FPS * 1),
        WriteOther(splits, "Data augmentation, Dropout, ...", 0.2, 0.2),
        Still(FPS * 5),
        WriteOther(splits, "Hyperparameter optimization", 0.2, 0.1),
        Still(FPS * 5),
    ]

    return Chapter("Randomization", sections, pbar_position=position)


def build_pairing(position=numbering()):
    v_pos = ZOrder(0)

    splits_example = VarianceSource("Data splits", position=2, total=3)
    example_row = ZOrder(0)

    sections = [ChapterTitle(FPS * 5, position, "Paired comparisons")]

    sections.append(
        Chapter(
            "",
            [
                splits_example,
                WriteLabel(splits_example),
                AddOrderSplit(
                    FPS * 1,
                    splits_example,
                    row=example_row(),
                    blocks=(5, 3),
                    new=True,
                    choices=list("EEEEEHHH"),
                    label="A",
                    shuffle=False,
                ),
                AddOrderSplit(
                    FPS * 1,
                    splits_example,
                    row=example_row(),
                    blocks=(5, 3),
                    new=True,
                    choices=list("EEEHHEEE"),
                    label="B",
                    shuffle=False,
                ),
                Still(FPS * 10),
                RemoveOrderSplitRow(FPS * 1, splits_example, row=example_row.get()),
                AddOrderSplit(
                    FPS * 1,
                    splits_example,
                    row=example_row.get(),
                    blocks=(5, 3),
                    new=True,
                    choices=list("EEEEEHHH"),
                    shuffle=False,
                ),
                AddOrderSplit(
                    int(FPS * 0.5),
                    splits_example,
                    row=example_row(),
                    blocks=(5, 3),
                    new=True,
                    choices=list("EEHEHEHH"),
                    shuffle=False,
                    label="A",
                ),
                AddOrderSplit(
                    int(FPS * 0.5),
                    splits_example,
                    row=example_row(),
                    blocks=(5, 3),
                    new=True,
                    choices=list("EEHEHEHH"),
                    shuffle=False,
                    label="B",
                ),
                Still(FPS * 5),
            ],
        )
    )

    order = VarianceSource("Data order", 2, total=3)
    order_row = ZOrder(0)

    sections.append(
        Chapter(
            "",
            [
                order,
                WriteLabel(order),
                AddOrderSplit(
                    FPS * 1, order, row=order_row(), blocks=(4, 4), seed=1, label="A"
                ),
                AddOrderSplit(
                    FPS * 1, order, row=order_row(), blocks=(4, 4), seed=1, label="B"
                ),
                AddOrderSplit(
                    int(FPS * 0.5),
                    order,
                    row=order_row(),
                    blocks=(4, 4),
                    seed=2,
                    label="A",
                ),
                AddOrderSplit(
                    int(FPS * 0.25),
                    order,
                    row=order_row(),
                    blocks=(4, 4),
                    seed=2,
                    label="B",
                ),
                Still(FPS * 5),
            ],
        )
    )

    init = VarianceSource("Weights init", 2, total=3)

    weights_row = ZOrder(0)

    sections.append(
        Chapter(
            "",
            [
                init,
                WriteLabel(init),
                AddWeights(int(FPS * 0.5), init, row=weights_row(), seed=1, label="A"),
                AddWeights(
                    int(FPS * 0.5),
                    init,
                    row=weights_row(),
                    arch=(2, 3, 2),
                    seed=1,
                    label="B",
                ),
                Still(FPS * 5),
                RemoveWeights(
                    int(FPS * 0.5), init, row=weights_row.get(), arch=(2, 3, 2)
                ),
                AddWeights(int(FPS * 0.5), init, row=weights_row.get(), seed=1),
                AddWeights(int(FPS * 0.5), init, row=weights_row(), seed=2, label="A"),
                AddWeights(int(FPS * 0.5), init, row=weights_row(), seed=2, label="B"),
                Still(FPS * 5),
            ],
        )
    )

    return Chapter("Pairing", sections, pbar_position=position)


def build_compute_pab(position=numbering()):
    comparison = ComparisonMethod(
        FPS * 1, "", 0.25, width=0.5, y_margin=0.5, height=0.2
    )
    sections = [
        ChapterTitle(FPS * 5, position, "Computation of $P(A>B)$"),
        comparison,
        AddModel(
            FPS * 1,
            comparison,
            "A",
            mean=1,
            std=2,
            scale=0.85,
            fontsize=24,
            clip_on=False,
        ),
        AddModel(
            FPS * 1,
            comparison,
            "B",
            mean=-1,
            std=2,
            scale=0.85,
            fontsize=24,
            clip_on=False,
        ),
        Still(FPS * 5),
        ComputePAB(FPS * 5, comparison),
        ToyPABSimulation(comparison, end_y=-0.5),
        Still(FPS * 5),
        AddPABWhiskers(FPS * 5, comparison),
        Still(FPS * 2),
        RemoveBlocks(FPS * 5, comparison),
    ]

    return Chapter("Compute PAB", sections, pbar_position=position)


def build_conf_interval(position=numbering()):
    blocks = AddBlocks(FPS * 1, pab=0.75, n=22)

    bootstrap_row = ZOrder(0)

    n_rows = 12

    end_n_frames_per_sample = 2
    end_n_frames_per_move = 15

    sections = [
        ChapterTitle(FPS * 5, position, "Computation of the confidence interval"),
        blocks,
        SampleBlocks(
            blocks,
            row=bootstrap_row(),
            n_frames_per_sample=(int(FPS / 2), end_n_frames_per_sample),
            n_frames_per_move=(FPS, end_n_frames_per_move),
        ),
    ]

    for i in range(n_rows - 2):
        sections.append(
            SampleBlocks(
                blocks,
                row=bootstrap_row(),
                n_frames_per_sample=2,
                n_frames_per_move=15,
            )
        )

    bootstrap_row = ZOrder(0)

    sections += [
        ComputeBootstrapPAB(blocks),
        Still(FPS * 5),
    ]

    sections += [
        Still(FPS * 5),
        SortBootstrapedPAB(blocks, n_frames_per_pab=15, n_frames_per_move=15),
        AddLowerBound(FPS * 1, blocks),
        AddUpperBound(FPS * 1, blocks),
        AddCI(FPS * 1, blocks),
        Still(FPS * 5),
    ]

    return Chapter("Conf interval", sections, pbar_position=position)


def build_statistical_test(position=numbering()):
    sections = [ChapterTitle(FPS * 5, position, "Statistical test with $P(A>B)$")]

    return Chapter("Stat test", sections, pbar_position=position)


def build_sample_size(position=numbering()):
    sections = [ChapterTitle(FPS * 5, position, "Sample size")]

    return Chapter("Sample size", sections, pbar_position=position)


def build_recap(position=numbering()):
    sections = [ChapterTitle(FPS * 5, position, "Recap")]

    return Chapter("Recap", sections, pbar_position=position)


chapters = dict(
    intro=build_intro,
    randomizing=build_random,
    pairing=build_pairing,
    sample_size=build_sample_size,
    compute_pab=build_compute_pab,
    conf_interval=build_conf_interval,
    statistical_test=build_statistical_test,
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
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--concat", action="store_true", default=False)

    options = parser.parse_args(argv)

    if options.chapters is None:
        options.chapters = list(chapters.keys())

    args = [(chapter, options) for chapter in options.chapters]

    if options.parallel:
        with Pool() as p:
            p.starmap(create_video, args)
    else:
        list(itertools.starmap(create_video, args))

    names = [
        "intro",
        "randomizing",
        "pairing",
        "compute_pab",
        "conf_interval",
        "statistical_test",
        "sample_size",
        "recap",
    ]

    if options.concat:
        concatenate_videos(names)


def concatenate_videos(names):
    videos = []
    for name in names:
        videos.append(VideoFileClip(f"procedure/{name}.mp4"))

    final_clip = concatenate_videoclips(videos)
    final_clip.write_videofile("procedure/procedure.mp4")


def create_video(chapter, options):

    width = 1280 / options.dpi
    height = 720 / options.dpi

    animate = Animation(
        [chapters[chapter]()],
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
    anim.save(f"procedure/{chapter}.mp4", writer=writer, dpi=options.dpi)


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
