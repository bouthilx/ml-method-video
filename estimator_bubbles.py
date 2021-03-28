import copy
import numpy
from matplotlib import pyplot as plt
from utils import linear, translate

import matplotlib.transforms
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from moustachos import adjust_h_moustachos, h_moustachos, moustachos, adjust_moustachos

from transitions import FadeOut
from compositions import Cascade, Section, reverse


def rain_std(rect, data):
    rect.set_width(data.std())


# Increase k
# Adjust rho


class EstimatorBubbles:
    def __init__(self, ax, n_rows, std, rho=0, color=None):
        self.n_rows = n_rows
        self.std = std
        self.rho = rho
        self.data = {}
        self.whisker = {}
        self.ax = ax
        self.scatter = ax.scatter(
            [], [], alpha=1, marker="|", clip_on=False, color=color
        )

        for row in range(n_rows):
            self.data[row] = []

        self.top_y = 2
        self.estimator_whisker = None

        for row in range(n_rows):
            self.whisker[row] = h_moustachos(
                ax,
                0,
                self._get_row_y(row),
                whisker_width=1,
                whisker_length=1,
                center_width=1,
                clip_on=False,
                # center_line_width=None,
                # whisker_line_width=None,
            )

    def add_top(self):
        self.estimator_whisker = h_moustachos(
            self.ax,
            0,
            self._get_row_y(self.n_rows + self.top_y),
            whisker_length=1,
            whisker_width=0.5,
            center_width=1,
            clip_on=False,
            # center_line_width=None,
            # whisker_line_width=None,
        )

    def get_k(self):
        return len(self.data[0])

    def _new_point(self):
        return numpy.random.normal(0, self.std)

    def _get_row_y(self, row):
        return row

    def increase_k(self):
        for row in self.data.keys():
            self.data[row].append(self._new_point())
        self.refresh()

    def refresh(self):
        standardized_data = self.get_standardized_data()
        self.update_whiskers(standardized_data)
        self.update_scatter(standardized_data)

    def adjust_rho(self, rho):
        self.rho = rho
        self.refresh()

    def get_mean_std(self, data):

        means = []
        for row_data in data.values():
            means.append(numpy.array(row_data).mean())
        means = numpy.array(means)

        return means.mean(), means.std()

    def get_std(self, row):
        k = len(self.data[row])
        return numpy.sqrt(self.std ** 1 / k + (k - 1) / k * self.rho * self.std ** 2)

    def get_standardized_data(self):
        mean, std = self.get_mean_std(self.data)
        standardized_data = {}
        for row in self.data.keys():
            data = numpy.array(self.data[row])
            row_mean = data.mean()
            # TODO: Separate data and standardized data. Do not reapply standardization
            # on standardized data.
            standardized_data[row] = list(
                data - mean + row_mean / std * self.get_std(row)
            )
            # self.data[row] = list(data - mean + mean * self.get_std(row))
            # self.data[row] = list((self.data[key]) / std * self.get_std(key))
        return standardized_data

    def update_whiskers(self, data):

        mean, std = self.get_mean_std(data)

        if self.estimator_whisker is not None:
            adjust_h_moustachos(
                self.estimator_whisker,
                mean,
                self._get_row_y(self.n_rows + self.top_y),
                whisker_length=std,
                whisker_width=0.5,
                center_width=1,
            )

        for row in self.data.keys():
            self.update_whisker(data[row], row)

    def update_whisker(self, data, row):
        mean = numpy.array(data).mean()
        std = numpy.array(data).std()

        adjust_h_moustachos(
            self.whisker[row],
            mean,
            self._get_row_y(row),
            whisker_length=std,
            whisker_width=0.2,
            center_width=0.4,
        )

    def update_scatter(self, data):
        scatter_data = []
        for row in data.keys():
            y = self._get_row_y(row)
            scatter_data += [(x, y) for x in data[row]]

        self.scatter.set_offsets(scatter_data)


class FadeInTopWhisker:
    def __init__(self, n_frames, estimators):
        self.n_frames = n_frames
        self.estimators = estimators
        self.initialized = False

    def create_fade_in(self, estimator, n_frames, row):
        return reverse(
            FadeOut(
                n_frames,
                ax=estimator.ax,
                x=-5,
                width=10,
                y=estimator._get_row_y(row) - 1,
                height=2,
                opacity=1,
                transform=estimator.ax.transData,
                # color=matplotlib.cm.get_cmap("tab10")(row),
            )
        )

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.fade_ins = {}
        for name, estimator in self.estimators.estimators.items():
            estimator.add_top()
            self.fade_ins[name] = self.create_fade_in(
                estimator, self.n_frames, estimator.n_rows + estimator.top_y
            )
            self.fade_ins[name].initialize(fig, ax, last_animation)
            # Force creation of white blocks
            self.fade_ins[name](0, fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for fade_in in self.fade_ins.values():
            fade_in(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)
        for fade_in in self.fade_ins.values():
            fade_in.animation.patch.remove()


class FadeInWhiskers:
    def __init__(self, n_frames, estimators, key, cascade=None):
        self.fade_in_n_frames = n_frames

        self.estimators = estimators
        self.key = key

        missing = max(estimators.n_rows - len(n_frames), 0)
        if cascade is None:
            cascade = n_frames[-1]

        self.cascade_frames = (cascade, [n_frames[-1] for _ in range(missing)])

        self.n_frames = sum(n_frames) + Cascade.infer_n_frames(*self.cascade_frames)

        self.initialized = False

    @property
    def ax(self):
        return self.estimators.estimators[self.key].ax

    def get_row_y(self, row):
        return self.estimators.estimators[self.key]._get_row_y(row)

    def create_fade_in(self, n_frames, row):
        return reverse(
            FadeOut(
                n_frames,
                ax=self.ax,
                x=-5,
                width=10,
                y=self.get_row_y(row) - 0.5,
                height=1,
                opacity=1,
                transform=self.ax.transData,
                # color=matplotlib.cm.get_cmap("tab10")(row),
            )
        )

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.seq_fade_ins = []
        row = self.estimators.n_rows - 1
        for n_frames in self.fade_in_n_frames:
            fade_in = self.create_fade_in(n_frames, row)
            fade_in.initialize(fig, ax, last_animation)
            # Force creation of white blocks
            fade_in(0, fig, ax, last_animation)
            # print(fade_in.x, fade_in.y, fade_in.height)
            self.seq_fade_ins.append(fade_in)
            row -= 1

        self.cascade_fade_ins = []
        for n_frames in self.cascade_frames[1]:
            fade_in = self.create_fade_in(n_frames, row)
            fade_in.initialize(fig, ax, last_animation)
            # Force creation of white blocks
            fade_in(0, fig, ax, last_animation)
            self.cascade_fade_ins.append(fade_in)
            row -= 1

        assert row == -1, row

        self.cascade_section = Cascade(self.cascade_frames[0], self.cascade_fade_ins)

        self.sections = Section(self.seq_fade_ins + [self.cascade_section])

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.sections(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)
        for fade_in in self.seq_fade_ins + self.cascade_fade_ins:
            fade_in.animation.patch.remove()


class AlignWhiskerPositions:
    def __init__(self, n_frames, estimators):
        self.n_frames = n_frames
        self.estimators = estimators
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for estimator in self.estimators.estimators.values():
            estimator.data_backup = copy.deepcopy(estimator.data)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        frac = linear(0, 1, i, self.n_frames)
        for estimator in self.estimators.estimators.values():
            for row in estimator.data.keys():
                data = numpy.array(estimator.data_backup[row])
                estimator.data[row] = list(data - frac * data.mean())
            print(i, numpy.array(estimator.data[0]).mean())
            estimator.refresh()

    def leave(self):
        self(self.n_frames, None, None, None)


class ResetWhiskerPositions:
    def __init__(self, n_frames, estimators):
        self.n_frames = n_frames
        self.estimators = estimators
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        frac = translate(0, 1, i, self.n_frames)
        for estimator in self.estimators.estimators.values():
            for row in estimator.data.keys():
                data = numpy.array(estimator.data_backup[row])
                estimator.data[row] = list(data + frac * data.mean())
            estimator.refresh()

    def leave(self):
        self(self.n_frames, None, None, None)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.set_xlim((-5, 5))
        ax.set_ylim((-1, 22))
        scatter = ax.scatter([], [], alpha=1, marker="s")

    estimator_left = EstimatorBubbles(axes[0], n_rows=20, std=1, rho=0)
    estimator_right = EstimatorBubbles(axes[1], n_rows=20, std=1, rho=0)
    # Sink the data
    estimator_right.data = estimator_left.data

    def animate(i):
        if i < 100:
            estimator_left.increase_k()
            estimator_right.refresh()
        else:
            estimator_right.adjust_rho((i - 100) / 100 * 2)
        return (scatter,)

    anim = FuncAnimation(
        fig,
        animate,
        frames=200,
        interval=20,
        blit=True,
    )

    Writer = animation.writers["ffmpeg"]
    writer = Writer(
        fps=30,
        metadata=dict(artist="Xavier Bouthillier"),
    )

    width = 1280 / 100
    height = 720 / 100

    fig.set_size_inches(width, height, True)
    anim.save("estimator.mp4", writer=writer, dpi=100)
