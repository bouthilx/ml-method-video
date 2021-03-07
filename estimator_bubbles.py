import copy
import numpy
from matplotlib import pyplot as plt
from utils import linear

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from moustachos import adjust_h_moustachos, h_moustachos, moustachos, adjust_moustachos


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
        self.scatter = ax.scatter([], [], alpha=1, marker="|", clip_on=False, color=color)

        for row in range(n_rows):
            self.data[row] = []

        self.estimator_whisker = h_moustachos(
            ax,
            0,
            self._get_row_y(n_rows + 1),
            whisker_width=1,
            whisker_length=1,
            center_width=1,
            # center_line_width=None,
            # whisker_line_width=None,
        )

        for row in range(n_rows):
            self.whisker[row] = h_moustachos(
                ax,
                0,
                self._get_row_y(row),
                whisker_width=1,
                whisker_length=1,
                center_width=1,
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

        adjust_h_moustachos(
            self.estimator_whisker,
            mean,
            self._get_row_y(self.n_rows + 1),
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
