import copy
import numpy
from matplotlib import pyplot as plt
from utils import linear

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def rain_std(rect, data):
    rect.set_width(data.std())


def rained_histogram(
    scatter,
    data,
    y_min,
    y_max,
    y_sky,
    step,
    steps,
    delta,
    block_height=1,
    spacing=1,
    subset=None,
    n_columns=10,
    marker_size=3,
    sky_padding=1.1,
    y_padding=0,
):
    normalized_data = copy.deepcopy(data)
    normalized_data -= normalized_data.min()
    normalized_data /= normalized_data.max()
    columns = numpy.linspace(0, 1, n_columns)
    data_col_idx = numpy.digitize(normalized_data, columns, right=False)
    # Build bins (columns)
    # Stack blocks in each bin

    data_final_y = numpy.ones(len(data)) * -1
    for i in range(len(data)):
        data_final_y[i] = (data_col_idx[i] == data_col_idx[:i]).sum() * block_height

    rescaled_x = data_col_idx / n_columns
    rescaled_y = data_final_y / data_final_y.max() * (y_max - y_min) + y_min
    data_steps = (numpy.arange(len(data))[::-1] - len(data)) * spacing + step * delta
    data_steps = numpy.clip(data_steps, 0, steps)
    rescaled_y = linear(y_sky * sky_padding, rescaled_y, data_steps, steps)
    rescaled_y += y_padding
    offsets = list(zip(rescaled_x, rescaled_y))

    if subset is not None:
        all_offsets = scatter.get_offsets()
        all_offsets[subset] = offsets
        offsets = all_offsets

    scatter.set_offsets(offsets)
    scatter.set_sizes([marker_size])

    return data[data_steps == steps]


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2)
    ax = axes[1]
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 10))
    scatter = ax.scatter([], [], alpha=1, marker="s")
    data = numpy.ones(400) * -1000
    scatter.set_offsets(list(zip(data, data)))

    rects = axes[0].barh(range(2), [0, 0])

    # axes[0].set_ylim(-0.5, 1.5)
    data_bottom = numpy.random.normal(size=200)
    data_top = numpy.random.normal(size=200)
    axes[0].set_xlim(0, max(data_bottom.std(), data_top.std()) * 1.25)

    def animate(i):
        if i < 100:
            hit_the_ground = rained_histogram(
                scatter,
                data_bottom,
                y_min=0,
                y_max=4,
                y_sky=10,
                step=i,
                steps=100,
                delta=5,
                subset=slice(0, 200),
                n_columns=20,
            )
            rain_std(rects[0], hit_the_ground)
        else:
            hit_the_ground = rained_histogram(
                scatter,
                data_top,
                y_min=5,
                y_max=10,
                y_sky=10,
                step=i - 100,
                steps=100,
                delta=10,
                subset=slice(200, 400),
                n_columns=20,
            )
            rain_std(rects[1], hit_the_ground)
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
    anim.save("raining.mp4", writer=writer, dpi=100)
