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


def plot_line(
    axe,
    x,
    y,
    err=None,
    label=None,
    alpha=0.5,
    color=None,
    linestyle=None,
    min_y=None,
    max_y=None,
):
    plots = {}
    plots["line"] = axe.plot(x, y, label=label, color=color, linestyle=linestyle)[0]

    if err is not None:
        plots["err"] = plot_err(
            axe,
            x,
            y,
            err=err,
            alpha=alpha,
            color=color,
            min_y=min_y,
            max_y=max_y,
        )

    return plots


def plot_err(
    axe,
    x,
    y,
    err=None,
    alpha=0.5,
    color=None,
    min_y=None,
    max_y=None,
):
    y = numpy.array(y)
    err = numpy.array(err)

    min_y_err = y - err
    if min_y is not None:
        min_y_err = min_y_err * (min_y_err > min_y) + min_y * (min_y_err <= min_y)

    max_y_err = y + err
    if max_y is not None:
        max_y_err = max_y_err * (max_y_err <= max_y) + max_y * (max_y_err > max_y)

    # axe.fill_between(x, max_y, min_y, linewidth=0, alpha=alpha, color=color)
    return axe.fill_between(
        x, min_y_err, max_y_err, linewidth=0, alpha=alpha, color=color
    )


class VLineLabel:
    def __init__(self, ax, fontsize=14, color="black", linestyle="--", **kwargs):
        self.label = ax.text(0, 0, "", fontsize=fontsize, clip_on=False)
        self.line = ax.plot([], [], color=color, linestyle=linestyle, clip_on=False)[0]

    def set_position(self, x, y, text, min_y=0, pad_y=1):
        self.label.set_position((x, y + pad_y))
        self.label.set_text(text)
        self.line.set_xdata([x, x])
        self.line.set_ydata([min_y, y])


def adjust_line(
    ax,
    plots,
    x,
    y,
    err=None,
    alpha=0.5,
    color=None,
    linestyle=None,
    min_y=None,
    max_y=None,
):
    plots["line"].set_xdata(x)
    plots["line"].set_ydata(y)
    if err is not None:
        plots["err"].remove()
        plots["err"] = plot_err(
            ax,
            x,
            y,
            err=err,
            alpha=alpha,
            color=color,
            min_y=min_y,
            max_y=max_y,
        )


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
