from collections import OrderedDict
from datetime import datetime
import numpy
import seaborn as sns
import hashlib
import json
import os


def ornstein_uhlenbeck_step(mean, past_position, stability, standard_deviation):
    return stability * (mean - past_position) + numpy.random.normal(
        0, standard_deviation
    )


def sigmoid(t):
    return 1 / (1 + numpy.exp(-t))


def linear(a, b, step, steps):
    if steps == 0:
        return b
    return a + numpy.clip(step / steps, a_min=0, a_max=1) * (b - a)


def translate(a, b, step, steps, saturation=10):
    return a + (sigmoid(step / steps * saturation) - 0.5) * 2 * (b - a)


def show_text(obj, text, step, steps, min_i=0, fill=True):
    # TODO: Detect any latex $$ and do not count them, remove only
    #       text within the $$ block.
    text_length = len(text) - text.count("$")
    n = int(linear(min_i, text_length + 1, step, steps))
    new_text = text[:n]
    while text[n : n + new_text.count("$")].count("$"):
        new_text = text[n : n + new_text.count("$")]
        n += len(new_text)
        # n += new_text.count("$")
    n += new_text.count("$")
    new_text = text[:n]
    if new_text.count("$") % 2 == 1:
        if new_text[-1] == "$":
            import pdb

            pdb.set_trace()
            new_text = new_text[:-1]
        else:
            new_text += "$"
    if fill:
        new_text += " " * (text_length - (n - new_text.count("$")))
    obj.set_text(new_text)


def despine(ax):
    sns.despine(ax=ax, bottom=True, left=True)
    # for side in ["top", "right", "bottom", "left"]:
    #     ax.spines[side].set_visible(False)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


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


def compute_identity(data, size=16):
    dhash = hashlib.sha256()

    for k, v in sorted(data.items()):
        dhash.update(k.encode("utf8"))

        if isinstance(v, (dict, OrderedDict)):
            dhash.update(compute_identity(v, size).encode("utf8"))
        else:
            dhash.update(str(v).encode("utf8"))

    return dhash.hexdigest()[:size]


class Cache:
    def __init__(self, cache_file, waittime=60):
        self.cache_file = cache_file
        self.last_update = None
        self.waittime = waittime
        self.load()
        self.hit = 0
        self.miss = 0

    def _get_key(self, item):
        simulation, test = item
        sim_hash = simulation.get_hash()
        test_hash = test.get_hash()
        return f"{sim_hash}:{test_hash}"

    def __contains__(self, item):
        return self._get_key(item) in self._cache

    def compute(self, simulation, test):
        key = self._get_key((simulation, test))
        if key not in self._cache:
            self.miss += 1
            result = test(simulation)
            self.update(key, result)
        else:
            self.hit += 1
            result = self._cache[key]

        return result

    def load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def save(self):
        tmp_file = self.cache_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(self._cache, f)
        os.rename(tmp_file, self.cache_file)
        print(f"hit {self.hit} {(self.hit / (self.hit + self.miss)) * 100:.1f}%")
        print(f"miss {self.miss} {(self.miss / (self.hit + self.miss)) * 100:.1f}%")

    @property
    def delay(self):
        return (datetime.now() - self.last_update).total_seconds()

    def update(self, key, result):
        self._cache[key] = result
        if self.last_update is None or self.delay > self.waittime:
            self.last_update = datetime.now()
            self.save()


def precision(number, decimals=3):
    return float(numpy.format_float_scientific(number, precision=decimals - 1))
