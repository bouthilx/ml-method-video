import copy
from collections import defaultdict
import os
import json
import time

import pandas
import numpy
import scipy.stats
import scipy.special
import xarray

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter
from itertools import groupby

import joblib

from sklearn.neighbors import KernelDensity

from olympus.studies.simul.main import load_results


WIDTH = (8.5 - 1.5) / 2
HEIGHT = (11 - 1.5) / 2

# Prepare matplotlib
plt.rcParams.update({"font.size": 8})
plt.close("all")
plt.rc("font", family="Times New Roman")
# plt.rc('text', usetex=True)
plt.rc("xtick", labelsize=6)
plt.rc("ytick", labelsize=8)
plt.rc("axes", labelsize=8)


# Use joblib to speed things up when rerunning
mem = joblib.Memory("joblib_cache")


@mem.cache
def load_simul_results(namespace, save_dir):
    with open(f"{save_dir}/simul_{namespace}.json", "r") as f:
        data = {
            hpo: {rep_type: xarray.Dataset.from_dict(d) for rep_type, d in reps.items()}
            for hpo, reps in json.loads(f.read()).items()
        }

    return data


@mem.cache
def load_variance_results(namespace, save_dir):
    with open(f"{save_dir}/variance_{namespace}.json", "r") as f:
        data = xarray.Dataset.from_dict(json.loads(f.read()))

    return data


# std = 'mean-normalized'
std = "raw"

# reporting_set = 'valid'
reporting_set = "test"


LABELS = {
    "vgg": "CIFAR10\nVGG11",
    "bert-sst2": "Glue-SST2\nBERT",
    "bert-rte": "Glue-RTE\nBERT",
    "bio-task2": "MHC\nMLP",
    "segmentation": "PascalVOC\nResNet",
    "test_error_rate": "Acc",
    "test_mean_jaccard_distance": "IoU",
    "test_aac": "AUC",
}


case_studies = {
    "vgg": "vgg",
    "segmentation": "segmentation",  # TODO: Set segmentation when data is ready
    "bert-sst2": "sst2",
    "bert-rte": "rte",
    "bio-task2": "bio-task2",
}  # TODO: Set bio-task2 when data is ready
# 'logreg': 'logreg'}


objectives = {
    "vgg": "test_error_rate",
    "segmentation": "test_mean_jaccard_distance",
    # 'segmentation': 'test_error_rate',
    "bert-sst2": "test_error_rate",
    "bert-rte": "test_error_rate",
    "bio-task2": "test_aac",
}


# 'bio-task2': 'test_aac'}
# 'logreg': 'test_error_rate'}

IDEAL_EST = "IdealEst($k$)"
FIXHOPT_EST = "FixHOptEst($k$, {var})"

VARIATIONS = ["Init", "Data", "All"]
ESTIMATORS = [FIXHOPT_EST.format(var=var) for var in VARIATIONS] + [IDEAL_EST]
TASKS = ["bert-rte", "bert-sst2", "bio-task2", "segmentation", "vgg"]
BUDGETS = list(range(1, 101))

colors_strs = [
    "#1f77ba",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#86564b",
    "#e377c2",
    "#d62728",
]
COLORS = dict(zip(ESTIMATORS, colors_strs))


def load_data(root):
    var_root = os.path.join(root, "variance")
    hpo_root = os.path.join(root, "hpo")
    simul_root = os.path.join(root, "simul")

    data = {}

    start = time.time()
    for key, name in case_studies.items():
        print(f"Loading {key}")
        data[key] = {
            "variance": load_variance_results(name + "-var", var_root),
            "simul": load_simul_results(name + "-simul", simul_root),
        }

    elapsed_time = time.clock() - start
    print(f"It took {elapsed_time}s to load all data.")

    return data


def cum_argmin(x):
    """Return the indices corresponding to an cumulative minimum
    (np.minimum.accumulate)
    """
    minima = numpy.minimum.accumulate(x, axis=0)
    diff = numpy.diff(minima, axis=0)
    jumps = numpy.vstack(numpy.arange(x.shape[0]) for _ in range(x.shape[1])).T
    jumps[1:, :] *= diff != 0
    jumps = numpy.maximum.accumulate(jumps, axis=0)
    return jumps


def get_test_metrics(valid, test):
    regrets_idx = cum_argmin(valid)

    regrets_idx = numpy.minimum(
        regrets_idx, test.shape[0] * numpy.ones(regrets_idx.shape) - 1
    ).astype(int)

    return test[regrets_idx[-1], numpy.arange(valid.shape[1])]


def get_var_metrics(valid, test):
    pass


noise_sources = set(["total"])


def std_of_std(std, n):
    # Cap n to 2 minimum, otherwise there is no STD.
    n = (n <= 1) * 2 + (n > 1) * n
    nmd2 = (n - 1) / 2
    gnmd2 = scipy.special.gamma(nmd2)
    gnd2 = scipy.special.gamma(n / 2)
    g_ratio = gnd2 / gnmd2
    return std * (1 / g_ratio) * numpy.sqrt(nmd2 - g_ratio * g_ratio)


def get_stat(a, mean, stat, budgets):
    if stat == "std":
        idx = numpy.arange(a.shape[0])
        N = 200
        stds = numpy.zeros((N, len(budgets)))
        for i in range(N):
            numpy.random.shuffle(idx)
            b = a[idx, :]
            means = b.cumsum(axis=0) / numpy.arange(1, b.shape[0] + 1)[:, None]
            stds[i] = means[numpy.array(budgets) - 1, :].std(1)
        return stds.mean(0), std_of_std(stds.mean(0), numpy.array(budgets))
        # return stds.mean(0), stds.std(0)
        # return stds, std_of_std(stds, numpy.array(budgets))
    elif stat == "bias":
        # return numpy.abs(a.mean() - mean)
        idx = numpy.arange(a.shape[0])
        N = 1
        biases = numpy.zeros((len(budgets), a.shape[1], N))
        for i in range(N):
            numpy.random.shuffle(idx)
            b = a[idx, :]
            means = b.cumsum(axis=0) / numpy.arange(1, b.shape[0] + 1)[:, None]
            biases[:, :, i] = means[numpy.array(budgets) - 1, :] - mean
        biases = biases.reshape((biases.shape[0], -1))
        return biases.mean(1), biases.std(1)
    elif stat == "corr":
        var_r = a.var(0).mean()  # mean variance of R
        var_tmu = a.mean(0).var()  # variance of mu
        k = a.shape[0]  # should be 100
        assert k == 100
        return (k * var_tmu - var_r) / ((k - 1) * var_r)
    elif stat == "rho_var":
        rho = get_stat(a, mean, "corr", budgets)
        return rho / a.var(0).mean()
    elif stat == "mse":
        stds, stds_err = get_stat(a, mean, "std", budgets)
        bias, bias_err = get_stat(a, mean, "bias", budgets)
        return stds ** 2 + bias ** 2, 0 * stds + 0 * bias_err
        # return var_tmu + (a.mean() - mean) ** 2


# stat = "bias"
STAT = "std"
# stat = 'corr'
# stat = "mse"


def get_splits_test_metrics(valid, test, budget):
    n = int(valid.shape[0] / budget)
    ideal_data = numpy.ones(n_ideals * 2) * numpy.nan
    # Split based on budgets
    ideal_data[:n_ideals] = get_test_metrics(valid[:100], test[:100])


def get_ideal_stat(data, estimator, task, budgets, stat_type=STAT):
    hpo_data = data["ideal"]
    max_epoch = int(hpo_data.epoch.max())
    hpo_data = hpo_data.sel(epoch=max_epoch)

    valid = hpo_data[objectives[task].replace("test", "validation")].values
    test = hpo_data[objectives[task]].values

    # NOTE: There is 50 HPOs of 200 points.
    # We divide the 200 in sets of 100 and get the equivalent of 100 HPOs (random search)
    # We did this because we needed 200 points per HPO to fit the surrogate models but
    # we simulate for budgets of 100 trials.
    n_ideals = len(hpo_data.seed)
    ideal_data = numpy.ones(n_ideals * 2) * numpy.nan
    # Split based on budgets
    ideal_data[:n_ideals] = get_test_metrics(valid[:100], test[:100])
    ideal_data[n_ideals:] = get_test_metrics(valid[100:], test[100:])

    # Now we have 100 HPO runs,
    # For 1 budget, we split and compute std over 100
    # For 2 budget, we split in 50 groups of two, compute average and then std on the average
    # For 3 budget, we split in 33 groups of three,
    # At 10 we move to analytical solution
    # Or use bootstrap to estimate? No, not more realistic.

    if stat_type == "mean":
        return ideal_data.mean()
    elif stat_type == "std":
        std = ideal_data[:100].std()
        stds = numpy.array([std / numpy.sqrt(budget) for budget in budgets])
        return stds, std_of_std(stds, numpy.array(budgets))
    elif stat_type == "bias":
        return [0 for budget in budgets], [0 for budget in budgets]
    elif stat_type == "corr":
        return [0 for budget in budgets], [0 for budget in budgets]
    elif stat_type == "mse":
        stds, stds_err = get_ideal_stat(data, estimator, task, budgets, stat_type="std")
        return stds ** 2, 0 * stds_err

    # Keeping for when we need timings
    case_timing = float(hpo_data.elapsed_time.mean())
    case_timing /= 60.0  # mins
    case_timing /= 60.0  # hours
    # Here 100 are based on the budgets, this should be adapted
    timings[key] = dict(ideal=case_timing * 100 * 100)


FIXHOPT_ESTIMATOR_KEYS = {
    FIXHOPT_EST.format(var="Init"): "weights_init",
    FIXHOPT_EST.format(var="Data"): "bootstrap",
    FIXHOPT_EST.format(var="All"): "biased",
}


def get_fix_hoptest_stat(data, estimator, task, budgets, stat=STAT, standardized=False):
    # return [i * 2 for i in budgets]
    hpo_data = data[FIXHOPT_ESTIMATOR_KEYS[estimator]]
    max_epoch = int(hpo_data.epoch.max())
    hpo_data = hpo_data.sel(epoch=max_epoch)

    test = hpo_data[objectives[task]].values
    # We have mean 100 samples, compute mean on up to n
    # Or use bootstrap to estimate variance.
    ideal_mean = get_ideal_stat(data, estimator, task, budgets, stat_type="mean")
    if standardized:
        assert stat not in [
            "bias",
            "mse",
        ], "cannot compute bias (and mse) if standardized"
        test = copy.deepcopy(test)
        test -= test.mean()
        test /= test.std()

    return get_stat(test, ideal_mean, stat, budgets)
    # return get_stat(test, ideal_data.mean(), stat, budgets)

    # Keeping for when timings are needed
    case_timing = float(hpo_data.elapsed_time.mean())
    case_timing /= 60.0  # mins
    case_timing /= 60.0  # hours
    # Here 100 are based on the budgets, this should be adapted
    timings[key]["e-weights_init"] = case_timing * (100 + 100)


ESTIMATE = {IDEAL_EST: get_ideal_stat}
for source in VARIATIONS:
    ESTIMATE[FIXHOPT_EST.format(var=source)] = get_fix_hoptest_stat


def get_timings(estimator, budgets):
    if estimator == IDEAL_EST:
        return 100 * numpy.array(budgets)
    else:
        return 100 + numpy.array(budgets)


def get_est_stats(data, estimator, budgets, tasks=TASKS):
    stats = {}
    for task in tasks:
        print(f"Computing stats for {estimator} on {task}")

        case_data = data[task]["simul"]["random_search"]
        stats[task] = ESTIMATE[estimator](case_data, estimator, task, budgets)

    return stats


start = time.clock()
stats = {}
timings = {}
# for key, case_data in data.items():


# RTE
# Train: 69 2 2213
# Valid: 0 2 277
# Test: 0 2 277

# SST-2
# Train: 2077 2 66477
# Valid: 0 2 872
# Test: 0 2 872

# CIFAR-10
# Train: 1249 2 40000
# Valid: 9 2 10000
# Test: 9 2 10000

# Not classif anyway
# PascalVOC
# Train: 45 2 1464
# Valid: 0 2 724
# Test: 0 2 725

# Not classif anyway
# Bio

sizes = {"bert-rte": 277, "bert-sst2": 872, "vgg": 10000}


perfs = {"bert-rte": 1 - 0.34, "bert-sst2": 1 - 0.95, "vgg": 1 - 0.092}

# This is the ratio of variance between unpaired and paired differences for the init study
# This is not the var ratios between data samplings only, it covers all \xi_o
var_ratios = {
    "vgg": 0.4084229233546418,
    "seg": 0.14456563218570773,
    "bert-sst2": 0.657842327682519,
    "bert-rte": 0.9632991300053148,
    "bio": 0.21598750239583644,
}


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


def overview_line_chart(estimators=ESTIMATORS, tasks=TASKS, budgets=BUDGETS):
    fig, ax = plt.subplots(figsize=(7, 2))

    for estimator in estimators:
        stats = get_stat(estimator, budgets=budgets)
        x = budgets
        y = stats.mean(axis="trials")
        err = stats.std(axis="trials")
        plot_line(ax, x, y, err, label=None, alpha=0.5, color=None, linestyle=None)

    fig.set_size_inches(WIDTH, HEIGHT)

    plt.savefig(f"standard_error_line_chart_{stat}.png", dpi=300)
    plt.savefig(f"standard_error_line_chart_{stat}.pdf", dpi=300)


def timings(budgets=BUDGETS, width=WIDTH, height=HEIGHT):
    fig, ax = plt.subplots(figsize=(7, 2), ncols=1, nrows=1)

    for estimator in [FIXHOPT_EST.format(var="All"), IDEAL_EST]:
        times = get_timings(estimator, budgets=budgets)
        x = budgets
        y = times
        plot_line(
            ax,
            x,
            y,
            err=None,
            label=estimator,
            alpha=0.5,
            color=colors[estimator],
            linestyle=None,
        )

    sns.despine(ax=ax, bottom=False, left=False)

    ax.set_xlabel("Number of samples for the estimator ($k$)")
    ax.set_ylabel("Number of trainings")

    ax.legend(
        bbox_to_anchor=(-0.05, 1, 1.1, 0.5),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )

    fig.set_size_inches(width, height)

    plt.savefig(f"timings.png", dpi=300)
    plt.savefig(f"timings.pdf", dpi=300)


def small_multiples(
    estimators=ESTIMATORS,
    tasks=TASKS,
    budgets=BUDGETS,
    width=WIDTH,
    height=HEIGHT,
    name=f"standard_error_line_chart_small_multiples_{STAT}",
    gridspec_top=0.9,
    gridspec_bottom=0.1,
):

    fig, ax = plt.subplots(
        figsize=(7, 2),
        ncols=1,
        nrows=len(tasks),
        sharex="col",
        gridspec_kw={
            "left": 0.20,
            "top": gridspec_top,
            "right": 0.85,
            "bottom": gridspec_bottom,
            "hspace": 0.2,
            "wspace": 0,
        },
    )

    estimators_plot = EstimatorsPlot()
    estimators_plot.load()

    for i, task in enumerate(tasks):
        estimators_plot.plot(ax[i], task, budgets=budgets, estimators=estimators)

    for i, task in enumerate(tasks):
        sns.despine(ax=ax[i], bottom=False, left=False)
        ax[i].text(
            1, 0.5, LABELS[task], ha="left", va="center", transform=ax[i].transAxes
        )

    ax[-1].set_xlabel("Number of samples for the estimator ($k$)")
    if len(tasks) < 3:
        for i in range(len(tasks)):
            ax[i].set_ylabel("Standard deviation\nof estimators")
    else:
        middle = int(len(tasks) / 2 + 0.5) - 1
        for i, task in enumerate(tasks):
            if i == middle:
                ax[i].set_ylabel(
                    f"Standard deviation\nof estimators\n({LABELS[objectives[task]]})"
                )
            else:
                ax[i].set_ylabel(f"({LABELS[objectives[task]]})")

    ax[0].legend(
        bbox_to_anchor=(-0.1, 1, 1.15, 0.5),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )

    fig.set_size_inches(width, height)

    plt.savefig(f"{name}.png", dpi=300)
    plt.savefig(f"{name}.pdf", dpi=300)


class EstimatorsPlot:
    def __init__(self, root="~/Dropbox/Olympus-Data", colors=COLORS):
        self.root = os.path.expanduser(root)
        self.colors = colors
        self.plots = {}

    def load(self):
        self.data = load_data(self.root)
        self.curves = []

    def get_stat(self, task, estimator, budgets, stat, standardized=True):
        case_data = self.data[task]["simul"]["random_search"]
        return get_fix_hoptest_stat(
            case_data, estimator, task, budgets, stat=stat, standardized=standardized
        )

    def plot(self, ax, task, budgets, estimators=ESTIMATORS):
        self.plots.setdefault(task, {})
        for estimator in estimators:
            self.plots[task].setdefault(estimator, {})
            stats = get_est_stats(self.data, estimator, budgets=budgets, tasks=[task])
            x = budgets
            y, err = stats[task]

            self.plots[task][estimator] = plot_line(
                ax,
                x,
                y,
                err=err,
                max_y=max(y),
                label=estimator,
                alpha=0.5,
                color=self.colors[estimator],
                linestyle=None,
            )

    def update(self, ax, task, budgets, estimators=ESTIMATORS):
        for estimator in estimators:
            stats = get_est_stats(self.data, estimator, budgets=budgets, tasks=[task])
            x = budgets
            y, err = stats[task]

            self.plots[task][estimator]["line"].set_xdata(x)
            self.plots[task][estimator]["line"].set_ydata(y)
            self.plots[task][estimator]["err"].remove()  # What to do with this?

            self.plots[task][estimator]["err"] = plot_err(
                ax,
                x,
                y,
                err=err,
                max_y=max(y),
                alpha=0.5,
                color=self.colors[estimator],
            )


if __name__ == "__main__":

    timings()

    small_multiples(
        tasks=["bert-rte", "segmentation"],
        width=(8.5 - 1.5) / 2,
        height=(11 - 1.5) / 3.5,
        gridspec_top=0.85,
        gridspec_bottom=0.15,
        name=f"standard_error_line_chart_core_{stat}",
    )

    small_multiples(
        tasks=TASKS,
        width=(8.5 - 1.5) / 2,
        height=(11 - 1.5) / 2,
        gridspec_top=0.9,
        gridspec_bottom=0.1,
        name=f"standard_error_line_chart_all_{stat}",
    )
