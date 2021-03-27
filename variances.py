import copy
import os
import time

import pandas
import numpy
import xarray
import json

from matplotlib import pyplot as plt
import seaborn as sns
from itertools import groupby


import joblib

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
def load_hpo_results(namespace, save_dir):
    with open(f"{save_dir}/hpo_{namespace}.json", "r") as f:
        data = {
            hpo: xarray.Dataset.from_dict(d) for hpo, d in json.loads(f.read()).items()
        }

    return data


@mem.cache
def load_variance_results(namespace, save_dir):
    with open(f"{save_dir}/variance_{namespace}.json", "r") as f:
        data = xarray.Dataset.from_dict(json.loads(f.read()))

    return data


# std = 'mean-normalized'
# std = 'raw'
# XXX: for now we normalize by the random search, but ideally we would
# normalize by the combination of all sources of variance (we need to
# have this, and for this, the computations should be finished)
std = "random_search"

stat_type = "std"
# stat_type = 'var'

# reporting_set = 'valid'
reporting_set = "test"


color_scheme = "muted"
palette = "bright"


colors_strs = """\
#86aec3
#0f68a4
#a2cf7a
#23901c
#eb8a89
#d30a0c
#edaf5f
#ef6f00
#baa2c6
#5a2d8a
#efef89
#a16918""".split(
    "\n"
)


### green-purple
if palette == "green_purple":
    colors_strs = """\
#6d0058
#9f5b8b
#4d005c
#3b1745
#240038
#79108E
#00535a
#00746b
#00945c
#00a950""".split(
        "\n"
    )

###bright-colors- blue-centric
if palette == "bright_blue":
    colors_strs = """\
#0433FF
#7A81FF
#9437FF
#00FA92
#D783FF
#00F900
#FCFA1B
#FC891B
#FDA090
#FC0D1B""".split(
        "\n"
    )

###bright-colors- red-centric
if palette == "bright":
    colors_strs = """\
#FC0D1B
#399F36
#FC891B
y
#FF7E79
#444444
#7A81FF
#9437FF
#0433FF
""".split(
        "\n"
    )


colors = dict(
    zip(
        [
            "Data (bootstrap)",
            "Weights init",
            "Data order",
            "Dropout",
            "Data augment",
            "Numerical noise",
            "Random Search",
            "Noisy Grid Search",
            "Bayes Opt",
        ],
        colors_strs,
    )
)


### special requests ¯\_(ツ)_/¯
colors["Numerical noise"] = "darkgrey"
colors["Data augment"] = "red"
colors["Weights init"] = "#EFBB12"
colors["Data order"] = "#96D800"
colors["Data (bootstrap)"] = "#F75479"
colors["Random Search"] = "#17E2D4"


# pastel
if color_scheme == "pastel":
    colors["Bayes Opt"] = "#6666FF"
    colors["Random Search"] = "#66FFFF"
    colors["Noisy Grid Search"] = "#6633CC"
    colors["Data (bootstrap)"] = "#FF6699"
    colors["Data augment"] = "#FF6633"
    colors["Weights init"] = "#FFCC66"
    colors["Data order"] = "#CCFF66"
    colors["Dropout"] = "#CC9933"


# muted
if color_scheme == "muted":
    colors["Bayes Opt"] = "#3333FF"
    colors["Random Search"] = "#009999"
    colors["Noisy Grid Search"] = "#663399"
    colors["Data (bootstrap)"] = "#CC3366"
    colors["Data augment"] = "#CC0000"
    colors["Weights init"] = "#CC9900"
    colors["Data order"] = "#99CC00"
    colors["Dropout"] = "#CC6600"

### SELFNOTE-AT: remove this paragraph later
colors["Bayes Opt"] = "#3333FF"
colors["Random Search"] = "#009999"
colors["Noisy Grid Search"] = "#663399"
colors["Data (bootstrap)"] = "#CC3366"
colors["Data augment"] = "#CC0000"
colors["Weights init"] = "#CC9900"
colors["Data order"] = "#99CC00"
colors["Dropout"] = "#CC6600"
DEFAULT_COLOR = "#000000"


colors = dict(
    zip(
        [
            "Data (bootstrap)",
            "Weights init",
            "Data order",
            "Dropout",
            "Data augment",
            "Numerical noise",
            "Random Search",
            "Noisy Grid Search",
            "Bayes Opt",
        ],
        colors_strs,
    )
)


DEFAULT_COLOR = "#000000"

LABELS = {
    "bootstrap_seed": "Data (bootstrap)",
    "bootstrapping_seed": "Data (bootstrap)",
    "random_state": "Weights init",
    "init_seed": "Weights init",
    "sampler_seed": "Data order",
    "transform_seed": "Data augment",
    "global_seed": "Dropout",
    "reference": "Numerical noise",
    "vgg": "CIFAR10\nVGG11",
    "": "",
    "bert-sst2": "Glue-SST2\nBERT",
    "bert-rte": "Glue-RTE\nBERT",
    "bio-task2": "MHC\nMLP",
    "segmentation": "PascalVOC\nResNet",
    "logreg": "Log Reg\nBreast Cancer",
    "random_search": "Random Search",
    "random_search-25": "Random Search - 25",
    "random_search-50": "Random Search",
    "random_search-100": "Random Search",
    "random_search-200": "Random Search - 200",
    "noisy_grid_search": "Noisy Grid Search",
    "noisy_grid_search-25": "Noisy Grid Search - 25",
    "noisy_grid_search-50": "Noisy Grid Search",
    "noisy_grid_search-100": "Noisy Grid Search",
    "noisy_grid_search-200": "Noisy Grid Search - 200",
    "bayesopt": "Bayes Opt",
    "bayesopt-25": "Bayes Opt - 25",
    "bayesopt-50": "Bayes Opt",
    "bayesopt-100": "Bayes Opt",
    "bayesopt-200": "Bayes Opt - 200",
}


for key, label in LABELS.items():
    colors[key] = colors.get(label, DEFAULT_COLOR)


case_studies = {
    "vgg": "vgg",
    "segmentation": "segmentation",
    "bert-sst2": "sst2",
    "bert-rte": "rte",
    "bio-task2": "bio-task2",
}
# 'logreg': 'logreg'}


objectives = {
    "vgg": "test_error_rate",
    "segmentation": "test_mean_jaccard_distance",
    "bert-sst2": "test_error_rate",
    "bert-rte": "test_error_rate",
    "logreg": "test_error_rate",
    "bio-task2": "test_aac",
}


def load_data(root):
    var_root = os.path.join(root, "variance")
    hpo_root = os.path.join(root, "hpo")
    simul_root = os.path.join(root, "simul")

    data = {}

    start = time.time()
    for key, name in case_studies.items():
        # print(f"Loading {key}")
        data[key] = {
            "variance": load_variance_results(name + "-var", var_root),
            "hpo": load_hpo_results(name + "-hpo", hpo_root),
            "simul": load_simul_results(name + "-simul", simul_root),
        }

    elapsed_time = time.time() - start
    # print(f"It took {elapsed_time}s to load all data.")

    return data


# Gather data


def cum_argmin(x):
    """Return the indices corresponding to an cumulative minimum
    (numpy.minimum.accumulate)
    """
    minima = numpy.minimum.accumulate(x, axis=0)
    diff = numpy.diff(minima, axis=0)
    jumps = numpy.vstack(numpy.arange(x.shape[0]) for _ in range(x.shape[1])).T
    jumps[1:, :] *= diff != 0
    jumps = numpy.maximum.accumulate(jumps, axis=0)
    return jumps


def get_var(key, data):
    max_epoch = max(data.epoch.values)

    objs = data.sel(epoch=max_epoch)[objectives[key]]
    # NOTE: Because we ran HPOs of 200 trials to fit our surrogate model,
    #       but what we are interested in is HPOs of 100 trials budget.
    n_hpos = len(objs.seed)
    budget = 100
    values = numpy.zeros(n_hpos * 2)
    values[:n_hpos] = objs.sel(order=range(0, budget)).min(dim="order").values
    values[n_hpos:] = objs.sel(order=range(budget, budget * 2)).min(dim="order").values

    return float(values.var())


def get_std(key, data):
    max_epoch = max(data.epoch.values)

    objs = data.sel(epoch=max_epoch)[objectives[key]]
    # NOTE: Because we ran HPOs of 200 trials to fit our surrogate model,
    #       but what we are interested in is HPOs of 100 trials budget.
    n_hpos = len(objs.seed)
    budget = 100
    values = numpy.zeros(n_hpos * 2)
    values[:n_hpos] = objs.sel(order=range(0, budget)).min(dim="order").values
    values[n_hpos:] = objs.sel(order=range(budget, budget * 2)).min(dim="order").values

    return values


def cum_argmin(x):
    """Return the indices corresponding to an cumulative minimum
    (numpy.minimum.accumulate)
    """
    minima = numpy.minimum.accumulate(x)
    diff = numpy.diff(minima)
    jumps = numpy.arange(len(x))
    jumps[1:] *= diff != 0
    jumps = numpy.maximum.accumulate(jumps)
    return jumps


def test_cum(x, valid, test):
    regrets_idx = cum_argmin(x[valid])
    x[valid] = x[valid].to_numpy()[regrets_idx]
    x[test] = x[test].to_numpy()[regrets_idx]
    return x


def load_variances(data):
    start = time.time()
    variances = {}
    for key, case_data in data.items():
        # print(f"Computing variances for {key}")
        case_variances = dict()
        variances[key] = case_variances
        max_epoch = int(case_data["variance"].epoch.max())
        seeds = list(case_data["variance"].noise.values)
        if key == "segmentation":
            seeds.append("reference")
        for seed in seeds:
            replaced_label = seed
            if seed == "random_state":
                replaced_label = "init_seed"
            elif seed == "bootstrap_seed":
                replaced_label = "bootstrapping_seed"
            noise_data = case_data["variance"].sel(seed=seed, epoch=max_epoch)
            case_variances[replaced_label] = noise_data[objectives[key]].values

        case_variances["everything"] = get_std(
            key, case_data["simul"]["random_search"]["ideal"]
        )

        # if "hpo" not in case_data:
        #     continue

        for hpo, hpo_data in case_data["hpo"].items():
            if hpo in ["grid_search", "nudged_grid_search"]:
                continue

            hpo_data = hpo_data.sel(epoch=max_epoch)

            if hpo == "bayesopt":
                seeds = list(
                    hpo_data[objectives[key]].dropna(dim="seed", how="all").seed.values
                )
                # print(len(seeds))
                hpo_data = hpo_data.sel(seed=seeds)
                thresh = min(len(seeds), 15)
                order = list(
                    hpo_data[objectives[key]]
                    .dropna(dim="order", how="all", thresh=thresh)
                    .order.values
                )
                hpo_data = hpo_data.sel(order=order)

            this_data = hpo_data.drop("noise").drop("params")
            metrics = [objectives[key].replace("test", "validation"), objectives[key]]
            this_data = this_data.to_dataframe()
            to_plot = this_data[metrics]
            to_plot = to_plot.reset_index()

            regrets = to_plot.groupby("seed")[metrics].apply(
                test_cum, valid=metrics[0], test=metrics[1]
            )

            test_to_plot = copy.deepcopy(to_plot)
            test_to_plot["regret"] = regrets[metrics[1]]
            test_to_plot = test_to_plot.drop("seed", axis=1)

            n_trials = min(100, test_to_plot.order.max())
            test_regret = test_to_plot.query(f"order == {n_trials}")["regret"]

            case_variances[hpo] = test_regret.to_numpy()

    elapsed_time = time.time() - start
    # print(f"It took {elapsed_time}s to compute variances.")

    return variances


def add_line(ax, xpos, ypos):
    line = plt.Line2D(
        [xpos - 0.4, xpos],
        [ypos, ypos],
        transform=ax.transAxes,
        color="darkslategrey",
        linestyle="--",
    )
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df):
    y_size = 0.4
    x_padding = 0.01
    ypos = -0.05
    scale = 1.0 / df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        if level == 0:
            for label, rpos in label_len(df.index, level):
                add_line(ax, 0, pos * scale)
                pos += rpos

        pos = 0
        for label, rpos in label_len(df.index, level):
            # lxpos = (pos + .5 * rpos)*scale
            lxpos = (pos + rpos) * scale - x_padding
            ax.text(
                ypos,
                lxpos,
                LABELS[label],
                ha="right",
                va="top",
                transform=ax.transAxes,
                color=colors.get(LABELS[label], DEFAULT_COLOR),
            )

            pos += rpos
        # add_line(ax, ypos, pos*scale)
        ypos -= y_size


def get_references(tbl):
    ### do we need to check for multiple occurences?
    if "random_search" in tbl.index:
        random_title = "random_search"
    elif "random_search-25" in tbl.index:
        random_title = "random_search-25"
    elif "random_search-50" in tbl.index:
        random_title = "random_search-50"
    elif "random_search-100" in tbl.index:
        random_title = "random_search-100"
    elif "random_search-200" in tbl.index:
        random_title = "random_search-200"

    if "noisy_grid_search" in tbl.index:
        grid_title = "noisy_grid_search"
    elif "noisy_grid_search-25" in tbl.index:
        grid_title = "noisy_grid_search-25"
    elif "noisy_grid_search-50" in tbl.index:
        grid_title = "noisy_grid_search-50"
    elif "noisy_grid_search-100" in tbl.index:
        grid_title = "noisy_grid_search-100"
    elif "noisy_grid_search-200" in tbl.index:
        grid_title = "noisy_grid_search-200"

    if "bayesopt" in tbl.index:
        bayes_title = "bayesopt"
    elif "bayesopt-25" in tbl.index:
        bayes_title = "bayesopt-25"
    elif "bayesopt-50" in tbl.index:
        bayes_title = "bayesopt-50"
    elif "bayesopt-100" in tbl.index:
        bayes_title = "bayesopt-100"
    elif "bayesopt-200" in tbl.index:
        bayes_title = "bayesopt-200"

    return bayes_title, random_title, grid_title


#### old figure code
# fig = plt.figure(figsize=(7.5, 2))


# axes = fig.subplots(1, len(frame_2d.columns),
#                    sharey=True,
#                    gridspec_kw={'left': .18,
#                        'top': .96, 'right': .999, 'bottom': .12,
#                        'hspace': 0.02, 'wspace': 0.07},
#                    )

# labels = frame_2d.index.to_series().replace(LABELS)

# for name, ax in zip(frame_2d.columns, axes):
#    col = frame_2d[name]
#    ax.barh(range(len(col)), col,
#            color=[colors[k] for k in frame_2d.index])
#    if ax is axes[0]:
#        ax.set_yticks(range(len(labels)))
#        ax.set_yticklabels(labels)
#        sns.despine(ax=ax)
#    else:
#        sns.despine(ax=ax, left=True)
#        ax.get_yaxis().set_visible(False)
#
#    ax.text(.5, .98, name, ha='center', transform=ax.transAxes)
#    #ax.set_facecolor('.95')
#    for i in range(len(labels)):
#        if not i % 2:
#            continue
#        ax.axhspan(i - .5, i + .5, color='.95', zorder=0)
#
# plt.savefig('test_var_2d.png')
# plt.savefig('test_var_2d.pdf')


class VariancesPlot:
    def __init__(self, root):
        self.root = os.path.expanduser("~/Dropbox/Olympus-Data")

    def load(self):
        self.data = load_data(self.root)
        self.variances = load_variances(self.data)
        for task, task_data in self.variances.items():
            for key, data in task_data.items():
                numpy.random.shuffle(data)

    def plot(self, ax):
        pass

    def save(self, name="variances"):
        plt.savefig(f"{name}.png", dpi=300)
        plt.savefig(f"{name}.pdf", dpi=300)


if __name__ == "__main__":
    variances_plot = VariancesPlot()
    variances_plot.load()
    data = variances_plot.data
    variances = variances_plot.variances

    # Convert to panda frames
    rows = []
    c_rows = []
    for key, case_variances in variances.items():
        for noise, variance in sorted(case_variances.items(), key=lambda item: item[1]):
            rows.append([key, noise, variance])
            c_rows.append([key, noise, colors.get(noise, DEFAULT_COLOR)])

    oframe = pandas.DataFrame(rows, columns=["case", "noise", "variance"])
    frame = pandas.DataFrame(rows, columns=["case", "noise", "variance"])
    # Fix a name variation in some experiments
    frame["noise"] = frame["noise"].replace(
        {
            "bootstrap_seed": "bootstrapping_seed",
            "random_state": "init_seed",
        }
    )
    frame.set_index(["case", "noise"], inplace=True)

    color_frame = pandas.DataFrame(c_rows, columns=["case", "noise", "variance"])
    color_frame.set_index(["case", "noise"], inplace=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.barh(
        range(len(oframe)), list(oframe["variance"]), color=[row[-1] for row in c_rows]
    )
    # Below 3 lines remove default labels
    labels = ["" for item in ax.get_yticklabels()]
    ax.set_ylim(-0.5, len(oframe) - 0.5)
    ax.set_yticklabels(labels)
    ax.get_yaxis().set_ticks([])
    ax.set_ylabel("")
    ax.set_xlabel("% of total variance")
    label_group_bar_table(ax, frame)
    fig.subplots_adjust(left=0.2 * frame.index.nlevels)
    plt.savefig("test_var.png")
    plt.savefig("test_var.pdf")

    # A 2D plot
    frame_2d = frame.unstack(level=0).droplevel(level=0, axis="columns")
    # frame_2d['average'] = frame.reset_index(level=1).groupby('noise').mean()
    # frame_2d = frame_2d.sort_values(by='average')

    fig = plt.figure(figsize=(8, 2))

    axes = fig.subplots(
        1,
        len(frame_2d.columns),
        sharey=True,
        gridspec_kw={
            "left": 0.16,
            "top": 0.96,
            "right": 0.999,
            "bottom": 0.12,
            "hspace": 0.02,
            "wspace": 0.07,
        },
    )

    ### getting the reference columns names
    bayes_title, random_title, grid_title = get_references(frame_2d)

    ###creating an empty line to create a separation between the \xi_O and the \xi_H
    empty = numpy.zeros(frame_2d.shape[1])
    empty = pandas.DataFrame(empty).T
    empty.columns = frame_2d.columns
    empty.index = [""]
    frame_2d = pandas.concat([frame_2d, empty])

    ### the lines below generate a random bayes opt variance
    # temp_bayesopt = numpy.abs(numpy.random.normal(0.5,0.25,frame_2d.shape[1]).reshape((1,frame_2d.shape[1])))
    # frame_2d[frame_2d.index == bayes_title] = temp_bayesopt
    frame_2d = frame_2d.reindex(
        [
            "reference",
            "global_seed",
            "init_seed",
            "sampler_seed",
            "transform_seed",
            "bootstrapping_seed",
            "",
            grid_title,
            random_title,
            bayes_title,
        ],
        axis=0,
    )
    labels = frame_2d.index.to_series().replace(LABELS)

    for name, ax in zip(frame_2d.columns, axes):
        col = frame_2d[name]
        ax.barh(range(len(col)), col * 100, color=[colors[k] for k in frame_2d.index])
        if ax is axes[0]:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            sns.despine(ax=ax)
        else:
            sns.despine(ax=ax, left=True)
            ax.get_yaxis().set_visible(False)

        # print(LABELS[name])
        ax.text(0.5, 0.98, LABELS[name], ha="center", transform=ax.transAxes)
        ax.tick_params(axis="both", which="both", length=0)
        # ax.set_facecolor('.95')
        for i in range(len(labels)):
            if not i % 2:
                continue
            ax.axhspan(i - 0.5, i + 0.5, color=".95", zorder=0)

    plt.gcf().text(-0.17, 0.91, r"hyperparameter", fontsize=10)
    plt.gcf().text(-0.17, 0.81, r"optimization", fontsize=10)
    plt.gcf().text(-0.17, 0.71, r"HOpt {$\xi_{H}$}", fontsize=10)
    plt.gcf().text(-0.17, 0.45, r"learning", fontsize=10)
    plt.gcf().text(-0.17, 0.35, r"procedure", fontsize=10)
    plt.gcf().text(-0.17, 0.25, r"{$\xi_{O}$}", fontsize=10)
    plt.gcf().text(-0.17, 1.1, r"source of variation", weight="bold", fontsize=11)
    plt.gcf().text(0.5, 1.1, r"case studies", weight="bold", fontsize=11)
    axes[2].set_xlabel("STD")
    plt.savefig(f"test_var_2d_all_annot_{color_scheme}.png", bbox_inches="tight")
    plt.savefig(f"test_var_2d_all_annot_{color_scheme}.pdf", bbox_inches="tight")
