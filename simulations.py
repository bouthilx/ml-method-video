import json
import numpy
import scipy.stats
import scipy.special
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import patches
import matplotlib.cm
from utils import adjust_line, plot_line
from moustachos import h_moustachos, adjust_h_moustachos


tab10 = matplotlib.cm.get_cmap("tab10")


def p_threshold(x, mu, sigma):
    # return scipy.stats.norm.isf((x - mu) / sigma)
    return 1 - scipy.stats.norm.cdf((x - mu) / sigma)
    # return 1 - 0.5 * (1 + scipy.special.erf((x - mu) / (sigma * numpy.sqrt(2))))


def quantile(p):
    return numpy.sqrt(2) * scipy.special.erfinv(2 * p - 1)


def pab(pa, pb):
    return (pa > pb).mean()


def normal_ci(pa, pb, sample_size=None, alpha=0.05):
    if sample_size is None:
        sample_size = pa.shape[0]
    p_a_b = pab(pa, pb)
    return scipy.stats.norm.isf(alpha / 2) * numpy.sqrt(
        p_a_b * (1 - p_a_b) / sample_size
    )


def percentile_bootstrap(pa, pb, alpha=0.05, bootstraps=None):
    if len(pa.shape) < 2:
        pa = pa.reshape((-1, 1))
        pb = pb.reshape((-1, 1))

    sample_size = pa.shape[0]
    simuls = pa.shape[1]

    if bootstraps is None:
        bootstraps = sample_size

    stats = numpy.zeros((bootstraps, simuls))
    for i in range(bootstraps):
        idx = numpy.random.randint(0, sample_size, size=sample_size)
        stats[i] = (pa[idx, :] > pb[idx, :]).mean(0)

    stats = numpy.sort(stats, axis=0)
    lower = numpy.percentile(stats, alpha / 2 * 100, axis=0)
    upper = numpy.percentile(stats, (1 - alpha / 2) * 100, axis=0)

    return lower, upper


class Simulations:
    def __init__(self, simulations):
        self.simulations = simulations

    def simulate(self):
        for simulation in self.simulations.values():
            simulation.simulate()

    def set_pab(self, pab):
        for simulation in self.simulations.values():
            simulation.set_pab(pab)

    def get_task(self, task):
        return self.simulations[task]


def load_stats():
    with open("stats.json", "r") as f:
        stats = json.load(f)

    return stats


# TODO: Will need simulations for all, for ideal-rte and fixhoptall-rte
def create_simulations(tasks, estimator, sample_size, simuls=10000, pab=0.75):
    simulations = {}
    for task in tasks:
        stds = numpy.ones(2) * stats[key]["sigma"]
        bias_stds = numpy.ones(2) * stats[task]["sigma_epsilon"]["10"][estimator]
        simulations[task] = Simulation(task, pab, stds, bias_stds, sample_size, simuls)

    return Simulations(simulations)


class Simulation:
    def __init__(self, name, pab, stds, bias_stds, sample_size, simuls):
        self.name = name
        self.means = numpy.zeros(2)
        self.bias_stds = bias_stds  # This
        self.stds = stds
        self.sample_size = sample_size
        self.simuls = simuls
        self.set_pab(pab)
        self.simulate()

    def simulate(self):
        # TODO: Need to verify is we simulate A > B or B > A
        mu_a_p = numpy.random.normal(self.means[0], self.bias_stds[0], size=self.simuls)
        mu_b_p = numpy.random.normal(self.means[1], self.bias_stds[1], size=self.simuls)
        self.mu_a = mu_a_p[None, :] + self.stds[0] * numpy.random.normal(
            0, 1, size=(self.sample_size, self.simuls)
        )
        self.mu_b = mu_b_p[None, :] + self.stds[1] * numpy.random.normal(
            0, 1, size=(self.sample_size, self.simuls)
        )

    def get_rho(self):
        # Standardize data, for a standardized rho
        a = copy.deepcopy(self.mu_a)
        a -= a.mean()
        a /= a.std()
        var_r = a.var(0).mean()  # mean variance of R
        var_tm = a.mean(0).var()  # variance of mu
        k = a.shape[0]  # should be sample_size
        assert k == self.sample_size
        return (k * var_tmu - var_r) / ((k - 1) * var_r)

    def set_pab(self, pab):
        pab = min(pab, 1 - 1e-3)
        self.pab = pab
        mean = scipy.stats.norm.isf(pab) * self.stds[0] * numpy.sqrt(2)
        # NOTE: mean is negative, so we flip the signs and get A > B
        # TODO: Why should we use mean * 2 as a diff? Oracle does not follow the same pattern...
        means = numpy.array([-mean, mean]) / 2
        self.adjust_simulation(means)

    def adjust_simulation(self, means):
        if hasattr(self, "mu_a"):
            self.mu_a -= self.means[0]
            self.mu_a += means[0]
            self.mu_b -= self.means[1]
            self.mu_b += means[1]

        self.means = means


class PAB:
    def __init__(self, simulation):
        self.simulation
        pass

    def simulate(self):
        pass

    def ajdust_simulation(self):
        pass


class TTest:
    PAPERS_WITH_CODE_THRESHOLD = 1.9952

    def __init__(self, gamma, alpha=0.05, beta=0.05):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def __call__(self, simulation):
        # This is to convert the gamma into a threshold in original distribution
        task_delta = (
            -scipy.stats.norm.isf(self.gamma) * simulation.stds[0] * numpy.sqrt(2)
        )

        est_std = numpy.sqrt((simulation.stds ** 2).sum())
        sample_size = (
            (quantile(1 - self.alpha) - quantile(self.beta)) * est_std / task_delta
        ) ** 2
        sample_size = max(int(numpy.ceil(sample_size)), 3)
        se = numpy.sqrt(sample_size)
        # sample_size = 100
        t = scipy.stats.t.ppf(1 - self.alpha, simulation.sample_size - 1)

        diff = simulation.means[0] - simulation.means[1]
        p = scipy.stats.norm.cdf(diff / (simulation.stds[0] * numpy.sqrt(2)))

        return p_threshold(
            t * est_std / se,
            diff,
            # est_std / numpy.sqrt(sample_size))
            numpy.sqrt(
                (
                    (simulation.stds ** 2) / simulation.sample_size
                    + simulation.bias_stds ** 2
                ).sum()
            ),
        )


class SimulationScatter:
    def __init__(self, ax, simulation, n_rows, colors=None):
        self.simulation = simulation
        self.n_rows = n_rows
        self.scatters = {}
        for name in "AB":
            self.scatters[name] = {}
            for simulation in range(self.n_rows):
                self.scatters[name][simulation] = ax.scatter(
                    [],
                    [],
                    alpha=0.5,
                    marker="|",
                    # s=1,
                    clip_on=False,
                    color=colors[name] if colors else None,
                )

    def _get_row_y(self, row, model):
        # NOTE: B is above A
        return row + (0.25 if model == "B" else -0.25)

    def redraw(self):
        self.update_scatter()

    def update_scatter(self):
        scatter_data = []
        for simulation in range(self.n_rows):
            for name in "AB":
                x = getattr(self.simulation, f"mu_{name.lower()}")[:, simulation]
                y = self._get_row_y(simulation, name)
                y = [y] * len(x)
                self.scatters[name][simulation].set_offsets(list(zip(x, y)))


class AverageTestViz:
    def __init__(self, ax, simulation, n_rows, test):
        self.ax = ax
        self.simulation = simulation
        self.n_rows = n_rows
        self.test = test
        self.whisker_width = 0.25
        self.rows = []
        self.delta_label = ax.text(0, 0, "", fontsize=14, clip_on=False)
        self.delta_line = ax.plot([], [], color="black", linestyle="--", clip_on=False)[
            0
        ]
        for simulation in range(n_rows):
            row = {}
            row["decision"] = ax.scatter(
                [0], [self._get_row_y(simulation)], marker="s", clip_on=False
            )
            row["whisker"] = h_moustachos(
                ax,
                x=0,
                y=0,
                whisker_width=0.01,
                whisker_length=0.01,
                center_width=0,
                clip_on=False,
            )

            self.rows.append(row)

    def _get_row_y(self, row):
        return row

    def redraw(self):

        display_coords = self.ax.transAxes.transform((0.05, 0))
        data_coords = self.ax.transData.inverted().transform(display_coords)
        min_x = data_coords[0]

        delta = self.test.get_delta(self.simulation)

        x_delta = min_x + delta
        self.delta_label.set_position((x_delta, self.n_rows + 2))
        self.delta_label.set_text("$\delta$")
        self.delta_line.set_xdata([x_delta, x_delta])
        self.delta_line.set_ydata([0, self.n_rows + 1])

        for simulation in range(self.n_rows):
            row = self.rows[simulation]
            A = self.simulation.mu_a[:, simulation]
            B = self.simulation.mu_b[:, simulation]

            decision = self.test.single_test(self.simulation, A, B)
            if decision:
                row["decision"].set_color(tab10(2))  # green
            else:
                row["decision"].set_color(tab10(3))  # red

            diff = max(A.mean() - B.mean(), 1e-5)
            adjust_h_moustachos(
                row["whisker"],
                x=min_x + diff / 2,
                y=self._get_row_y(simulation),
                whisker_width=self.whisker_width,
                whisker_length=diff / 2,
                center_width=0,
            )


class PABTestViz:
    def __init__(self, ax, simulation, n_rows, test):
        self.ax = ax
        self.simulation = simulation
        self.n_rows = n_rows
        self.test = test
        self.whisker_width = 0.25
        self.rows = []
        self.null_line = VLineLabel(ax, fontsize=14)
        self.gamma_line = VLineLabel(ax, fontsize=14)
        for simulation in range(n_rows):
            row = {}
            row["decision"] = ax.scatter(
                [0], [self._get_row_y(simulation)], marker="s", clip_on=False
            )
            row["whisker"] = h_moustachos(
                ax,
                x=0,
                y=0,
                whisker_width=0.01,
                whisker_length=0.01,
                center_width=0,
                clip_on=False,
            )

            self.rows.append(row)

    def _get_row_y(self, row):
        return row

    def redraw(self):

        display_coords = self.ax.transAxes.transform((0.05, 0))
        data_coords = self.ax.transData.inverted().transform(display_coords)
        min_x = data_coords[0]

        delta = self.test.get_delta(self.simulation)

        x_delta = min_x + delta
        self.delta_label.set_position((x_delta, self.n_rows + 2))
        self.delta_label.set_text("$\delta$")
        self.delta_line.set_xdata([x_delta, x_delta])
        self.delta_line.set_ydata([0, self.n_rows + 1])

        for simulation in range(self.n_rows):
            row = self.rows[simulation]
            A = self.simulation.mu_a[:, simulation]
            B = self.simulation.mu_b[:, simulation]

            decision = self.test.single_test(self.simulation, A, B)
            if decision:
                row["decision"].set_color(tab10(2))  # green
            else:
                row["decision"].set_color(tab10(3))  # red

            diff = max(A.mean() - B.mean(), 1e-5)
            adjust_h_moustachos(
                row["whisker"],
                x=min_x + diff / 2,
                y=self._get_row_y(simulation),
                whisker_width=self.whisker_width,
                whisker_length=diff / 2,
                center_width=0,
            )


class AverageTest:
    PAPERS_WITH_CODE_THRESHOLD = 1.9952

    def __init__(self, gamma, sample_size=None):
        self.gamma = gamma
        self.sample_size = sample_size

    def get_delta(self, simulation):

        # This is to convert the gamma into a threshold in original distribution
        # TODO: Where is this linear regression?
        # Then we scale based on common increment in tasks available on paperswithcode
        return (
            -scipy.stats.norm.isf(self.gamma)
            * simulation.stds[0]
            * numpy.sqrt(2)
            * self.PAPERS_WITH_CODE_THRESHOLD
        )

    def single_test(self, simulation, mu_a, mu_b):
        task_delta = self.get_delta(simulation)
        r = (mu_a.mean() - mu_b.mean()) > task_delta
        return (mu_a.mean() - mu_b.mean()) > task_delta

    def __call__(self, simulation):
        task_delta = self.get_delta(simulation)
        if self.sample_size is None:
            sample_size = simulation.sample_size
        else:
            sample_size = self.sample_size
        return p_threshold(
            task_delta,
            simulation.means[0] - simulation.means[1],
            numpy.sqrt(
                ((simulation.stds) ** 2 / sample_size + simulation.bias_stds ** 2).sum()
            ),
        )


class PABTest:
    def __init__(self, gamma, alpha=0.05, beta=0.05, ci_type="bootstrap"):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.ci_type = ci_type

    def __call__(self, simulation):
        p_a_b = (simulation.mu_a > simulation.mu_b).mean(0)

        if False:  # self.ci_type == "normal":
            ci = scipy.stats.norm.isf(self.alpha / 2) * numpy.sqrt(
                p_a_b * (1 - p_a_b) / simulation.sample_size
            )
            lower = max(p_a_b - ci, 0)
            upper = min(p_a_b + ci, 1)
        elif self.ci_type == "bootstrap":
            stats = numpy.zeros((simulation.sample_size, simulation.simuls))
            for i in range(simulation.sample_size):
                idx = numpy.random.randint(
                    0, simulation.sample_size, size=simulation.sample_size
                )
                stats[i] = (simulation.mu_a[idx, :] > simulation.mu_b[idx, :]).mean(0)
            stats = numpy.sort(stats, axis=0)
            lower = numpy.percentile(stats, self.alpha / 2 * 100, axis=0)
            upper = numpy.percentile(stats, (1 - self.alpha / 2) * 100, axis=0)

        return ((0.5 < lower).mean() * (self.gamma <= upper)).mean()


def compute_error_rates_theta(pab, delta_p, method, noise_source, *args):
    error_rates = numpy.zeros(len(stats))
    for i, key in enumerate(stats):
        # j=0 is false positives
        # j=1 is false negatives
        original_sigma = numpy.ones(2) * stats[key]["sigma"]
        # TODO: Set index '20' back to '100' when segmentation simul is fully fetched.
        if noise_source != "ideal":
            epsilon_sigma = (
                numpy.ones(2) * stats[key]["sigma_epsilon"]["10"][noise_source]
            )
        else:
            epsilon_sigma = numpy.zeros(2)

        mean = scipy.stats.norm.isf(pab) * stats[key]["sigma"] * numpy.sqrt(2)
        task_delta_p = (
            -scipy.stats.norm.isf(delta_p) * stats[key]["sigma"] * numpy.sqrt(2)
        )

        error_rates[i] = method(
            numpy.array([0.0, mean]),
            original_sigma,
            epsilon_sigma,
            *args,
            delta=task_delta_p,
        )

    return error_rates


def compute_error_rates():
    pass


def comparison(noise_source):

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    axes = fig.subplots(ncols=1, nrows=1)
    # axes = numpy.array([[axes]])

    sns.despine(ax=axes)

    methods = [
        ("optimal", p_unpaired_t_test2, "ideal"),
        ("single_point", p_unpaired_single_point, "biased"),
        ("average", p_unpaired_averages2, "ideal"),
        ("average", p_unpaired_averages2, "biased"),
        ("ratio", p_unpaired_ratio, "ideal"),
        ("ratio", p_unpaired_ratio, "biased"),
        # ('t-test', p_unpaired_t_test2)
    ]

    # axes[-1, 0].set_xlabel('True differences'
    nx = 30

    colors = "#86aec3 #0f68a4 #a2cf7a #23901c #eb8a89 #d30a0c".split(" ")
    colors = dict(
        optimal="#377eb8", single_point="#4daf4a", average="#984ea3", ratio="#ff7f00"
    )

    # import sys
    # sys.exit(0)

    sample_size = 50
    delta_p = 0.75
    deltas = numpy.linspace(0.4, 1, nx)
    simulations = {}

    for name, method, noise_source in tqdm(methods, leave=False, desc="method"):

        y = []
        e = []
        for delta in tqdm(deltas, leave=True, desc="delta"):
            if name == "optimal":
                error_rates = compute_error_rates_theta(
                    delta, delta_p, method, "ideal", None
                )
            elif name == "ratio":
                error_rates = compute_error_rates_theta(
                    delta, delta_p, method, noise_source, sample_size, "normal"
                )
            else:
                error_rates = compute_error_rates_theta(
                    delta, delta_p, method, noise_source, sample_size
                )
            y.append(error_rates.mean() * 100)
            e.append(error_rates.std() * 100)

        if noise_source == "ideal" and name != "optimal":
            linestyle = "--"
        else:
            linestyle = "-"

        if name == "optimal" and noise_source == "ideal":
            x = 0.65
            x_65_idx = int((numpy.array(deltas) <= x).sum()) - 1
            axes.annotate(
                "Optimal\noracle",
                xy=(x, y[x_65_idx]),
                xycoords="data",
                xytext=(0.56, 60),
                textcoords="data",
                arrowprops=dict(arrowstyle="-|>", facecolor="black"),
                horizontalalignment="center",
                verticalalignment="center",
            )

        plot_line(
            axes, deltas, y, e, label=name, color=colors[name], linestyle=linestyle
        )

    # legend = axes.legend(bbox_to_anchor=(0., 1., 1.0, .102), loc='lower left',
    #                      ncol=len(methods), mode="expand", borderaxespad=0., frameon=False)

    x = 0.11
    y = 0.95
    axes.text(x, y, "IdealEst", transform=fig.transFigure)
    axes.text(x + 0.12, y, "FixHOptEst", transform=fig.transFigure)
    axes.text(x + 0.3, y, "Comparison Methods", transform=fig.transFigure)

    pad = 0.3
    y -= 0.065 + 0.03
    axes.text(x + pad, y, "Single point comparison", transform=fig.transFigure)
    y -= 0.12
    axes.text(
        x + pad,
        y,
        "Average comparison thresholded based\non typical published improvement",
        transform=fig.transFigure,
    )
    y -= 0.06
    axes.text(
        x + pad,
        y,
        "Proposed testing probability of improvement",
        transform=fig.transFigure,
    )

    custom_lines = [
        Line2D([0], [0], color="white"),  # sp ideal
        Line2D([0], [0], color=colors["average"], lw=1.5, linestyle="--"),  # avg ideal
        Line2D([0], [0], color=colors["ratio"], lw=1.5, linestyle="--"),  # p_ab ideal
        Line2D(
            [0], [0], color=colors["single_point"], lw=1.5, linestyle="-"
        ),  # avg biased
        Line2D([0], [0], color=colors["average"], lw=1.5, linestyle="-"),  # avg biased
        Line2D([0], [0], color=colors["ratio"], lw=1.5, linestyle="-"),  # p_ab biased
    ]

    labels = [
        "",  # sp ideal
        "\n",  # avg ideal
        "",  # p_ab ideal
        "",  # sp biased
        "\n",  # avg biased
        "",  # p_ab biased
    ]

    # custom_lines = [Line2D([0], [0], color='white'), Line2D([0], [0], color='white')]
    # labels = ['IdealEst', 'BiasedEst']
    # for method in ['single_point', 'average', 'ratio']:
    #     for noise_source in ['ideal', 'biased']:
    #         if noise_source == 'ideal' and method == 'single_point':
    #             line = Line2D([0], [0], color='white')
    #             label = ''
    #         elif noise_source == 'ideal':
    #             line = Line2D([0], [0], color=colors[method], lw=1.5, linestyle='--')
    #             label = method
    #         else:
    #             line = Line2D([0], [0], color=colors[method], lw=1.5, linestyle='-')
    #             label = method
    #         custom_lines.append(line)
    #         labels.append(label)

    legend = axes.legend(
        custom_lines,
        labels,
        bbox_to_anchor=(0.0, 1, 0.28, 0.102),
        loc="lower left",
        # handletextpad=0.,
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    # axes.add_artist(legend)

    # colors = [
    #     '#4cc04c',
    #     '#1f77ba']
    #'#86564b', '#e377c2', '#d62728']

    # axes[0].barh(y_labels, yn, xerr=en, color=colors[:len(methods)], error_kw=dict(linewidth=0.8))
    # axes[1].barh(y_labels, yp, xerr=ep, color=colors[:len(methods)], error_kw=dict(linewidth=0.8))

    # axes[0].set_title('False Positives\n$\mu_a=\mu_b$')
    # axes[1].set_title('False Negatives\n$\mu_a-\mu_b>\delta$')
    # axes[0].text(-1, 0.65, 'BiasedEst()', transform=axes[0].transAxes)
    # axes[0].text(-1, 0.20, 'IdealEst()', transform=axes[0].transAxes)

    # axes[0].annotate(
    #     'SDL', xy=(-0.5, 0.65), xytext=(-0.6, 0.65), xycoords='axes fraction',
    #      fontsize=14*1.5, ha='center', va='bottom',
    #      bbox=dict(boxstyle='square', fc='white'),
    #      arrowprops=dict(arrowstyle='-[, widthB=5.0, lengthB=1.5', lw=2.0))

    # axes[0].set_xlabel('')

    # axes[-1, -1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    patch = patches.Rectangle(
        (0.5, 0),
        delta_p - 0.5,
        100,
        linewidth=1,
        edgecolor="gainsboro",
        facecolor="gainsboro",
        zorder=-50,
    )
    axes.add_patch(patch)

    x = 0.435
    y = 50
    axes.text(x, y, "$H_0$", va="center", ha="center")
    axes.text(x, y - 10, "(lower=better)", va="center", ha="center", fontsize=6)

    x = 0.81
    y = 55
    axes.text(x, y, "$H_1$", va="center", ha="center")
    axes.text(x, y - 10, "(higher=better)", va="center", ha="center", fontsize=6)

    x = 0.52
    y = 85
    axes.text(x, y, "$H_0$", va="bottom", ha="left")
    axes.plot([x, x + 0.025], [y, y + 10], color="black", linewidth=1)
    x = 0.56
    axes.text(x, y, "$H_1$", va="bottom", ha="left")
    axes.plot([x, x + 0.025], [y, y + 10], color="black", linewidth=1)

    axes.set_ylabel("Rate of Detections", fontsize=24)
    axes.set_xlabel("P(A > B)", fontsize=24)

    # plt.show()
    fig.set_size_inches(WIDTH, HEIGHT)
    plt.savefig(f"error_rates_v2_methods_recommand_{noise_source}.png", dpi=300)
    plt.savefig(f"error_rates_v2_methods_recommand_{noise_source}.pdf", dpi=300)


def simulate_pab(pab, simulations, comparison_method, tasks=None):
    if tasks is None:
        tasks = sorted(simulations.simulations.keys())
    simulations.set_pab(pab)
    detection_rates = numpy.zeros(len(tasks))
    for i, task in enumerate(tasks):
        simulation = simulations.get_task(task)
        detection_rates[i] = comparison_method(simulation)

    detection_rates *= 100

    return detection_rates.mean(), detection_rates.std()


class Curve:
    def __init__(
        self,
        ax,
        simulations,
        comparison_method,
        min_pab=0.4,
        max_pab=1,
        n_points=100,
        color=None,
        linestyle=None,
    ):
        # TODO: Compute PAB in block based on simulation and comparison method
        #       Compute PAB on single pab outsite of curve, for the simulations
        #          in the comparison column
        self.ax = ax
        self.simulations = simulations
        self.comparison_method = comparison_method
        self.pabs = numpy.linspace(min_pab, max_pab, n_points, endpoint=True)
        self.color = color
        self.linestyle = linestyle
        self.pab = min_pab
        self.compute()
        self.draw()

    def compute(self):
        rates_and_errs = [
            simulate_pab(pab, self.simulations, self.comparison_method)
            for pab in self.pabs
        ]
        self.rates, self.err = zip(*rates_and_errs)

    def draw(self):
        index = numpy.searchsorted(self.pabs, self.pab) + 1
        index = max(index, 1)
        x = self.pabs[:index]
        y = self.rates[:index]
        err = self.err[:index]
        self.plots = plot_line(
            self.ax,
            x,
            y,
            err,
            alpha=0.5,
            color=self.color,
            linestyle=self.linestyle,
            min_y=0,
            max_y=100,
        )

    def redraw(self):
        index = numpy.searchsorted(self.pabs, self.pab) + 1
        index = max(index, 1)
        x = self.pabs[:index]
        y = self.rates[:index]
        err = self.err[:index]
        adjust_line(
            self.ax,
            self.plots,
            x,
            y,
            err,
            alpha=0.5,
            color=self.color,
            linestyle=self.linestyle,
            min_y=0,
            max_y=100,
        )

    def set_pab(self, pab):
        self.pab = pab
        self.redraw()


class SimulationBuilder:
    def load(self):
        with open("stats.json", "r") as f:
            self.stats = json.load(f)

    def create_simulations(self, tasks, estimator, sample_size, simuls=10000, pab=0.75):
        simulations = {}
        for task in tasks:
            stds = numpy.ones(2) * self.stats[task]["sigma"]

            if estimator == "ideal":
                bias_std = 0
            else:
                bias_std = self.stats[task]["sigma_epsilon"]["10"][estimator]

            bias_stds = numpy.ones(2) * bias_std

            simulations[task] = Simulation(
                task, pab, stds, bias_stds, sample_size, simuls
            )

        return Simulations(simulations)


class SimulationPlot:
    def __init__(self, gamma, sample_size, n_points, simuls, pab_kwargs={}):
        # NOTE: pab is used to determine where grey area ends, where H1 becomes true
        self.gamma = gamma
        self.sample_size = sample_size
        self.n_points = n_points
        self.simuls = simuls
        self.pab_kwargs = pab_kwargs
        self.simulation_builder = SimulationBuilder()
        self.simulation_builder.load()
        self.tasks = sorted(self.simulation_builder.stats)

        self.colors = dict(
            oracle="#377eb8",
            single="#4daf4a",
            average="#984ea3",
            pab="#ff7f00",
        )

    def format_ax(self, ax):
        ax.set_ylabel("Rate of Detections", fontsize=24)
        ax.set_xlabel("P(A > B)", fontsize=24)
        ax.xaxis.set_label_coords(1.17, -0.025)

        ax.set_xlim(0.37, 1.02)
        ax.set_ylim(0, 100)

        sns.despine(ax=ax)

    def add_h0(self, ax):

        x = 0.435
        y = 50
        ax.text(x, y, "$H_0$", va="center", ha="center", fontsize=18)
        ax.text(x, y - 10, "(lower=better)", va="center", ha="center", fontsize=14)

        patch = patches.Rectangle(
            (0.5, 0),
            self.gamma - 0.5,
            100,
            linewidth=1,
            edgecolor="gainsboro",
            facecolor="gainsboro",
            zorder=-50,
        )
        ax.add_patch(patch)

    def add_h01(self, ax):
        x = 0.52
        y = 85
        ax.text(x, y, "$H_0$", va="bottom", ha="left", fontsize=18)
        ax.plot([x, x + 0.025], [y, y + 10], color="black", linewidth=1)

        x = 0.56
        ax.text(x, y, "$H_1$", va="bottom", ha="left", fontsize=18)
        ax.plot([x, x + 0.025], [y, y + 10], color="black", linewidth=1)

    def add_h1(self, ax):
        x = 0.82
        y = 55
        ax.text(x, y, "$H_1$", va="center", ha="center", fontsize=18)
        ax.text(x, y - 10, "(higher=better)", va="center", ha="center", fontsize=14)

    def add_oracle_annotation(self, ax, pabs, y):

        # NOTE: y is based on oracle data.
        x = 0.63
        x_65_idx = numpy.searchsorted(pabs, x)
        # ax.scatter([x], [y[x_65_idx]], s=5, marker="+")
        ax.annotate(
            "Optimal\noracle",
            xy=(x, y[x_65_idx]),
            xycoords="data",
            xytext=(0.56, 60),
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", facecolor="black", linewidth=1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
        )

    def add_legend(self, ax, fontsize=18):

        x = 1.09
        y = 100
        ax.text(x, y, "IdealEst", va="top", clip_on=False, fontsize=fontsize)
        x_pad = 0.11
        ax.text(x + x_pad, y, "FixHOptEst", va="top", clip_on=False, fontsize=fontsize)
        x_pad += 0.15
        ax.text(
            x + x_pad,
            y,
            "Comparison Methods",
            va="top",
            clip_on=False,
            fontsize=fontsize,
        )

        y_pad = 3
        y_height = 15
        y -= y_height + y_pad
        ax.text(
            x + x_pad,
            y,
            "Single point",
            va="top",
            clip_on=False,
            fontsize=fontsize - 2,
        )
        y -= y_height + y_pad
        ax.text(
            x + x_pad,
            y,
            "Average",
            # "Average comparison thresholded based\non typical published improvement",
            va="top",
            fontsize=fontsize - 2,
        )
        y -= y_height + y_pad
        ax.text(
            x + x_pad,
            y,
            "Probability of outperforming",
            # "Proposed testing probability of improvement",
            va="top",
            fontsize=fontsize - 2,
        )

        linewidth = 4
        custom_lines = [
            Line2D(
                [0], [0], color=self.colors["single"], lw=linewidth, linestyle="--"
            ),  # single ideal
            Line2D(
                [0], [0], color=self.colors["average"], lw=linewidth, linestyle="--"
            ),  # avg ideal
            Line2D(
                [0], [0], color=self.colors["pab"], lw=linewidth, linestyle="--"
            ),  # p_ab ideal
            Line2D([0], [0], color="white"),  # biased, no single
            Line2D(
                [0], [0], color=self.colors["average"], lw=linewidth, linestyle="-"
            ),  # avg biased
            Line2D(
                [0], [0], color=self.colors["pab"], lw=linewidth, linestyle="-"
            ),  # p_ab biased
        ]

        labels = [
            "",  # sp ideal
            "",  # avg ideal
            "",  # p_ab ideal
            "",  # sp biased
            "",  # avg biased
            "",  # p_ab biased
        ]

        legend = ax.legend(
            custom_lines,
            labels,
            bbox_to_anchor=(1.13, 0.37, 0.31, 0.5),
            loc="lower left",
            labelspacing=2.5,
            # handletextpad=0.,
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
            handlelength=4,
            frameon=False,
        )

    def build_simulations(self):
        self.simulations = {}
        self.simulations["ideal"] = self.simulation_builder.create_simulations(
            self.tasks, "ideal", self.sample_size, self.simuls, pab=0.4
        )
        self.simulations["biased"] = self.simulation_builder.create_simulations(
            self.tasks, "biased", self.sample_size, self.simuls, pab=0.4
        )

    def build_curves(self, ax):
        self.curves = {}
        self.curves["oracle"] = Curve(
            ax,
            self.simulations["ideal"],
            TTest(gamma=self.gamma),
            n_points=self.n_points,
            color=self.colors["oracle"],
            linestyle="-",
        )
        self.curves["single"] = Curve(
            ax,
            self.simulations["ideal"],
            AverageTest(gamma=self.gamma, sample_size=1),
            n_points=self.n_points,
            color=self.colors["single"],
            linestyle="--",
        )
        self.curves["ideal-avg"] = Curve(
            ax,
            self.simulations["ideal"],
            AverageTest(gamma=self.gamma),
            n_points=self.n_points,
            color=self.colors["average"],
            linestyle="--",
        )
        self.curves["biased-avg"] = Curve(
            ax,
            self.simulations["biased"],
            AverageTest(gamma=self.gamma),
            n_points=self.n_points,
            color=self.colors["average"],
            linestyle="-",
        )
        self.curves["ideal-pab"] = Curve(
            ax,
            self.simulations["ideal"],
            PABTest(gamma=self.gamma, **self.pab_kwargs),
            n_points=self.n_points,
            color=self.colors["pab"],
            linestyle="--",
        )
        self.curves["biased-pab"] = Curve(
            ax,
            self.simulations["biased"],
            PABTest(gamma=self.gamma, **self.pab_kwargs),
            n_points=self.n_points,
            color=self.colors["pab"],
            linestyle="-",
        )
