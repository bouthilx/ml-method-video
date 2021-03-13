import json
import numpy
import scipy.stats
import scipy.special
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import patches
import matplotlib.cm
from utils import adjust_line, plot_line, linear
from moustachos import h_moustachos, adjust_h_moustachos
from scipy.interpolate import UnivariateSpline
from utils import Cache, compute_identity, precision
from tqdm import tqdm

# from arch.bootstrap import IIDBootstrap


tab10 = matplotlib.cm.get_cmap("tab10")
cache = Cache("simulations.json", waittime=60)


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


def jackknife(foo, simulation):
    a = simulation.mu_a[: simulation.sample_size, :]
    b = simulation.mu_b[: simulation.sample_size, :]

    results = numpy.zeros(a.shape)
    for i in range(a.shape[0]):
        idx = list(range(0, i)) + list(range(i, a.shape[0]))
        results[i] = foo(a[idx], b[idx])

    return results


def bootstrap(simulation, foo, n_bootstraps):
    stats = numpy.zeros((n_bootstraps, simulation.simuls))
    # for i in tqdm(range(n_bootstraps), desc="bootstrap"):  # simulation.sample_size):
    for i in range(n_bootstraps):
        idx = numpy.random.randint(
            0, simulation.sample_size, size=simulation.sample_size
        )
        stats[i] = (simulation.mu_a[idx, :] > simulation.mu_b[idx, :]).mean(0)

    return stats


def normal_pab_ci(simulation, foo, alpha=0.05):
    pa = simulation.mu_a[: simulation.sample_size, :]
    pb = simulation.mu_b[: simulation.sample_size, :]

    data = foo(pa, pb)
    ci = scipy.stats.norm.isf(alpha / 2) * numpy.sqrt(
        data * (1 - data) / simulation.sample_size
    )
    lower = numpy.clip(data - ci, a_min=0, a_max=1)
    upper = numpy.clip(data + ci, a_min=0, a_max=1)
    return lower, data, upper


def percentile_bootstrap(simulation, foo, alpha=0.05, bootstraps=None):
    pa = simulation.mu_a[: simulation.sample_size, :]
    pb = simulation.mu_b[: simulation.sample_size, :]

    if bootstraps is None:
        bootstraps = simulation.sample_size

    stats = bootstrap(simulation, foo, bootstraps)

    stats = numpy.sort(stats, axis=0)
    lower = numpy.percentile(stats, alpha / 2 * 100, axis=0)
    upper = numpy.percentile(stats, (1 - alpha / 2) * 100, axis=0)

    return lower, foo(pa, pb), upper


def bca_bootstrap(simulation, foo, alpha=0.05, bootstraps=None):
    pa = simulation.mu_a[: simulation.sample_size, :]
    pb = simulation.mu_b[: simulation.sample_size, :]

    jn = jackknife(foo, simulation)

    ql = scipy.stats.norm.ppf(alpha)
    qu = scipy.stats.norm.ppf(1 - alpha)

    # Acceleration factor
    num = numpy.sum((jn.mean(0) - jn) ** 3, axis=0)
    den = 6 * numpy.sum((jn.mean(0) - jn) ** 2, axis=0) ** 1.5
    ahat = num / den
    # print("num", num)
    # print("den", den)
    # print("ahat", ahat)
    assert ahat.shape == (simulation.simuls,)
    ahat = numpy.nan_to_num(ahat, nan=0)
    # Bias correction factor
    # NOTE: store_theta is a group of bootrtsap
    # here we should compare the bootstraped p_a_b with p_a_b itself
    # again, take care to not compute mean on the simulation axis

    if bootstraps is None:
        bootstraps = simulation.sample_size

    bootstraped = bootstrap(simulation, foo, bootstraps)

    data = foo(pa, pb)

    # print(data.shape)
    # print(bootstraped.shape)
    # print("mean", numpy.mean(bootstraped < data, axis=0))
    zhat = scipy.stats.norm.ppf(numpy.mean(bootstraped < data, axis=0))
    zhat = numpy.nan_to_num(zhat, neginf=-0.9999)
    a1 = scipy.stats.norm.cdf(zhat + (zhat + ql) / (1 - ahat * (zhat + ql)))
    a2 = scipy.stats.norm.cdf(zhat + (zhat + qu) / (1 - ahat * (zhat + qu)))
    # print("boot", bootstraped[:, :5])
    # print("a1", a1)
    # print("a2", a2)
    # print("zhat", zhat)
    try:
        lower = numpy.quantile(bootstraped, a1, axis=0)
        upper = numpy.quantile(bootstraped, a2, axis=0)
    except:
        import pdb

        pdb.set_trace()

    return lower, data, upper


class Simulations:
    def __init__(self, simulations):
        self.simulations = simulations

    @property
    def sample_size(self):
        return next(iter(self.simulations.values())).sample_size

    def set_sample_size(self, sample_size):
        for simulation in self.simulations.values():
            simulation.sample_size = sample_size

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

    def get_hash(self):
        return compute_identity(
            dict(
                name=self.name,
                means=list(map(precision, self.means)),
                stds=list(map(precision, self.stds)),
                bias_stds=list(map(precision, self.bias_stds)),
                sample_size=self.sample_size,
                simuls=self.simuls,
            )
        )

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

    def get_hash(self):
        return compute_identity(
            dict(gamma=precision(self.gamma), alpha=self.alpha, beta=self.beta)
        )

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
        # sample_size = max(sample_size, simulation.sample_size)
        se = numpy.sqrt(sample_size)
        # sample_size = 100
        t = scipy.stats.t.ppf(1 - self.alpha, sample_size - 1)

        diff = simulation.means[0] - simulation.means[1]
        p = scipy.stats.norm.cdf(diff / (simulation.stds[0] * numpy.sqrt(2)))

        return p_threshold(
            t * est_std / se,
            diff,
            # est_std / numpy.sqrt(sample_size))
            numpy.sqrt(
                ((simulation.stds ** 2) / sample_size + simulation.bias_stds ** 2).sum()
            ),
        )


class SimulationScatter:
    def __init__(self, ax, simulation, n_rows, colors=None):
        self.simulation = simulation
        self.sample_size = simulation.sample_size
        self.ys = dict()
        for key in "AB":
            self.ys[key] = numpy.ones(simulation.mu_a.shape) * -100
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

    def simulate(self, i, n_frames, key, lines):
        idx = int(numpy.round(linear(0, self.simulation.sample_size, i, n_frames)))
        if lines[-1] == -1:
            lines.pop(-1)
            assert lines[0] < self.n_rows
            lines += list(range(lines[0], self.n_rows))
        for line in lines:
            line = self.n_rows - line - 1
            self.ys[key][:idx, line] = line
            self.ys[key][idx:, line] = -100
        # TODO: Replace row in _get_row_y() by a matrix of shape
        #       (sample_size, simuls)
        #       When called, move down random points, otherwise they have y=-nrows
        #       so that they are out of figure
        #       We could simply take one point by one, since they should be in random order
        #       Just be careful if simulation was sorted before by SwitchSimulation
        self.update_scatter()

    def decrease_sample_size(self, new_sample_size, i, n_frames, key, lines):
        idx = int(
            numpy.round(
                linear(self.simulation.sample_size, new_sample_size, i, n_frames)
            )
        )
        if lines[-1] == -1:
            lines.pop(-1)
            assert lines[0] < self.n_rows
            lines += list(range(lines[0], self.n_rows))
        for line in lines:
            line = self.n_rows - line - 1
            self.ys[key][idx:, line] = -100
        self.update_scatter()

    def _get_row_y(self, row, model):
        # NOTE: B is above A
        return self.ys[model][:, row] + (0.25 if model == "B" else -0.25)

    def redraw(self):
        self.update_scatter()

    def update_scatter(self):
        scatter_data = []
        for simulation in range(self.n_rows):
            for name in "AB":
                x = getattr(self.simulation, f"mu_{name.lower()}")[
                    : self.sample_size, simulation
                ]
                y = self._get_row_y(simulation, name)
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

            diff = max(
                A[: self.test.sample_size].mean() - B[: self.test.sample_size].mean(),
                1e-5,
            )
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

        self.null_line.set_positon(min_x + 0.5, self.n_rows + 1, text="0.5", pad_y=1)
        self.gamma_line.set_positon(
            min_x + self.test.gamma, self.n_rows + 1, text="$\gamma$", pad_y=1
        )

        for simulation in range(self.n_rows):
            row = self.rows[simulation]
            A = self.simulation.mu_a[:, simulation]
            B = self.simulation.mu_b[:, simulation]

            lower, pab, upper = self.test.get_pab_bounds(
                self.simulation, A[:, None], B[:, None]
            )
            decision = (0.5 < lower) and (self.gamma <= upper)

            if decision:
                row["decision"].set_color(tab10(2))  # green
            else:
                row["decision"].set_color(tab10(3))  # red

            diff = max(A.mean() - B.mean(), 1e-5)
            adjust_h_moustachos(
                row["whisker"],
                x=pab,
                y=self._get_row_y(simulation),
                whisker_width=self.whisker_width,
                whisker_length=(pab - lower, upper - pab),
                center_width=self.whisker_width * 0.5,
            )


class AverageTest:
    PAPERS_WITH_CODE_THRESHOLD = 1.9952

    def __init__(self, gamma, sample_size=None):
        # Hard set to 0.75 because that was the value used to compute the threshold based on
        # paperswithcode data
        self.gamma = 0.75  # gamma
        self.sample_size = sample_size

    def get_hash(self):
        return compute_identity(
            dict(gamma=precision(self.gamma), sample_size=self.sample_size)
        )

    def get_delta(self, simulation):

        # delta = (
        #     -scipy.stats.norm.isf(self.gamma)
        #     * simulation.stds[0]
        #     * numpy.sqrt(2)
        #     * self.PAPERS_WITH_CODE_THRESHOLD
        # )
        # print("name", simulation.name)
        # print("avg", delta)
        # delta = scipy.stats.norm.cdf(delta / (simulation.stds[0] * numpy.sqrt(2)))
        # print("pab", delta)

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
        if self.sample_size is None:
            sample_size = simulation.sample_size
        else:
            sample_size = self.sample_size

        return (mu_a[:sample_size].mean() - mu_b[:sample_size].mean()) > task_delta

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
    def __init__(
        self, gamma, alpha=0.05, beta=0.05, ci_type="bootstrap"
    ):  #  ci_type="bootstrap"):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.ci_type = ci_type

    def get_hash(self):
        return compute_identity(
            dict(
                gamma=precision(self.gamma),
                alpha=self.alpha,
                beta=self.beta,
                ci_type=self.ci_type,
            )
        )

    def single_test(self, simulation, mu_a, mu_b):
        lower, _, upper = self.get_pab_bounds(simulation, mu_a[:, None], mu_b[:, None])
        return (0.5 < lower) and (self.gamma <= upper)

    def get_pab_bounds(self, simulation, mu_a, mu_b):
        def simul_pab(mu_a, mu_b):
            return (mu_a > mu_b).mean(0)

        p_a_b = simul_pab(simulation.mu_a, simulation.mu_b)

        if self.ci_type == "normal":
            lower, p_a_b, upper = normal_pab_ci(simulation, simul_pab, alpha=self.alpha)

        elif self.ci_type == "bootstrap":
            lower, p_a_b, upper = percentile_bootstrap(
                simulation, simul_pab, alpha=self.alpha, bootstraps=50
            )

        elif self.ci_type == "bca_bootstrap":
            lower, p_a_b, upper = bca_bootstrap(
                simulation, simul_pab, alpha=self.alpha, bootstraps=50
            )

        return lower, p_a_b, upper

    def __call__(self, simulation):
        lower, _, upper = self.get_pab_bounds(
            simulation, simulation.mu_a, simulation.mu_b
        )
        # return ((0.5 < lower).mean() * (self.gamma <= upper)).mean()
        return ((0.5 < lower) * (self.gamma <= upper)).mean()


# TODO: Save every tests in a json, so that we can avoid rerunning them
# (simul_name (type, task), test args (gamma, sample_size))


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


def simulate_pab(pab, simulations, comparison_method, tasks=None):
    pab = precision(pab, 3)

    if tasks is None:
        tasks = sorted(simulations.simulations.keys())
    simulations.set_pab(pab)
    detection_rates = numpy.zeros(len(tasks))
    for i, task in enumerate(tasks):
        simulation = simulations.get_task(task)
        detection_rates[i] = cache.compute(simulation, comparison_method)

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
        pabs = numpy.linspace(self.pabs.min(), self.pabs.max(), 50, endpoint=True)
        rates_and_errs = [
            simulate_pab(pab, self.simulations, self.comparison_method) for pab in pabs
        ]
        rates, err = zip(*rates_and_errs)
        self.rates = rates
        self.err = err

        # def smooth(y):
        #     spl = UnivariateSpline(pabs, y)
        #     spl.set_smoothing_factor(0.5)
        #     return spl(self.pabs)

        # self.rates = smooth(rates)
        # self.err = smooth(err)

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
        self.h0_text = ax.text(x, y, "$H_0$", va="center", ha="center", fontsize=18)
        self.h0_lb_text = ax.text(
            x, y - 10, "(lower=better)", va="center", ha="center", fontsize=14
        )

        self.h01_rect = patches.Rectangle(
            (0.5, 0),
            self.gamma - 0.5,
            100,
            linewidth=1,
            edgecolor="gainsboro",
            facecolor="gainsboro",
            zorder=-50,
        )
        ax.add_patch(self.h01_rect)

    def redraw_h01_rect(self):
        self.h01_rect.set_width(self.gamma - 0.5)

    def add_h01(self, ax):
        x = 0.52
        y = 85
        self.not_h0_text = ax.text(x, y, "$H_0$", va="bottom", ha="left", fontsize=18)
        self.not_h0_strike = ax.plot(
            [x, x + 0.025], [y, y + 10], color="black", linewidth=1
        )[0]

        x = 0.56
        self.not_h1_text = ax.text(x, y, "$H_1$", va="bottom", ha="left", fontsize=18)
        self.not_h1_strike = ax.plot(
            [x, x + 0.025], [y, y + 10], color="black", linewidth=1
        )[0]

    def add_h1(self, ax):
        x = self.gamma + 0.07
        y = 55
        self.h1_text = ax.text(x, y, "$H_1$", va="center", ha="center", fontsize=18)
        self.h1_lb_text = ax.text(
            x, y - 10, "(higher=better)", va="center", ha="center", fontsize=14
        )

    def redraw_h1(self):
        x = self.gamma + 0.07
        pos = self.h1_text.get_position()
        self.h1_text.set_position((x, pos[1]))
        pos = self.h1_lb_text.get_position()
        self.h1_lb_text.set_position((x, pos[1]))

    def add_oracle_annotation(self, ax, pabs, y):
        return

        # NOTE: y is based on oracle data.
        x = 0.63
        x_65_idx = numpy.searchsorted(pabs, x)
        # ax.scatter([x], [y[x_65_idx]], s=5, marker="+")
        # self.optimal_oracle_text = ax.annotate(
        self.optimal_oracle_text = matplotlib.text.Annotation(
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
        ax.add_artist(self.optimal_oracle_text)

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

    def set_sample_size(self, sample_size):
        self.sample_size = sample_size
        for simulation in self.simulations.values():
            simulation.set_sample_size(sample_size)

    def set_gamma(self, gamma):
        self.gamma = gamma
        self.redraw_h01_rect()
        self.redraw_h1()
        for name in ["oracle", "ideal-pab", "biased-pab"]:
            curve = self.curves[name]
            curve.comparison_method.gamma = gamma
            curve.compute()
            curve.redraw()

        # ax = self.optimal_oracle_text.axes
        # self.optimal_oracle_text.remove()
        # self.add_oracle_annotation(
        #     ax, self.curves["oracle"].pabs, self.curves["oracle"].rates
        # )

    def build_curves(self, ax):
        self.curves = {}
        print("Creating oracle curve")
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
        print("Creating pab curve")
        self.curves["ideal-pab"] = Curve(
            ax,
            self.simulations["ideal"],
            PABTest(gamma=self.gamma, **self.pab_kwargs),
            n_points=self.n_points,
            color=self.colors["pab"],
            linestyle="--",
        )
        print("Creating pab curve")
        self.curves["biased-pab"] = Curve(
            ax,
            self.simulations["biased"],
            PABTest(gamma=self.gamma, **self.pab_kwargs),
            n_points=self.n_points,
            color=self.colors["pab"],
            linestyle="-",
        )
        print("done")

    def redraw(self):
        for curve in self.curves.values():
            curve.compute()
            curve.redraw()
