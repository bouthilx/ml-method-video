from matplotlib import pyplot as plt

from utils import ZOrder


FPS = 60

zorder = ZOrder()


class Animation:
    def __init__(self, plots, width, height, start=0, end=-1, fps=FPS):
        self.plots = plots
        self.start = start
        self._end = end
        self.fps = fps
        self.j = 0
        self.counter = 0

        self.fig = plt.figure(figsize=(width, height))
        self.fig.tight_layout()
        self.ax = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1), label="main")
        # self.ax.plot([0.6, 4], [0.1, 0.8])
        plt.gca().set_position([0, 0, 1, 1])
        self.scatter = self.ax.scatter([], [], alpha=1, zorder=zorder())
        self.initialized = False

    def initialize(self):
        # self.pbar = tqdm(total=self.n_frames, desc="Full video")
        self.initialized = True
        total = 0
        for j, plot in enumerate(self.plots):
            if total + plot.n_frames > self.start:
                self.counter = int(self.start - total)
                self.j = j
                break
            else:
                plot.initialize(self.fig, self.ax, self.plots[j - 1] if j > 0 else None)
                plot.leave()
            total += plot.n_frames

    @property
    def n_frames(self):
        return self.end - self.start

    @property
    def end(self):
        if self._end < 0:
            return sum(plot.n_frames for plot in self.plots)
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    @property
    def step(self):
        return int(FPS / self.fps)

    def __call__(self, i):
        if not self.initialized:
            self.initialize()
            return (self.scatter,)

        i *= self.step
        i += self.start
        i = int(i)
        plot = self.plots[self.j]
        if plot.n_frames <= self.counter and self.j + 1 >= len(self.plots):
            plot.leave()
            return (self.scatter,)
        elif plot.n_frames <= self.counter:
            plot.leave()

            # self.j += 1
            # self.counter -= plot.n_frames
            plot = self.plots[self.j]
            plot.initialize(
                self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
            )
            while plot.n_frames <= self.counter:
                plot.leave()
                self.j += 1
                self.counter -= plot.n_frames
                plot = self.plots[self.j]
                plot.initialize(
                    self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
                )

            self.counter = max(self.counter, 0)
        plot.initialize(
            self.fig, self.ax, self.plots[self.j - 1] if self.j > 0 else None
        )
        plot(
            self.counter,
            self.fig,
            self.ax,
            self.plots[self.j - 1] if self.j > 0 else None,
        )
        self.counter = int(self.counter + self.step)
        # self.pbar.update(self.step)
        return (self.scatter,)


class Cover:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.title = plt.text(
            0.5,
            0.7,
            ("Accounting for Variance\n" "in Machine Learning Benchmarks"),
            fontsize=30,
            horizontalalignment="center",
            transform=plt.gcf().transFigure,
            verticalalignment="center",
        )

        authors = """
        Xavier Bouthillier$^{1,2}$, Pierre Delaunay$^3$, Mirko Bronzi$^2$, Assya Trofimov$^{1,2,4}$,
        Brennan Nichyporuk$^{2,5,6}$, Justin Szeto$^{2,5,6}$, Naz Sepah$^{2,5,6}$,
        Edward Raff$^{7,8}$, Kanika Madan$^{1,2}$, Vikram Voleti$^{1,2}$,
        Samira Ebrahimi Kahou$^{2,6}$, Vincent Michalski$^{1,2}$, Dmitriy Serdyuk$^{1,2}$,
        Tal Arbel$^{2,5,6,10}$, Christopher Pal$^{2,9,10,11}$, Gaël Varoquaux$^{2,6,12}$, Pascal Vincent$^{1,2,10}$"""

        affiliations = """
        $^1$Université de Montréal, $^2$Mila, $^3$Independant, $^4$IRIC, $^5$Center for Intelligent Machines,
        $^6$McGill University, $^7$Booz Allen Hamilton, $^8$University of Maryland, Baltimore County,
        $^9$Polytechnique, $^{10}$CIFAR, $^{11}$Element AI, $^{12}$Inria
        """

        self.authors_text = plt.text(
            0.5,
            0.45,
            authors,
            fontsize=16,
            horizontalalignment="center",
            transform=plt.gcf().transFigure,
            verticalalignment="center",
        )

        self.affiliations_text = plt.text(
            0.5,
            0.2,
            affiliations,
            fontsize=12,
            horizontalalignment="center",
            transform=plt.gcf().transFigure,
            verticalalignment="center",
        )
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass
        # self.title.set_position((-1, -1))
        # self.authors_text.set_position((-1, -1))
        # self.affiliations_text.set_position((-1, -1))


class RemoveCover:
    def __init__(self, cover):
        self.n_frames = 0
        self.cover = cover
        self.initialized = True

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self.cover.title.remove()
        self.cover.authors_text.remove()
        self.cover.affiliations_text.remove()
