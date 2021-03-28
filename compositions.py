from tqdm import tqdm
from utils import linear


class Parallel:
    def __init__(self, animations):
        self.animations = animations
        self.initialized = False

    @property
    def n_frames(self):
        return max(animation.n_frames for animation in self.animations)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for animation in self.animations:
            animation.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        for animation in self.animations:
            if i <= animation.n_frames:
                animation(i, fig, ax, last_animation)
            else:
                animation.leave()

    def leave(self):
        for animation in self.animations:
            animation.leave()


def reverse(animation):
    if isinstance(animation, Section):
        return Section([Reverse(anim) for anim in animation.plots[::-1]])
    elif isinstance(animation, Parallel):
        return Parallel([Reverse(anim) for anim in animation.animations[::-1]])

    return Reverse(animation)


class Cascade:
    def __init__(self, n_frames, animations):
        self.n_frames_cascade = n_frames
        self.animations = animations
        self.n_frames_per_animation = int(self.n_frames_cascade / len(animations))
        self.initialized = False

    @staticmethod
    def infer_n_frames(n_frames_cascade, anim_n_frames):
        return n_frames_cascade + anim_n_frames[-1]

    @property
    def n_frames(self):
        return self.infer_n_frames(
            self.n_frames_cascade, [anim.n_frames for anim in self.animations]
        )

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        for anim in self.animations:
            anim.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        n = int(linear(0, len(self.animations), i, self.n_frames_cascade))
        for j, anim in enumerate(self.animations[:n]):
            assert i >= j * self.n_frames_per_animation
            anim(i - j * self.n_frames_per_animation, fig, ax, last_animation)

    def leave(self):
        for anim in self.animations:
            anim.leave()


class Reverse:
    def __init__(self, animation):
        self.n_frames = animation.n_frames
        self.animation = animation
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.animation.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.animation(self.n_frames - i, fig, ax, last_animation)

    def leave(self):
        self.animation(0, None, None, None)


class Section:
    def __init__(self, plots):
        self.plots = plots
        self.j = 0
        self.last_i = 0
        self.counter = 0
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.fig = fig
        self.initialized = True

    @property
    def n_frames(self):
        return sum([plot.n_frames for plot in self.plots])

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(fig, ax, last_animation)

        step = i - self.last_i
        self.counter += step
        self.last_i = i

        if self.j >= len(self.plots):
            return

        plot = self.plots[self.j]
        if plot.n_frames <= self.counter and self.j + 1 >= len(self.plots):
            plot.leave()
            return 0

        elif plot.n_frames <= self.counter:
            # self.j += 1
            # self.counter -= plot.n_frames
            plot = self.plots[self.j]
            plot.initialize(fig, ax, self.plots[self.j - 1] if self.j > 0 else None)
            while plot.n_frames <= self.counter:
                plot(
                    plot.n_frames,
                    fig,
                    ax,
                    self.plots[self.j - 1] if self.j > 0 else None,
                )
                plot.leave()
                self.j += 1
                if self.j >= len(self.plots):
                    return 0
                self.counter -= plot.n_frames
                plot = self.plots[self.j]
                plot.initialize(fig, ax, self.plots[self.j - 1] if self.j > 0 else None)

            self.counter = max(self.counter, 0)
        plot.initialize(fig, ax, self.plots[self.j - 1] if self.j > 0 else None)
        plot(
            self.counter,
            fig,
            ax,
            self.plots[self.j - 1] if self.j > 0 else None,
        )
        return step

    def leave(self):
        self(self.n_frames, None, None, None)


class Chapter(Section):
    def __init__(self, name, plots, pbar_position=None):
        super(Chapter, self).__init__(plots)
        self.name = name
        self.pbar_position = pbar_position

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.name:
            self.pbar = tqdm(
                total=self.n_frames,
                leave=True,
                desc=self.name,
                position=self.pbar_position,
            )
        super(Chapter, self).initialize(fig, ax, last_animation)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        step = i - self.last_i
        super(Chapter, self).__call__(i, fig, ax, last_animation)
        if self.name:
            self.pbar.update(step)

    def leave(self):
        self.fig.clear()
        if self.name:
            self.pbar.close()
