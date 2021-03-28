from matplotlib import patches
import numpy

from base import FPS, zorder
from utils import linear, show_text

FADE_OUT = FPS * 2


class FadeOut:
    def __init__(
        self,
        n_frames,
        ax=None,
        opacity=1,
        pause=True,
        zorder_pad=0,
        transform=None,
        color="white",
        x=0,
        y=0,
        width=1,
        height=1,
    ):
        self.n_frames = n_frames
        self.ax = ax
        self.opacity = opacity
        self.pause = 2 if pause else 1
        self.zorder_pad = zorder_pad
        self.transform = transform
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.ax is None:
            self.ax = fig.add_axes([-1, -1, 1, 1], zorder=zorder() + self.zorder_pad)

        # TODO: Add white rectangle

        self.patch = patches.Rectangle(
            (self.x, self.y),
            self.width,
            self.height,
            fill=True,
            color=self.color,
            zorder=zorder() + self.zorder_pad,
            transform=fig.transFigure if self.transform is None else self.transform,
            clip_on=False,
            alpha=0.0,
        )
        self.ax.add_patch(self.patch)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        alpha = min(
            linear(0, self.opacity, i, int(self.n_frames / self.pause)), self.opacity
        )
        self.patch.set_alpha(alpha)

    def leave(self):
        self(self.n_frames, None, None, None)


class Still:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        pass


class Black:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.black_patch = patches.Rectangle((0, 0), 0, 0, fill=True, color="black")
        ax.add_patch(self.black_patch)
        self.black_patch.set_width(5)
        self.black_patch.set_height(5)
        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        if not self.initialized:
            self.initialize(ax)

    def leave(self):
        self.black_patch.set_width(0)
        self.black_patch.set_height(0)


class Zoom:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def initialize(self, fig, ax, last_animation):
        pass

    def __call__(self, i, fig, ax, last_animation):
        old_x = 0.2
        new_x = -2
        old_y = 0.6
        new_y = 5
        saturation = 3
        tmp_x = translate(old_x, new_x, i, self.n_frames, saturation=saturation)
        tmp_y = translate(old_y, new_y, i, self.n_frames, saturation=saturation)
        ax.set_position([tmp_x, tmp_x, tmp_y, tmp_y])

    def leave(self):
        pass
