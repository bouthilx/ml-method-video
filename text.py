import numpy

from base import zorder, FPS
from utils import linear, show_text, despine
from transitions import FadeOut, FADE_OUT, Still
from compositions import Section, Parallel, reverse


TEXT_SPEED = 35  # Characters per second


class FadeText:
    def __init__(self, n_frames, text_object, alpha=1, **kwargs):
        self.n_frames = n_frames
        self.text_object = text_object
        self.alpha = alpha
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        alpha = linear(0, self.alpha, i, self.n_frames)
        self.text_object.set_alpha(alpha)

    def leave(self):
        self(self.n_frames, None, None, None)


class WriteText:
    def __init__(self, text, text_object, min_i=0, fill=True):
        self.n_frames = int(FPS * len(text) / TEXT_SPEED)
        self.text = text
        self.text_object = text_object
        self.min_i = min_i
        self.fill = fill
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        show_text(
            self.text_object,
            self.text,
            i,
            self.n_frames,
            min_i=self.min_i,
            fill=self.fill,
        )

    def leave(self):
        self(self.n_frames, None, None, None)


class WriteTextLine:
    def __init__(self, n_frames, text, y, x_padding=0.02, fontsize=24):
        self.n_frames = n_frames
        self.text = text
        self.y = y
        self.x_padding = x_padding
        self.fontsize = fontsize
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes([-1, -1, 0.1, 0.1], zorder=zorder())

        self.text_object = self.ax.text(
            self.x_padding,
            self.y,
            "",
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        self.text = WriteText(
            self.text,
            self.text_object,
            fill=False,
        )
        self.text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class SlideTitle:
    def __init__(
        self, n_frames, position, text, x_padding=0.02, y_padding=0.05, fontsize=34
    ):
        self.n_frames = n_frames
        self.number = position
        if position is not None:
            self.text = f"{position}. {text}"
        else:
            self.text = text
        self.x_padding = x_padding
        self.y_padding = y_padding
        self.fontsize = fontsize
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes([-1, -1, 0.1, 0.1], zorder=zorder())

        self.text_object = self.ax.text(
            self.x_padding,
            1 - self.y_padding,
            "",
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        if self.number is not None:
            min_i = int(numpy.log10(self.number)) + 1
        else:
            min_i = 0
        self.text = WriteText(
            self.text,
            self.text_object,
            min_i=min_i,
            fill=False,
        )
        self.text.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class BulletPoint:
    def __init__(
        self, n_frames, text, animation_builder, position, total, y=0.85, fontsize=24
    ):
        self.n_frames = n_frames
        self.text = f"{position}.\n{text}"
        self.fontsize = 32
        self.x_padding = 0.025
        self.y = y
        self.animation_builder = animation_builder
        self.position = position
        self.total = total
        self.width = 1 / total
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        if self.animation_builder:
            x, y = (self.position - 1) * self.width, 0
        else:
            x, y = -1, -1

        self.ax = fig.add_axes([x, y, self.width, 1], zorder=zorder())
        if self.animation_builder:
            self.animation = self.animation_builder(self.n_frames, self.ax)
            self.animation.initialize(fig, ax, last_animation)
        self.text_object = self.ax.text(
            self.x_padding + (self.position - 1) * self.width,
            self.y,
            f"{self.position}. " + self.text,
            va="top",
            ha="left",
            transform=fig.transFigure,
            fontsize=self.fontsize,
            zorder=10000,  # zorder(),
            clip_on=False,
        )

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        show_text(self.text_object, self.text, i, self.n_frames / 10, min_i=3)
        if self.animation_builder:
            self.animation(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class TextBox:
    def __init__(
        self,
        texts,
        ax,
        x,
        y,
        height,
        min_alpha=0.1,
        fontsize=16,
        minfontsize=8,
        colors=None,
        **kwargs,
    ):
        self.texts = texts
        self.current_text = texts[0]
        self.create_objects(
            ax, x, y, height, fontsize=fontsize, colors=colors, **kwargs
        )
        self.x = x
        self.y = y
        self.min_alpha = min_alpha
        self.minfontsize = minfontsize
        self.fontsize = fontsize
        self.max_delta = height * (len(texts) - 1)
        self.initiate_move(texts[0])
        self.set_text(texts[0])

    def create_objects(self, ax, x, y, height, colors, **kwargs):
        self.objects = []
        for i, text in enumerate(self.texts):
            self.objects.append(
                ax.text(
                    x,
                    y - i * height,
                    text,
                    color=colors[i] if colors else None,
                    **kwargs,
                )
            )

    def get_i(self, text):
        return self.texts.index(text)

    def get_snapshot(self):
        return [text_object.get_position() for text_object in self.objects]

    def initiate_move(self, text):
        self.positions = self.get_snapshot()
        self.moving_to_text = text

    def move_text(self, frac, text):
        assert text == self.moving_to_text, f"Movement to {text} not initiated"
        i = self.get_i(text)
        x, y = self.positions[i]
        diff = (self.y - y) * frac
        for position, text_object in zip(self.positions, self.objects):
            x, y = position
            new_y = y + diff
            text_object.set_position((x, new_y))

            delta = numpy.abs(self.y - new_y)

            delta_ratio = delta / self.max_delta

            opacity = delta_ratio * self.min_alpha + (1 - delta_ratio)
            text_object.set_alpha(opacity)

            fontsize = (
                delta_ratio * self.minfontsize + (1 - delta_ratio) * self.fontsize
            )

            text_object.set_fontsize(fontsize)

        # TODO: Set alpha based on how far the label is from focus

    def set_text(self, text):
        self.move_text(1, text)
        self.current_text = text


class MovingTextBox:
    def __init__(self, n_frames, text_box, target_text):
        self.n_frames = n_frames
        self.text_box = text_box
        self.target_text = target_text
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.text_box.initiate_move(self.target_text)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.text_box.move_text(i / self.n_frames, self.target_text)

    def leave(self):
        self.text_box.set_text(self.target_text)


class CodeTitle:
    def __init__(self, n_frames, title, code_block, x, y, fontsize=28):
        self.n_frames = n_frames
        self.title = title
        self.code_block = code_block
        self.x = x
        self.y = y
        self.fontsize = fontsize
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        text_object = self.code_block.ax.text(
            self.x, self.y - 0.25, "", va="bottom", ha="left", fontsize=self.fontsize
        )

        self.code_block.ax.axhline(0, color="black")

        self.write_text = WriteText(self.title, text_object, fill=False)

        # TODO: Maybe add a line

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.write_text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class CodeLine:
    def __init__(self, n_frames, line, comment, comment_side="left", fontsize=20):
        self.n_frames = n_frames
        self.line = line
        self.comment = comment
        self.fontsize = fontsize
        self.index = 0
        self.comment_side = comment_side
        self.x_padding = 0.05
        if comment_side == "left":
            self.comment_x = 1
        else:
            self.comment_x = 0
            self.x_padding *= -1
        self.code_block = None
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        text_object = self.code_block.ax.text(
            0, self.index, "", va="bottom", ha="left", fontsize=self.fontsize
        )

        self.write_text = WriteText(self.line, text_object, fill=False)

        if self.comment:
            text_object = self.code_block.ax.text(
                self.comment_x + self.x_padding,
                self.index,
                "",
                va="bottom",
                ha=self.comment_side,
                fontsize=self.fontsize,
            )

            self.write_comment = WriteText(self.comment, text_object, fill=False)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.write_text(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class ShowComment:
    def __init__(self, n_frames, line):
        self.n_frames = n_frames
        self.line = line
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.line.write_comment(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class CodeBlock:
    def __init__(self, title, lines, x=0, y=0, width=1, height=None, line_height=0.1):
        self.lines = lines
        self.n_frames = 0
        self.title = title
        self.x = x
        self.y = y
        self.width = width
        if height is None:
            height = line_height * (len(lines) + title.count("\n") + 2)
        self.height = height
        self.line_height = line_height
        self.initialized = False

    def show_title(self, n_frames):
        # TODO: Create the text object
        return CodeTitle(n_frames, self.title, self, x=0, y=0)

    def show_comment(self, n_frames, i):
        commented_lines = [line for line in self.lines if line.comment is not None]
        return ShowComment(n_frames, commented_lines[i])

    def show_lines(self):
        return Section(self.lines)

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes(
            [self.x, 1 - self.y - self.height, self.width, self.height], zorder=zorder()
        )
        self.ax.set_ylim((len(self.lines), -(self.title.count("\n") + 1)))
        despine(self.ax)

        for i, line in enumerate(self.lines):
            line.index = i + 1
            line.code_block = self

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        pass

    def leave(self):
        self(self.n_frames, None, None, None)


class SectionTitle:
    def __init__(self, n_frames, title, fade_ratio=0.5, opacity=0.8, **kwargs):
        self.n_frames = n_frames
        self.title = title
        self.x_padding = 0.1
        self.fontsize = 34
        self.fade_ratio = fade_ratio
        self.opacity = opacity
        self.fade_out_kwargs = kwargs
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.ax = fig.add_axes([-1, -1, 0.1, 0.1], zorder=zorder())

        # self.text_object = self.ax.text(
        #     self.x_padding,
        #     0.6,
        #     "",
        #     va="top",
        #     ha="left",
        #     zorder=zorder(2),
        #     transform=fig.transFigure,
        #     fontsize=self.fontsize,
        # )

        # self.text = WriteText(self.title, self.text_object)

        # self.fade_out = FadeOut(
        #     self.text.n_frames * 2,
        #     self.ax,
        #     opacity=self.opacity,
        #     pause=False,
        #     zorder_pad=-1,
        # )
        # self.fade_out.initialize(fig, ax, last_animation)

        # fade_in = Parallel(
        #     [
        #         self.fade_out,
        #         self.text,
        #     ]  # Section([Still(self.text.n_frames), self.text])]
        # )

        self.fade_out = FadeOut(
            int(FADE_OUT / 2 * self.fade_ratio),
            self.ax,
            opacity=self.opacity,
            pause=False,
            **self.fade_out_kwargs,
        )
        self.fade_out.initialize(fig, ax, last_animation)

        y = self.fade_out_kwargs.get("y", 0)
        text_y = 0.6 * (1 - y) + y

        self.text_object = self.ax.text(
            self.x_padding,
            text_y,
            self.title,  # "",
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        # self.text = WriteText(self.title, self.text_object)
        self.text = FadeText(self.fade_out.n_frames, self.text_object)

        fade_in = Parallel([self.fade_out, self.text])

        fade_out = reverse(fade_in)

        n_frames_still = max(self.n_frames - (fade_in.n_frames + fade_out.n_frames), 0)
        self.section = Section([fade_in, Still(n_frames_still), fade_out])
        self.section.initialize(fig, ax, last_animation)
        assert (
            self.section.n_frames == self.n_frames
        ), f"{self.section.n_frames} != {self.n_frames}"

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        self.section(i, fig, ax, last_animation)

    def leave(self):
        self(self.n_frames, None, None, None)


class ChapterTitle:
    def __init__(self, n_frames, number, title, animation_builder=None):
        self.n_frames = n_frames
        self.number = number
        self.title = f"{number}.\n{title}"
        self.animation_builder = animation_builder
        self.fontsize = 38
        self.x_padding = 0.1
        self.initialized = False

    def initialize(self, fig, ax, last_animation):
        if self.initialized:
            return

        self.fig = fig

        if self.animation_builder:
            x, y = 0.5, 0
        else:
            x, y = -1, -1

        self.ax = fig.add_axes([x, y, 0.5, 1])

        if self.animation_builder:
            self.animation = self.animation_builder(self.n_frames, self.ax)
            self.animation.initialize(fig, ax, last_animation)

        self.text_object = self.ax.text(
            self.x_padding,
            0.6,
            self.title,
            va="top",
            ha="left",
            zorder=zorder(),
            transform=fig.transFigure,
            fontsize=self.fontsize,
        )

        self.fade_out = FadeOut(FADE_OUT, self.ax)
        self.fade_out.initialize(fig, ax, last_animation)

        self.initialized = True

    def __call__(self, i, fig, ax, last_animation):
        show_text(
            self.text_object,
            self.title,
            i,
            self.n_frames / 10,
            min_i=int(numpy.log10(self.number)) + 1,
        )
        if self.animation_builder:
            self.animation(i, fig, ax, last_animation)

        if i > self.n_frames - self.fade_out.n_frames:
            self.fade_out(
                i - (self.n_frames - self.fade_out.n_frames), fig, ax, last_animation
            )

    def leave(self):
        self.fig.clear()
