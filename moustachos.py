def adjust_moustachos(
    plots,
    x,
    y,
    whisker_width=0,
    whisker_length=0,
    center_width=0,
    center_line_width=None,
    whisker_line_width=None,
):
    if not isinstance(whisker_length, tuple):
        whisker_length = (whisker_length, whisker_length)

    if whisker_width:
        plots["bottom_whisker"].set_xdata([x - whisker_width, x + whisker_width])
        plots["top_whisker"].set_xdata([x - whisker_width, x + whisker_width])
        plots["bottom_whisker"].set_ydata(
            [y - whisker_length[0], y - whisker_length[0]]
        )
        plots["top_whisker"].set_ydata([y + whisker_length[1], y + whisker_length[1]])

    if whisker_length[0] or whisker_length[1]:
        plots["whisker_body"].set_xdata([x, x])
        plots["whisker_body"].set_ydata([y - whisker_length[0], y + whisker_length[1]])

    if center_width:
        plots["center"].set_xdata([x - center_width, x + center_width])
        plots["center"].set_ydata([y, y])


def moustachos(
    ax,
    x,
    y,
    whisker_width=0,
    whisker_length=0,
    center_width=0,
    center_line_width=None,
    whisker_line_width=None,
):
    if not isinstance(whisker_length, tuple):
        whisker_length = (whisker_length, whisker_length)

    plots = {}

    if whisker_width:
        plots["bottom_whisker"] = ax.plot(
            [x - whisker_width, x + whisker_width],
            [y - whisker_length[0], y - whisker_length[0]],
            color="black",
        )[0]
        plots["top_whisker"] = ax.plot(
            [x - whisker_width, x + whisker_width],
            [y + whisker_length[1], y + whisker_length[1]],
            color="black",
        )[0]

    if whisker_length[0] or whisker_length[1]:
        plots["whisker_body"] = ax.plot(
            [x, x],
            [y - whisker_length[0], y + whisker_length[1]],
            color="black",
        )[0]

    if center_width:
        plots["center"] = ax.plot(
            [x - center_width, x + center_width],
            [y, y],
            color="black",
            linewidth=3,
        )[0]

    return plots


def adjust_h_moustachos(
    plots,
    x,
    y,
    whisker_width=0,
    whisker_length=0,
    center_width=0,
    center_line_width=None,
    whisker_line_width=None,
):
    if not isinstance(whisker_length, tuple):
        whisker_length = (whisker_length, whisker_length)

    if whisker_width:
        plots["left_whisker"].set_xdata([x - whisker_length[0], x - whisker_length[0]])
        plots["right_whisker"].set_xdata([x + whisker_length[1], x + whisker_length[1]])
        plots["left_whisker"].set_ydata([y - whisker_width, y + whisker_width])
        plots["right_whisker"].set_ydata([y - whisker_width, y + whisker_width])

    if whisker_length[0] or whisker_length[1]:
        plots["whisker_body"].set_xdata([x - whisker_length[0], x + whisker_length[1]])
        plots["whisker_body"].set_ydata([y, y])

    if center_width:
        plots["center"].set_xdata([x, x])
        plots["center"].set_ydata([y - center_width, y + center_width])


def h_moustachos(
    ax,
    x,
    y,
    whisker_width=0,
    whisker_length=0,
    center_width=0,
    center_line_width=None,
    whisker_line_width=None,
    **kwargs,
):
    if not isinstance(whisker_length, tuple):
        whisker_length = (whisker_length, whisker_length)

    plots = {}

    if whisker_width:
        plots["left_whisker"] = ax.plot(
            [x - whisker_length[0], x - whisker_length[0]],
            [y - whisker_width, y + whisker_width],
            color="black",
            **kwargs,
        )[0]
        plots["right_whisker"] = ax.plot(
            [x + whisker_length[1], x + whisker_length[1]],
            [y - whisker_width, y + whisker_width],
            color="black",
            **kwargs,
        )[0]

    if whisker_length[0] or whisker_length[1]:
        plots["whisker_body"] = ax.plot(
            [x - whisker_length[0], x + whisker_length[1]],
            [y, y],
            color="black",
            **kwargs,
        )[0]

    if center_width:
        plots["center"] = ax.plot(
            [x, x],
            [y - center_width, y + center_width],
            color="black",
            linewidth=3,
            **kwargs,
        )[0]

    return plots
