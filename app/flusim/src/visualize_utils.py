from math import sqrt
import numpy as np
import scipy as sp

from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.plotting import figure, show


def _get_diff(X: np.array, alpha: float) -> np.float64:
    """
    Computes half length of confidence interval

    """
    n = len(X)
    return sp.stats.t.ppf(1 - alpha / 2, n - 1) * sqrt(np.std(X, ddof=1) / n)


def _get_total_sick(data: np.ndarray, n_sick_duration: int) -> np.ndarray:
    """
    Converts result of newly sick people each day
    to result of number of sick people each day

    """
    n = data.shape[1]
    for i in range(1, n):
        data[:, i] += data[:, i - 1]
    for i in range(0, n - n_sick_duration):
        data[:, n - i - 1] -= data[:, n - i - 1 - n_sick_duration]
    return data


def visualize_single_period(single_period_result) -> None:
    """
    Utility using Bokeh to visualize the result of
    a single period flu simulation

    """
    n_pop_size = single_period_result.data.shape[1]
    x = np.arange(n_pop_size)
    y = single_period_result.data[0, :]
    M = single_period_result.data.shape[0]
    if single_period_result.flu_end is not None:
        M = single_period_result.flu_end
    data = list(single_period_result.data[:M + 1, :].astype(np.int8))

    source = ColumnDataSource(data=dict(x=x, y=y))

    is_ended_str = f"flu ended at day {M}"\
                   if single_period_result.flu_end is not None\
                   else "flu didn't end before max_days."

    plot = figure(
        y_range=(-1, 2),
        width=200 * max(1, (n_pop_size // 15)),
        height=200,
        title=f"Health Status v.s. Days ({is_ended_str})",
        x_axis_label="Person",
        y_axis_label="Status"
        )

    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None
    plot.xaxis.major_label_text_font_size = '0pt'
    plot.yaxis.major_label_text_font_size = '0pt'

    plot.circle('x', 'y', source=source, line_width=3, line_alpha=0.6)

    days = Slider(start=0, end=M, value=0, step=1, title="time_stamp")

    callback = CustomJS(args=dict(
        source=source,
        days=days,
        data=data
        ),
        code="""
        const day = days.value
        const x = source.data.x
        const y = data[day]
        console.log(y)
        source.data = { x, y }
    """)

    days.js_on_change('value', callback)

    show(row(plot, days))


def visualize_multiple_period(
        multiple_period_result,
        n_sick_duration: int,
        alpha: float = 0.05,
        ) -> None:
    """
    Utility using Bokeh to visualize computed confidence intervals
    of the result of a multiple period flu simulation

    """
    data = multiple_period_result.data
    m, n = data.shape
    data = _get_total_sick(data, n_sick_duration)
    x = np.arange(n)
    y_0 = np.mean(data, axis=0)
    d = np.apply_along_axis(lambda X: _get_diff(X, alpha), 0, data)
    y_l, y_h = (y_0 - d, y_0 + d)

    plot = figure(
        width=max(200, (4 * n) // 5),
        height=600,
        title=f"Conf. Intvl. of # of Sick Each Day (alpha={alpha})",
        x_axis_label="Day",
        y_axis_label="Interval"
        )

    plot.line(x, y_0, legend_label="Mean", line_color="red", line_width=2)
    plot.line(x, y_l, legend_label="Lower", line_color="blue", line_width=.3)
    plot.line(x, y_h, legend_label="Upper", line_color="green", line_width=.3)

    show(plot)
