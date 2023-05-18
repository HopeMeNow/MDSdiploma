from typing import Tuple
import matplotlib.pyplot as plt
import math


def get_rad_ticks(range_start: int = 0, range_stop: int = 2, step: int = 1) -> Tuple:
    # """Generate ticks and labels for them in given interval

    #   Parameters
    #   ----------
    #   range_start: int
    #     Left edge of the interval. Will be included in the range.
    #   range_stop: int
    #     Right edge of the interval. Won't be included in the range.
    #   step: int
    #      An integer number specifying the incrementation.

    #   Returns
    #   -------
    #   Tuple
    #     Tuple with two arrays: ticks and labels
    # """
    labels = []

    for i in range(range_start, range_stop, step):
        if i == 0:
            labels.append('0')
            continue
        if i == 1:
            labels.append('π')
            continue
        if i == -1:
            labels.append('-π')
            continue
        labels.append(str(i) + 'π')
    
    ticks = [i*math.pi for i in range(range_start, range_stop, step)]
    return ticks, labels


def set_up_plot(title: str, figsize: Tuple = (12, 6), ticks_range: Tuple = (0, 2)):
    # """Sets up common parameters of plots

    #   Parameters
    #   ----------
    #   title: str
    #     Title of figure
    #   figsize: Tuple
    #     Size of subplots
    #   ticks_range: Tuple
    #      Min and max values of the axes

    #   Returns
    #   -------
    #   Void
    # """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.tight_layout(pad=5.)
    fig.suptitle(title, fontsize=16)

    ax1.set_yticks(*get_rad_ticks(*ticks_range))
    ax1.set_title('Dependence of the function on time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Function in time point')

    ax2.set_xticks(*get_rad_ticks(*ticks_range))
    ax2.set_title('Dependence of the derivative on function')
    ax2.set_xlabel('Function in time point')
    ax2.set_ylabel('Derivative of function in time point')

    return ax1, ax2