from typing import Tuple, MutableSequence
import random
import math
import numpy as np
import matplotlib.pyplot as plt


series_len = 5000
random.seed(0)


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


def noise(freq, series_len):
    # """Return single frequency noise

    #   Parameters
    #   ----------
    #   freq: int
    #     Noise frequency
    #   series_len: int
    #     Length of noise vector
    #
    #   Returns
    #   -------
    #   [float]
    #     The noise vector of a given frequency
    # """
    phase = random.uniform(0, 2*math.pi)
    return np.array([math.sin(2*math.pi * freq*x/series_len + phase) for x in range(series_len)])


def generate_noise(series_len = series_len, amplitude = lambda f: 1, frequencies = range(1, 1001), random_seed = 0, debugging = False):
    # """Generates noise of different spectrum

    #   Parameters
    #   ----------
    #   series_len: int
    #     Length of noise vector
    #   amplitude: Callable[[int], float]
    #     Should return amplitude depending on frequency. Return const 1 by default.
    #   frequencies: [int]
    #     Range of frequencies
    #   random_seed: int
    #     Value for random.seed()
    #   debugging: boolean
    #     Flag to activate debug mode
    #
    #   Returns
    #   -------
    #   [float]
    #     The noise vector of a given spectrum
    # """
    random.seed(random_seed)
    sum_of_noises = [0.0] * series_len
    amplitudes = np.array([amplitude(f) for f in frequencies])
    noises = np.array([noise(f, series_len) for f in frequencies])

    if debugging:
        print('amplitudes length = {}'.format(len(amplitudes)))
        print('noises length = {}'.format(len(noises)))

    for k in range(len(noises)):
        sum_of_noises += amplitudes[k] * noises[k]

    return sum_of_noises


def derivative_next_value(x_t: float, f_t: float = 0) -> float:
    # """Calculate value of x'(t+1) depending on x(t) and f(t)

    #   Parameters
    #   ----------
    #   x_t: float
    #     Value of the function in point t
    #   f_t: float
    #     Noise in point t

    #   Returns
    #   -------
    #   float
    #     Value of the derivative in point t+1
    # """
    return np.sin(x_t) + f_t


def function_next_value(x_t: float, delta: float = 0.1, f_t: float = 0) -> float:
    # """Calculate value of x(t+1) depending on x(t), f(t) and delta t

    #   Parameters
    #   ----------
    #   x_t: float
    #     Value of the function in point t
    #   delta: float
    #     Increment of the argument
    #   f_t: float
    #     Noise in point t

    #   Returns
    #   -------
    #   float
    #     Value of the function in point t+1
    # """
    d_t = derivative_next_value(x_t, f_t)
    X = x_t + delta*d_t
    return x_t + (delta/2)*(d_t + derivative_next_value(X))


def get_samples(series_len: int = series_len, x_t_init: float = 0, random_seed: int = 0, noise: MutableSequence[float] = None, delta: float = 0.1) -> Tuple:
    # """Calculate value of x(t+1) depending on x(t), f(t) and delta t

    #   Parameters
    #   ----------
    #   series_len: int
    #     Length of noise vector
    #   random_seed: int
    #     Value for random.seed()
    #   noise: MutableSequence[float]
    #     Noise vector
    #   delta: float
    #     Increment of the argument

    #   Returns
    #   -------
    #   Tuple
    #     Two vectors: function values in the time points and values of derivative of this function in the same points of time
    # """
    random.seed(random_seed)

    if noise is None:
        noise = [0 for _ in range(series_len)]

    x_t = x_t_init
    x_ts = np.array([])
    d_ts = np.array([])
    x_ts = np.append(x_ts, x_t)
    d_ts = np.append(d_ts, derivative_next_value(x_t))
    for t in range(1, series_len):
        x_t = function_next_value(x_t, delta, f_t=noise[t])
        d_t = derivative_next_value(x_t, f_t=noise[t])
        x_ts = np.append(x_ts, x_t)
        d_ts = np.append(d_ts, d_t)

    return x_ts, d_ts


def calc_time_between_transitions(samples: MutableSequence[float], delta_t: float = 0.1) -> MutableSequence[float]:
    # """Calculate delta t between transition point
    #
    #   Parameters
    #   ----------
    #   samples: MutableSequence[float]
    #     Function values
    #   delta_t: float
    #     Time increment
    #
    #   Returns
    #   -------
    #   MutableSequence[float]
    #     Array with values of time 
    # """
    time_gaps = []
    n_delta_t = 0
    previous_level = round(samples[0]/math.pi)
    for i in range(1, len(samples)):
        sample_level = round(samples[i]/math.pi)
        n_delta_t += 1

        if sample_level != previous_level:
            time_gaps.append(n_delta_t*delta_t)
            n_delta_t = 0
            previous_level = sample_level
    return time_gaps


def mark_bifurcation_points(samples: MutableSequence[float]) -> MutableSequence[float]:
    # """Mark each point with 1 if a transition on the other level has occurred and 0 if not
    #
    #   Parameters
    #   ----------
    #   samples: MutableSequence[float]
    #     Function values
    #
    #   Returns
    #   -------
    #   MutableSequence[int]
    #     Array with ones and zeros with the length of the samples array
    # """
    result = np.zeros(len(samples))
    previous_level = round(samples[0]/math.pi)
    for i in range(1, len(samples)):
        sample_level = round(samples[i]/math.pi)

        if sample_level != previous_level:
            result[i] = 1
            previous_level = sample_level

    return result


def set_up_plot(title: str, figsize: Tuple = (12, 6), ticks_range: Tuple = (0, 2)):
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