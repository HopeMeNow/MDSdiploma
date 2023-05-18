from typing import Tuple, MutableSequence
import random
import math
import numpy as np


series_len = 5000
random.seed(0)


def noise(freq, series_len) -> float:
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
    return np.array([np.sin(2*math.pi * freq*x/series_len + phase) for x in range(series_len)])


def generate_noise(
    series_len = series_len,
    amplitude = lambda f: 1,
    frequencies = range(1, 1001),
    random_seed = 0,
    debugging = False
) -> MutableSequence[float]:
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


def _sin(x: float) -> float:
    # """Calculate sin(x) by the first 19 members of the taylor series

    #   Parameters
    #   ----------
    #   x: float
    #     Argument for sin

    #   Returns
    #   -------
    #   float
    #     Value of sin in point x
    # """
    result = 0
    sign = 1
    xx = x * x
    pw = x
    ft_i = 1.

    for i in range(1, 20, 2):
        ft_i /= i
        result += sign * pw * ft_i
        ft_i /= (i + 1)
        sign = -sign
        pw  *= xx

    return result


def derivative_next_value(x_t: float, f_t: float = 0, alpha: float = 1) -> float:
    # """Calculate value of x'(t+1) depending on x(t) and f(t)

    #   Parameters
    #   ----------
    #   x_t: float
    #     Value of the function in point t
    #   f_t: float
    #     Noise in point t
    #   alpha: float
    #     Noise multiplier 

    #   Returns
    #   -------
    #   float
    #     Value of the derivative in point t+1
    # """
    return np.sin(x_t) + alpha*f_t


def function_next_value(x_t: float, delta: float = 0.1, f_t: float = 0, alpha: float = 1) -> float:
    # """Calculate value of x(t+1) depending on x(t), f(t) and delta t

    #   Parameters
    #   ----------
    #   x_t: float
    #     Value of the function in point t
    #   delta: float
    #     Increment of the argument
    #   f_t: float
    #     Noise in point t
    #   alpha: float
    #     Noise multiplier

    #   Returns
    #   -------
    #   float
    #     Value of the function in point t+1
    # """
    d_t = derivative_next_value(x_t, f_t, alpha)
    x_1 = x_t + delta*d_t
    return x_t + (delta/2)*(d_t + derivative_next_value(x_1, f_t, alpha))


def get_samples(
    series_len: int = series_len,
    x_t_init: float = 0,
    random_seed: int = 0,
    noise: MutableSequence[float] = None,
    delta: float = 0.1,
    alpha: float = 1,
) -> Tuple:
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
    #   alpha: float
    #     Noise multiplier

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
    d_ts = np.append(d_ts, derivative_next_value(x_t, noise[0], alpha))
    for t in range(1, series_len):
        x_t = function_next_value(x_t, delta, noise[t], alpha)
        d_t = derivative_next_value(x_t, noise[t], alpha)
        x_ts = np.append(x_ts, x_t)
        d_ts = np.append(d_ts, d_t)

    return x_ts, d_ts