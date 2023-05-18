from typing import MutableSequence
import math
import numpy as np


def calc_time_between_transitions(samples: MutableSequence[float]) -> MutableSequence[float]:
    # """Calculate delta t between transition point
    #
    #   Parameters
    #   ----------
    #   samples: MutableSequence[float]
    #     Function values
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
            time_gaps.append(n_delta_t)
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
    result = np.zeros(len(samples), dtype=int)
    previous_level = round(samples[0]/math.pi)
    for i in range(1, len(samples)):
        sample_level = round(samples[i]/math.pi)

        if sample_level != previous_level:
            result[i] = 1
            previous_level = sample_level

    return result


# def build_series(input):
#     result = input[:chunk_size].copy()
#     for i in range(len(input)-chunk_size+1):
#         x = []
#         for j in range(i, i + chunk_size):
#             x.append(input[j])

#         x = np.array(x)
#         inp = x.reshape(1, chunk_size, 1)

#         pred = round(base_lstm.predict(inp, verbose=False)[0][0])

#         result = np.append(result, pred)

#     return result