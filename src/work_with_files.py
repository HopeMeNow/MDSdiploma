from typing import MutableSequence
from collections.abc import Callable
import numpy as np
from data_generation import generate_noise, get_samples
from utils import mark_bifurcation_points


def add_series_to_file(filename: str, series: MutableSequence[int], path: str):
    # """Adds array in the end of file

    #   Parameters
    #   ----------
    #   filename: str
    #     Name of file without full path
    #   series: MutableSequence[int]
    #     Array to write in file
    #   path: str
    #      Path to folder where file should be saved.

    #   Returns
    #   -------
    #   Void
    # """
    with open(f'{path}{filename}.txt', 'a') as f:
        f.write(' '.join(map(str, series)))
        f.write(' ')


def generate_data(
    noise_amplitude: Callable[[int], float] = lambda f: 1,
    filename: str = 'white',
    path: str = '/content/drive/MyDrive/Colab Notebooks/diploma/',
):
    # """Generates and write in the file series by chunks

    #   Parameters
    #   ----------
    #   noise_amplitude: Callable[[int], float]
    #     Should return amplitude depending on frequency. Return const 1 by default.
    #   filename: str
    #     Name of file without full path
    #   path: str
    #      Path to folder where file should be saved. Folder on google drive by default

    #   Returns
    #   -------
    #   Void
    # """
    series_len = 1_000_000
    series_chunk_len = 10_000
    i = 0
    current_init_point = 0
    while i < series_len:
        noise = generate_noise(series_chunk_len, amplitude=noise_amplitude)
        x_ts, _ = get_samples(series_chunk_len, x_t_init = current_init_point, noise=noise)
        current_init_point = x_ts[-1:]
        x_ts = mark_bifurcation_points(x_ts)

        add_series_to_file(filename, x_ts, path)

        i += series_chunk_len


def read_series_from_file(
    filename: str = 'white',
    path: str = '/content/drive/MyDrive/Colab Notebooks/diploma/',
) -> MutableSequence[int]:
    # """Reads array from the file

    #   Parameters
    #   ----------
    #   filename: str
    #     Name of file without full path
    #   path: str
    #      Path to folder where file should be saved.

    #   Returns
    #   -------
    #   [int]
    #     The series from the file type of ndarray
    # """
    with open(f'{path}{filename}.txt', 'r') as f:
        series = f.read().split(' ')
        series = list(map(int, series[:len(series) - 1]))

    return np.array(series)