import sys
from collections.abc import Iterable

from pathlib import Path
import numpy as np

from lux.constants import IS_KAGGLE


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


def lprint(*args, **kwargs):
    # return
    if IS_KAGGLE:
        return
    # if "file" in kwargs:
    kwargs["file"] = sys.stderr
    print(*args, **kwargs, flush=True)


def _flatten(container):
    """Flattens nested lists"""
    if isinstance(container, str):
        yield container
        return

    for i in container:
        if isinstance(i, Iterable):
            for j in flatten(i):
                yield j
        else:
            yield i


def flatten(container):
    return list(_flatten(container))


def get_n_primes(n):
    primes = []
    i = 2
    while len(primes) < n:
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
        i += 1
    return primes


def cross_filter(N, remove_center=False):
    """NxN filter that excludes the corners"""
    cross_filter = np.ones((N, N))
    cross_filter[0, 0] = 0
    cross_filter[N - 1, N - 1] = 0
    cross_filter[0, N - 1] = 0
    cross_filter[N - 1, 0] = 0
    if remove_center:
        center = N // 2
        cross_filter[center - 1 : center + 2, center - 1 : center + 2] = 0

    return cross_filter


def create_abort_file():
    if IS_KAGGLE:
        return

    path = Path("delete_to_abort.txt")
    if not path.exists():
        path.touch()


def abort_if_file_gone():
    if IS_KAGGLE:
        return

    path = Path("delete_to_abort.txt")
    if not path.exists():
        assert False, "delete_to_abort.txt File not found"


def set_inverse_zero(matrix, point, max_distance, min_distance=None, get_uniques=False):
    x, y = point.xy
    x_start, x_end = max(0, x - max_distance), min(matrix.shape[0], x + max_distance)
    y_start, y_end = max(0, y - max_distance), min(matrix.shape[1], y + max_distance)

    matrix[:x_start, :] = 0
    matrix[x_end:, :] = 0
    matrix[:, :y_start] = 0
    matrix[:, y_end:] = 0

    if min_distance is not None:
        x_start, x_end = max(0, x - min_distance), min(
            matrix.shape[0], x + min_distance
        )
        y_start, y_end = max(0, y - min_distance), min(
            matrix.shape[1], y + min_distance
        )

        matrix[x_start:x_end, y_start:y_end] = 0

    if get_uniques:
        submatrix = matrix[x_start:x_end, y_start:y_end]
        items = submatrix.ravel()
        return matrix, np.unique(items[items > 0])

    return matrix
