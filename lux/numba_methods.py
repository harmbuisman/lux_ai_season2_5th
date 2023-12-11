from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def _simulate_lichen_growth(
    tree_level_sizes,
    max_t,
    MIN_LICHEN_TO_SPREAD,
    MAX_LICHEN_PER_TILE,
    LICHEN_WATERING_COST_FACTOR,
):
    """Simulates the lichen growth for a given number of turns"""

    # lichen tree starts at level=0 with adjacent tiles of factory
    depth = len(tree_level_sizes)
    lichen = np.zeros(depth)

    # tiles_in_time = {}
    # lichen_in_time = {}
    # watering_cost_in_time = {}

    total_water_end = 0  # water needed until end of the game when starting now
    total_water_max = -1  # water needed until max lichen is reached
    total_water_spread_last = -1  # water needed until last lichen spread
    total_water_spread_max = -1  # water needed until lichen can no longer spread

    t = 0

    max_reached = False
    stopped_spreading = False

    prev_max_level = -1
    while t <= max_t:
        n_tiles = 0
        total_lichen = 0
        prev_level_lichen = 0

        max_level = -1
        for level, n_tiles_level in enumerate(tree_level_sizes):
            # grow lichen
            if level == 0 or prev_level_lichen >= MIN_LICHEN_TO_SPREAD:
                new_lichen = min(lichen[level] + 1, MAX_LICHEN_PER_TILE)
                lichen[level] = new_lichen
                prev_level_lichen = new_lichen

                n_tiles += n_tiles_level
                total_lichen += new_lichen * n_tiles_level
            else:
                if not stopped_spreading and level == depth - 1:
                    stopped_spreading = True
                    total_water_spread_max = total_water_end
                    # print(
                    #     f"lichen stopped spreading at level {level} {total_water_spread_max, watering_cost}"
                    # )
                break
            max_level = level

        # watering cost is determined by the number of tiles after growth!
        watering_cost = int(np.ceil(n_tiles / LICHEN_WATERING_COST_FACTOR))

        # watering_cost_in_time[t] = watering_cost
        # tiles_in_time[t] = n_tiles
        # lichen_in_time[t] = total_lichen
        total_water_end += watering_cost

        # all lichen tiles are 100 lichen
        if total_lichen == n_tiles * MAX_LICHEN_PER_TILE and not max_reached:
            max_reached = True
            total_water_max = total_water_end

        if prev_max_level < max_level:
            # we grew a new level
            # print(
            #     f"lichen spread to level {max_level} and has {n_tiles_level} tiles to a total of {n_tiles} tiles"
            # )
            total_water_spread_last = total_water_end
        prev_max_level = max_level

        # water at end of turn, grow lichen at start of turn

        t += 1
    return (
        total_water_end,
        total_water_max,
        total_water_spread_last,
        total_water_spread_max,
    )
