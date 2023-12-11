import math
import numpy as np
from skimage.measure import label

from lux.constants import (
    DIGS_TO_TRANSFER_ICE,
    EARLY_GAME_MAX_LICHEN_TILES_PER_HUB,
    MIN_ICE_START,
)
from lux.utils import lprint
from skimage.segmentation import find_boundaries, expand_labels
from lux.numba_methods import _simulate_lichen_growth


def lake_large_enough(self):
    if not self.ice_hubs:
        return True
    lake_size = self.get_lichen_lake_size()
    supported_tiles = self.supported_lichen_tiles(len(self.ice_hubs))
    return lake_size > 0 and lake_size >= supported_tiles


def water_for_power(self, verbose=False):
    lake_size = self.get_lichen_lake_size()

    water_cost = self.water_cost()
    step = self.game_board.step
    if lake_size == 0 or water_cost == 0:
        return False
    nearby_enemy_factories = len(
        [f for f in self.game_board.agent.opponent_factories if f.distance_to(self) < 5]
    )
    start_watering = self.has_lichen or (
        lake_size > MIN_ICE_START
        if nearby_enemy_factories <= 1
        else lake_size > 2 * MIN_ICE_START
    )
    close_ice = self.ice_points and self.ice_points[0].distance_to(self) < 3

    steps_left = self.game_board.steps_left
    consider = (
        close_ice
        and (not self.power_hub_push or self.full_power_hub())
        and (
            (steps_left < 50 and self.cargo.water - water_cost > steps_left)
            or (
                start_watering
                and (
                    self.cargo.water
                    >= (75 if self.dangerous_enemy_near() else 35 if step < 100 else 50)
                )
                and (
                    self.available_water()
                    >= (35 + water_cost if step < 100 else 50 + water_cost)
                )
                and (
                    not self.supports_factory
                    or (
                        self.supports_factory.available_water()
                        > (
                            75
                            if self.supports_factory.center.closest_enemy_factory_distance
                            > 12
                            else 150
                        )
                        or self.available_water()
                        > (
                            150
                            if self.center.closest_enemy_factory_distance > 12
                            else 200
                        )
                    )
                )
            )
        )
    )
    if verbose:
        lprint(
            f"{self}: water_cost: {water_cost} av-water: {self.available_water()} consider:{consider} "
            f" connected: {self.n_connected_tiles}, growth: {self.n_growth_tiles} Supported tiles: "
            f"{self.supported_lichen_tiles()} lakesize: {lake_size} min_lichen: {self.min_lichen_value}"
        )
    if not consider:
        return False

    if self.min_lichen_value == 100 and self.n_growth_tiles == self.n_connected_tiles:
        return False

    # if a lot of power, build some water reserve such that hub can do an ore run
    if steps_left > 100 and self.available_power() > 2000:
        if self.available_water() < 100:
            return False

    if (
        self.available_water() < 250
        and (not self.ice_points or self.ice_points[0].distance_to(self) > 2)
        and steps_left > 100
    ):
        if verbose:
            lprint(f"{self}: not watering, not enough water and no ice points nearby")
        return False

    ineffient_watering = self.n_connected_tiles == self.n_all_lichen_tiles and (
        self.n_growth_tiles % self.env_cfg.LICHEN_WATERING_COST_FACTOR
    ) in [1, 2, 3, 4, 5]
    efficient = not ineffient_watering

    if (
        (self.available_water() > 250 + water_cost)
        and self.n_growth_tiles > self.n_connected_tiles
        and efficient
    ):
        if verbose:
            lprint(f"{self } A lot of water available, watering")
        return True

    if (
        (self.available_water() > 300 + water_cost)
        and self.n_growth_tiles < lake_size
        and efficient
    ):
        if verbose:
            lprint(f"{self } A lot of water available, watering")
        return True

    more_ice_hub_potential = len(self.ice_hubs) > len(self.heavy_ice_hubs)
    max_tiles = (
        self.supported_lichen_tiles()
        if (
            self.game_board.steps_left < 250
            or self.cargo.water > 250
            or more_ice_hub_potential
        )
        else len(self.heavy_ice_hubs) * EARLY_GAME_MAX_LICHEN_TILES_PER_HUB
    )
    if verbose:
        lprint(
            f"max_tiles: {max_tiles} "
            f"more_ice_hub_potential {more_ice_hub_potential} "
            f"supported_lichen_tiles {self.supported_lichen_tiles()} "
            f"lakesize: {lake_size} n_growth_tiles: {self.n_growth_tiles}"
        )

    if (
        self.available_water() > 250
        and self.n_growth_tiles > self.n_connected_tiles
        and efficient
    ):
        return True

    if self.available_water() > 350 and self.n_connected_tiles < lake_size:
        if verbose:
            lprint(f"{self}: huge water, watering")

        return True

    if self.available_water() < 100 and ineffient_watering:
        if verbose:
            lprint(f"{self}: not watering, ineffient_watering")
        return False

    return (
        lake_size > self.n_growth_tiles
        and max_tiles > self.n_growth_tiles
        and self.available_water() > min(steps_left, 35) + water_cost
    )


def supported_lichen_tiles(self, n_hubs=None):
    """Returns the number of lichen tiles that can be supported by the
    active heavy ice hubs of this factory given half watering scheme"""
    unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
    water_per_hub = (
        unit_cfg.DIG_RESOURCE_GAIN
        // self.env_cfg.ICE_WATER_RATIO
        * (DIGS_TO_TRANSFER_ICE / (1 + DIGS_TO_TRANSFER_ICE))
    )

    if n_hubs is None:
        n_hubs = len(self.heavy_ice_hubs)

        # consider that we can turn off an ore hub and covert to ice mining
        if (
            not n_hubs
            and len(
                [
                    u
                    for u in self.units
                    if u.is_heavy and u.last_point.distance_to(self) <= 4
                ]
            )
            > 0
            and len(self.ice_hubs) > 0
        ):
            n_hubs = 1

    if n_hubs == 0:
        return 0

    supported_tiles = (
        math.ceil(n_hubs * water_per_hub - 1)
        * self.env_cfg.LICHEN_WATERING_COST_FACTOR
        * 2
        - self.env_cfg.FACTORY_WATER_CONSUMPTION
    )
    return supported_tiles


def get_lichen_connector_rubble(self):
    gb = self.game_board
    board = gb.game_state.board
    rb = (
        1.0
        * (board.rubble == 0)
        * (board.ice == 0)
        * (board.ore == 0)
        * (board.factory_occupancy_map < 0)
    )
    for p in self.points:
        rb[p.xy] = 1

    clusters = label(rb, connectivity=1)
    my_id = clusters[self.center.xy]
    my_lake = clusters == my_id
    other_clusters = clusters.copy()
    other_clusters[my_lake] = 0

    distance = 2

    expansion = expand_labels(other_clusters, distance) > 0
    my_expansion = expand_labels(my_lake, distance)

    expansion[my_expansion == 0] = 0
    candidate_map = (
        expansion
        * (board.ice == 0)
        * (board.ore == 0)
        * (board.factory_occupancy_map < 0)
        * (board.rubble > 0)
    )
    return gb.get_points_by_idxs(gb.grid_to_ids(candidate_map))


def simulate_lichen_growth(self):
    """Simulate lichen growth"""
    if not hasattr(self, "_simulated_growth"):
        lichen_tree = self.get_lichen_tree()

        if not lichen_tree:
            return [0, 0, 0, 0]

        tree_level_sizes = np.array([len(level) for level in lichen_tree])
        steps_left = self.game_board.steps_left
        max_t = steps_left

        MIN_LICHEN_TO_SPREAD = self.env_cfg.MIN_LICHEN_TO_SPREAD
        MAX_LICHEN_PER_TILE = self.env_cfg.MAX_LICHEN_PER_TILE
        LICHEN_WATERING_COST_FACTOR = self.env_cfg.LICHEN_WATERING_COST_FACTOR

        self._simulated_growth = _simulate_lichen_growth(
            tree_level_sizes,
            max_t,
            MIN_LICHEN_TO_SPREAD,
            MAX_LICHEN_PER_TILE,
            LICHEN_WATERING_COST_FACTOR,
        )
    return self._simulated_growth


def has_enough_water_to(self, purpose):
    """Check if there is enough water to fulfill given purpose
    ["end", "max", "spread_last", "spread_max"]
    end: to end of the game
    max: to max out the lichen
    spread: to spread the lichen to all tiles with at least 1 lichen
    """
    (
        total_water_end,
        total_water_max,
        total_water_spread_last,
        total_water_spread_max,
    ) = self.simulate_lichen_growth()

    if total_water_end <= 0:
        return False

    water_purpose = {
        "end": total_water_end,
        "max": total_water_max,
        "spread": total_water_spread_last,
        "spread_max": total_water_spread_max,
    }
    factory_factor = len(self.lichen_cluster_factories)
    steps_left = self.game_board.steps_left
    water_available = self.cargo.water + self.avg_nett_water_production() * steps_left

    #
    if purpose == "max" and water_purpose["max"] < 0:
        return False
    if purpose == "spread_max" and water_purpose["spread_max"] < 0:
        return False

    return water_purpose[purpose] / factory_factor <= water_available


def build_lichen_tree(self, depth=0, visited=None):
    """Returns a lichen tree"""
    if depth == 0:
        self._lichen_tree = []
        points = self.neighbour_tiles()
        visited = set()
    else:
        points = self.next_growth_points(self._lichen_tree[depth - 1])

    points = self.filter_growth_points(points, visited)
    visited = visited.union(points)

    if not points:
        self._lichen_lake_size = len(visited)
        return self._lichen_tree
    self._lichen_tree.append(points)

    return self.build_lichen_tree(depth + 1, visited)


def filter_growth_points(self, points, visited=None, skip_rubble=False):
    """Filters out points that can't grow lichen"""
    if visited is None:
        visited = []
    points = [
        p
        for p in points
        if not p.ice
        and not p.ore
        and (skip_rubble or not p.rubble)
        and not p.factory
        and (not p.lichen or p.lichen_strains == self.strain_id)
        and (not p.adjacent_factories or self.unit_id in p.adjacent_factories)
        and (
            p.lichen_strains == self.strain_id
            or not any(
                [
                    ap.lichen and ap.lichen_strains != self.strain_id
                    for ap in p.adjacent_points()
                ]
            )
        )
        and p not in visited
    ]

    return points
