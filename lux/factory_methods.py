import math

import numpy as np

from lux.constants import (
    MIN_ICE_START,
    POWERHUB_BEAM_DISTANCE,
    POWERHUB_OREFACTORY_BEAM_DISTANCE,
    TTMAX,
    BREAK_EVEN_ICE_TILES,
)
from lux.utils import flatten, lprint


def set_hubs(self, verbose=False):
    # to add here is how many hubs we can support, multiple hubs is always ice
    candidates = [p for p in self.adjacent_points if p.ice or p.ore]
    if not candidates:
        self.hubs = []
        # assert IS_KAGGLE, "No hubs found?"
        return

    ore_points = [p for p in candidates if p.ore].copy()
    ice_points = [p for p in candidates if p.ice].copy()

    rss_points = ice_points + ore_points

    if verbose:
        lprint(f"ore_points: {ore_points}, ice_points: {ice_points}")
    # for corners, remove one of the hubs as it cannot be handled with a single light charger?
    corners = [p for p in self.edge_tiles() if p.is_factory_corner]
    for corner in corners:
        if corner.unit and corner.unit.is_charger and corner.unit.is_heavy:
            # a heavy charger can support two hubs
            continue
        chubs = [p for p in corner.adjacent_points() if p in rss_points]

        if len(chubs) == 2:
            exclude = chubs[0]
            for p in [p for p in corner.adjacent_points() if p in rss_points]:
                if p.ore or (not any(u.is_hub for u in p.dug_by)):  # p.unit.is_hub):
                    exclude = p
                    break

            if exclude in ore_points:
                ore_points.remove(exclude)
            if exclude in ice_points:
                ice_points.remove(exclude)
    if verbose:
        lprint(
            f"After corner removal ore_points: {ore_points}, ice_points: {ice_points}"
        )

    has_ice = len(ice_points) > 0

    if self.power_hub_push and not ore_points:
        ore_points = [p for p in candidates if p.ore][:1]
    include_ore = len(ore_points) > 0 and self.cargo.metal < 250

    # n_supported_hubs = 1 + self.n_connected_tiles // BREAK_EVEN_ICE_TILES

    power_production = self.power_production()
    hubs_consumption = self.power_consumption()

    n_supported_hubs = max(1, (power_production - hubs_consumption) // 50)
    if self.power_hub_push:
        n_supported_hubs = 2 if self.n_heavies >= 2 else 1

    hubs = []

    def select_best(points, n=1):
        # prefer points furthest from enemy needs closest_factory_distance enemy version
        # prefer points close to center and furthest from center of the map
        points = [p for p in points if p not in hubs]
        sorted_candidates = sorted(
            points,
            key=lambda p: (
                p.distance_to(self.center),
                -p.distance_to(self.game_board.get_point((24, 24))),
            ),
        )
        if verbose:
            lprint(f"select_best: sorted_candidates: {sorted_candidates} n: {n}")
        return sorted_candidates[: int(n)]

    if verbose:
        lprint(f"n_supported_hubs: {n_supported_hubs}", self.power_hub_push)
    if include_ore:
        # has both ice and ore depends on lichen lake
        # select to go for ice or ore.
        if (
            self.power_hub_push or self.get_lichen_lake_size()
        ) < MIN_ICE_START and self.cargo.water > 100:
            # select best ore
            best = select_best(ore_points)[0]
            best.is_hub_candidate = True
            hubs.append(best)

    if has_ice:
        bests = select_best(ice_points, n_supported_hubs - len(hubs))
        for best in bests:
            best.is_hub_candidate = True
            hubs.append(best)
    if include_ore:
        bests = select_best(ore_points, n_supported_hubs - len(hubs))
        for best in bests:
            best.is_hub_candidate = True
            hubs.append(best)

    if verbose:
        lprint(f"hubs{hubs}")

    self.hubs = hubs


def is_crowded(self, max_factor=0.4, verbose=False):
    """Returns True if this factory is crowded"""

    # number of units at factory itself
    units_in_factory = len([p for p in self.points if p.unit])
    if units_in_factory > math.ceil(max_factor * len(self.points)):
        if verbose:
            lprint(
                f"Crowded because of units in factory:{units_in_factory}> {max_factor * len(self.points)}"
            )
        return True

    # number of units directly around the factory
    neighbours = self.neighbour_tiles()
    units_in_neighbourhood = len([p for p in neighbours if p.unit and p.unit.is_own])
    if units_in_neighbourhood > math.ceil(max_factor * len(neighbours)):
        if verbose:
            lprint(
                f"Crowded because of units in neighbourhood:{units_in_neighbourhood}> {max_factor * len(neighbours)}"
            )
        return True

    # number of units within 6 tiles
    points_close = self.points_within_distance(6)
    units_close = len([p for p in points_close if p.unit and p.unit.is_own])

    if units_close > math.ceil(len(points_close) * max_factor**2):
        if verbose:
            lprint(
                f"Crowded because of units close:{units_close}> {len(points_close) * max_factor ** 2}"
            )
        return True

    return False


def get_power_hub_positions(self, factory_only=False, verbose=False):
    if self._power_hub_positions is None or verbose:
        f_x, f_y = self.center.xy

        def is_in_beam(p):
            x, y = p.xy
            return abs(f_x - x) <= 1 or abs(f_y - y) <= 1

        beam_rss = [
            p
            for p in self.ore_points + self.ice_points
            if p.closest_own_factory == self
            and p.distance_to(self)
            <= (
                POWERHUB_OREFACTORY_BEAM_DISTANCE
                if self.ore_only
                else POWERHUB_BEAM_DISTANCE
            )
            and is_in_beam(p)
        ]

        if verbose:
            lprint(f"{self}: beam_rss: {beam_rss}")

        sides = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        sides = [(0, 1), (0, -1), (1, 1), (1, -1)]  # (x or y, direction)
        center = self.center.xy

        candidates = []
        for idx, sign in sides:
            inv_idx = 1 - idx
            #     for p in beam_rss:
            #         print(idx, sign, p.xy, f"{sign*p.xy[idx]}>{sign*center[idx]}", f"{abs(center[inv_idx]-p.xy[inv_idx])}", (sign*p.xy[idx]) > center[idx] and abs(center[inv_idx]-p.xy[inv_idx])<=1)
            is_free = not any(
                [
                    p
                    for p in beam_rss
                    if (sign * p.xy[idx]) > sign * center[idx]
                    and abs(center[inv_idx] - p.xy[inv_idx]) <= 1
                ]
            )

            if is_free:
                xy = np.array(center)
                xy[idx] += 2 * sign  # outside

                if xy[0] < 0 or xy[0] > 47 or xy[1] < 0 or xy[1] > 47:
                    continue

                position = self.game_board.get_point(xy)

                candidates.append(position)
                candidates.append(self.closest_tile(position))

        charger_positions = [
            self.closest_tile(p) for p in self.ore_hubs + self.ice_hubs
        ]

        cross_points = [
            p
            for p in charger_positions + candidates
            if p.factory and not p.is_factory_corner
        ]

        drop_position = None
        if len(cross_points) == 4 and candidates:
            # drop the closest to the enemy, as we likely want to spam units that way
            drop_position = min(
                [p for p in candidates if p.factory],
                key=lambda p: p.closest_enemy_factory_distance,
            )

        candidates = [p for p in candidates if p != drop_position]
        self._power_hub_positions = candidates
    candidates = self._power_hub_positions

    if factory_only:
        return [p for p in candidates if p.factory]

    return candidates


def edge_tiles(self, unit=None, no_chargers=False):
    """
    Returns the tiles surrounding the center
    """
    if no_chargers and self.chargers:
        points = self.center.surrounding_points()
        chargers = [
            u
            for u in self.chargers
            if u.is_heavy or not (len(u.charges_units) == 1 and unit in u.charges_units)
        ]
        powerhubs = [
            p
            for p in points
            if any([u.last_point == p and u.is_power_hub for u in p.visited_by])
        ]
        charger_points = [u.last_point for u in chargers] + powerhubs
        return [p for p in points if p not in charger_points]

    return self.center.surrounding_points()


def n_lichen_tiles_div_unit_count_div_dist(self, unit, verbose=False):
    if self._n_lichen_tiles_div_unit_count_div_dist[unit.unit_type] is None or verbose:
        distance = min(unit.point.distance_to(self), self.game_board.steps_left, TTMAX)

        metric = (
            self.n_connected_tiles
            if self.game_board.steps_left > 100
            else self.lichen / 100
        )
        units = [
            u
            for u in self.units
            if unit.is_heavy == u.is_heavy and u.point.distance_to(self) <= 6
        ]
        unit_count = len(units)
        if unit.is_heavy:
            unit_count += max(
                min(
                    self.metal_in_time[distance] // 100,
                    self.power_in_time[distance] // 500,
                ),
                0,
            )
        else:
            unit_count += min(
                min(
                    self.metal_in_time[distance] // 10,
                    self.power_in_time[distance] // 50,
                ),
                5,
            )

        total_power = (
            min(
                self.power + sum([u.power for u in units]),
                unit_count * unit.battery_capacity,
            )
            / unit.battery_capacity
        )

        if verbose:
            lprint(
                self,
                "distance",
                distance,
                "unit_count",
                unit_count,
                "metric",
                metric,
                "self.n_connected_tiles",
                self.n_connected_tiles,
                "total_power",
                total_power,
            )

        # in case we have a lot of units there already, makes no sense to send more?
        # own_units_factor = max(1, self.n_connected_lichen_tiles / len(self.own_units_in_range()))

        if unit_count == 0 or total_power == 0:
            result = metric / max(1, distance / 10)
        else:
            result = metric / unit_count / total_power / max(1, distance / 10)
        self._n_lichen_tiles_div_unit_count_div_dist[unit.unit_type] = result
    return self._n_lichen_tiles_div_unit_count_div_dist[unit.unit_type]


def is_ore_risk(self, verbose=False):
    next_to_ore = [p for p in self.ore_points if p.distance_to(self) <= 1]
    if next_to_ore:
        return False

    close_to_ore = [p for p in self.ore_points if p.distance_to(self) <= 3]
    if self.is_enemy:
        if close_to_ore:
            return False

    nearby_ore = [
        p
        for p in self.ore_points
        if p.distance_to(self) < (8 if self.is_own else 10)
        and (
            p.closest_factory.team_id == self.team_id
            or p.closest_factory.is_enemy
            and p.closest_factory_distance > 5
        )
    ]
    if verbose:
        lprint(f"{self}: nearby_ore: {nearby_ore}")
    return not self.ore_hubs and len(nearby_ore) < (3 if close_to_ore else 4) + (
        0 if self.is_enemy else 1
    )


def own_units_in_range(self):
    if self._own_units_in_range is None:
        self._own_units_in_range = set(
            flatten(
                [
                    u
                    for p in self.center.points_within_distance(8)
                    for u in p.visited_by
                    if p.unit and u.is_own
                ]
            )
        )
    return self._own_units_in_range


def clear_rubble_for_lichen(self):
    if not self.power_hub_push:
        return True

    if self.ore_only:
        return False

    # must be double touch when here so 2 hubs + 2 chargers (could be corner.)
    if self.n_heavies >= 3 + self.n_power_hub_positions:
        return True

    return False


def consider_lichen_assault(self, unit=None):
    if self.n_all_lichen_tiles == 0:
        return False

    if self.game_board.steps_left < 100:
        return True

    if self.n_lichen_tiles < BREAK_EVEN_ICE_TILES:
        return False

        # (f.n_lichen_tiles >= BREAK_EVEN_ICE_TILES)
        # or f.available_water() > 200
        # or steps_left < 75
        # or len(agent.units) / max(1, len(agent.opponent_units)) > 2
        # or agent.n_opp_heavies > agent.n_opp_lights

    return self.n_expand_frontier < 6 or self.n_expand_frontier < 3 * len(
        self.own_units_in_range()
    )
