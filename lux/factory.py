from collections import defaultdict

import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries

from lux.cargo import UnitCargo
from lux.constants import (
    IS_KAGGLE,
    MAX_EFFICIENT_LIGHT_DISTANCE,
    MAX_FREE_LIGHT_MOVEMENT,
    MOVES,
    TMAX,
    TTMAX,
)
from lux.factory_lichen import (
    build_lichen_tree,
    filter_growth_points,
    get_lichen_connector_rubble,
    has_enough_water_to,
    lake_large_enough,
    simulate_lichen_growth,
    supported_lichen_tiles,
    water_for_power,
)
from lux.factory_methods import (
    is_crowded,
    set_hubs,
    get_power_hub_positions,
    edge_tiles,
    is_ore_risk,
    n_lichen_tiles_div_unit_count_div_dist,
    own_units_in_range,
    clear_rubble_for_lichen,
    consider_lichen_assault,
)
from lux.point import Point
from lux.router import get_rss_optimal_path
from lux.utils import flatten, lprint
from lux.rubble import get_transport_rubble_points, get_assault_rubble_points
from lux.actions import PowerHub


def set_lichen_lake_ids(gb):
    """Connects rubble-free spots"""
    rb = (
        1.0
        * (gb.board.rubble == 0)
        * (gb.board.ice == 0)
        * (gb.board.ore == 0)
        #     * (fb.board.factory_occupancy_map < 0)
    )
    clusters = label(rb, connectivity=1)

    gb.lichen_clusters = clusters

    cluster_to_factory = defaultdict(list)
    for f in gb.factories:
        f.lichen_lake_id = clusters[f.center.xy]
        cluster_to_factory[f.lichen_lake_id].append(f)

    for _, factories in cluster_to_factory.items():
        for f in factories:
            f.lichen_cluster_factories = factories


class Factory:
    def __init__(self, env_cfg, **f_data) -> None:
        self.env_cfg = env_cfg
        self.team_id = f_data["team_id"]
        self.unit_id = f_data["unit_id"]
        self.strain_id = f_data["strain_id"]
        self.power = f_data["power"]
        self.cargo = UnitCargo(f_data["cargo"])
        self.pos = np.array(f_data["pos"])
        self.lichen_points = []
        self.lichen_lake_id = None
        self.lichen_cluster_factories = []
        self.is_own = None
        self.is_enemy = None
        self.id = int(".".join([s for s in self.unit_id if s.isdigit()]))
        self.supports_factory = None

        self.n_all_lichen_tiles = 0

        self.reset()

        self.points = []
        self.adjacent_points = []

        self.center: Point = None

        self._neighbour_tiles = None
        self._distance_to = {}
        self._points_within_distance = {}

    def reset(self):
        # new attributes
        self.game_board = None
        self.power_in_time = []
        self.water_in_time = []
        self.metal_in_time = []
        self.ore_in_time = []
        self.ore_processed_in_time = []
        self.ice_in_time = []
        self.ice_processed_in_time = []
        self.chargers = []
        self.ice_hubs = []
        self.ore_hubs = []
        self.heavy_ice_hubs = {}
        self.heavy_ore_hubs = {}

        self.units = []
        self.heavies = []
        self.lights = []
        self._lichen_tree = None
        self.rubble_rss_targets = None
        self._lichen_rubble_points = None
        self._transport_rubble_points = None
        self._assault_rubble_points = None
        self._visit_rubble_points = None
        self._closest_tile_no_chargers = {}
        self._lichen_lake_size = None
        self._n_lichen_tiles_div_unit_count_div_dist = {"LIGHT": None, "HEAVY": None}
        self.area_target_grid = {}
        self.rubble_priorities = {}
        self._own_units_in_range = None
        self._power_hub_positions = None

        self.n_connected_tiles = 0
        self.n_connected_x_plus = 0

        self.has_uncleared_oreroute = False
        self.hub_units = []
        self.power_hubs = []

    def init_power_in_time(self):
        power = self.power
        lichen_tiles = (
            self.n_connected_x_plus
        )  # for power only consider more than X lichen
        charge_rate = self.env_cfg.FACTORY_CHARGE
        lichen_power = self.env_cfg.POWER_PER_CONNECTED_LICHEN_TILE

        for _ in range(TTMAX + 1):
            self.power_in_time.append(power)

            # for now assume lichen will stay the same
            power += lichen_tiles * lichen_power
            power += charge_rate
        # if self.id == 1:
        #     lprint(self, "self.power_in_time", self.power_in_time)

    def _init(self, game_board):
        self.reset()
        self.game_board = game_board

        # initialize power and water
        water = self.cargo.water
        metal = self.cargo.metal

        water_consumption = self.env_cfg.FACTORY_WATER_CONSUMPTION

        for t in range(TTMAX + 1):
            self.water_in_time.append(water)
            self.metal_in_time.append(metal)
            self.ore_in_time.append(0 if t > 0 else self.cargo.ore)
            self.ice_in_time.append(0 if t > 0 else self.cargo.ice)
            self.ore_processed_in_time.append(0)
            self.ice_processed_in_time.append(0)
            water -= water_consumption
        # lprint(self.unit_id, "self.metal_in_time", self.metal_in_time)
        # add ore and ice, set at t=1 since t=0 wont process
        if self.cargo.ice > 0:
            self.update_resource(1, 0, self.cargo.ice)
        if self.cargo.ore > 0:
            self.update_resource(1, 1, self.cargo.ore)

        self.center = game_board.get_point(self.pos)
        self.center.factory_center = self
        self.center.unit = None

        points = [self.center] + self.edge_tiles()
        for p in points:
            p.factory = self
            if abs(p.x - self.center.x) == 1 and abs(p.y - self.center.y) == 1:
                p.is_factory_corner = True

        self.points = points
        self.tuple_points = tuple(points)

        adjacent_points = []
        for p in points:
            adjacent_points += p.adjacent_points()

        self.adjacent_points = list(set(adjacent_points) - set(points))

        game_state = game_board.game_state
        self.lichen = np.sum(
            game_state.board.lichen
            * (game_state.board.lichen_strains == self.strain_id)
        )
        self.has_lichen = self.lichen > 0
        self.n_growth_tiles = None
        self.n_connected_tiles = None

        # add reference to factory
        for p in self.adjacent_points:
            if p.ice:
                self.ice_hubs.append(p)
            if p.ore:
                self.ore_hubs.append(p)

            for move in MOVES:
                if p.apply(move) in self.points:
                    p.adjacent_factories[self.unit_id] = move
                    if self.is_enemy:
                        p.next_to_enemy_factory = True
        self.closest_tiles = {}

        self.ore_only = self.ore_hubs and not self.ice_hubs
        self.double_hub = (len(self.ore_hubs) > 0) and (len(self.ice_hubs) > 0)
        self.power_hub_push = self.ore_only or (
            self.double_hub
            # and (
            #     len([p.ice or p.ore or p.rubble > 0 for p in self.adjacent_points])
            #     >= 9  # out of 12
            #     or self.available_power() < 1000
            # )
            and np.sum(
                game_state.board.lichen_strains[game_state.board.lichen > 0] == self.id
            )
            == 0
            and game_board.step < 300
        )

    def avg_nett_water_production(self):
        """Average water production per turn, already removes water consumption"""
        water_in_time = self.water_in_time
        return (water_in_time[-1] - water_in_time[0]) / len(water_in_time)

    def clear_rubble_for_lichen(self):
        """Clear rubble to make space for lichen"""
        return clear_rubble_for_lichen(self)

    @property
    def n_power_hubs(self):
        return len([u for u in self.heavies if u.is_power_hub])

    @property
    def n_heavies(self):
        return len(self.heavies)

    @property
    def n_lights(self):
        return len(self.lights)

    @property
    def n_lights_exclude(self):
        return len(
            [
                u
                for u in self.lights
                if u.point.closest_own_factory_distance < 15 and not u.is_shield
            ]
        )

    def closest_tile(self, point):
        if point not in self.closest_tiles:
            self.closest_tiles[point] = min(self.points, key=point.distance_to)
        return self.closest_tiles[point]

    def closest_tile_no_chargers(self, point):
        if point not in self._closest_tile_no_chargers:
            self._closest_tile_no_chargers[point] = min(
                self.edge_tiles(no_chargers=True), key=point.distance_to
            )
        return self._closest_tile_no_chargers[point]

    def has_free_ice_hubs(self):
        return [p for p in self.ice_hubs if not p.unit]

    def has_free_ore_hubs(self):
        return [p for p in self.ore_hubs if not p.unit]

    def set_hubs(self, verbose=False):
        return set_hubs(self, verbose=verbose)

    def out_of_water_time(self):
        return next((i for i, w in enumerate(self.water_in_time) if w <= 0), TTMAX)

    def build_heavy_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST

    def build_heavy_power_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST

    def can_build_heavy(self, game_state):
        return self.power >= self.build_heavy_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_heavy_metal_cost(game_state)

    def build_heavy(self):
        return 1

    def build_light_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.METAL_COST

    def build_light_power_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.POWER_COST

    def can_build_light(self, game_state):
        return self.power >= self.build_light_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_light_metal_cost(game_state)

    def build_light(self):
        return 0

    # for now use TMAX due to difficulty in predicting future lichen production
    def available_power(self, t_max=TMAX):
        return max(0, min(self.power_in_time[:t_max]))

    def available_water(self, t_max=TMAX):
        return max(0, min(self.water_in_time[:t_max]))

    def available_metal(self, t_max=TMAX):
        return max(0, min(self.metal_in_time[:t_max]))

    def owned_lichen_tiles(self, game_state):
        return (game_state.board.lichen_strains == self.strain_id).sum()

    def rubble_border_tiles(self):
        return set(
            [
                ap
                for ap in flatten(
                    [p.adjacent_points() for p in self.lichen_points + self.points]
                )
                if ap.rubble
            ]
        )

    def water_cost(self):
        """
        Water required to perform water action
        """
        n_growth_tiles = self.n_growth_tiles
        return self.water_cost_tiles(n_growth_tiles)

    def water_cost_tiles(self, n_growth_tiles):
        return int(np.ceil(n_growth_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR))

    def can_water(self, game_state):
        return self.cargo.water >= self.water_cost(game_state)

    @property
    def pos_slice(self):
        return slice(self.pos[0] - 1, self.pos[0] + 2), slice(
            self.pos[1] - 1, self.pos[1] + 2
        )

    def water(self):
        return 2

    def edge_tiles(self, unit=None, no_chargers=False):
        """
        Returns the tiles surrounding the center
        """
        return edge_tiles(self, unit=unit, no_chargers=no_chargers)

    def neighbour_tiles(self):
        if self._neighbour_tiles is None:
            neighbour_tiles = []
            for p in self.edge_tiles():
                neighbour_tiles += p.adjacent_points()
            neighbour_tiles = [p for p in neighbour_tiles if not p.factory]
            self._neighbour_tiles = neighbour_tiles
        return self._neighbour_tiles

    def can_spawn(self, game_board):
        """Returns True if this factory can spawn a new unit
        Can spawn if point available next tick
        """
        free_1 = len(game_board.unit_grid_in_time[1][self.center]) == 0
        free_2 = len(game_board.unit_grid_in_time[2][self.center]) == 0
        return free_1 and free_2

    def update_resource(self, t, resource_id, amount):
        # resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        if t >= TTMAX:
            return

        # ice
        if resource_id == 0:
            amount += self.ice_in_time[t]
            water_ratio = self.env_cfg.ICE_WATER_RATIO
            max_processed = self.env_cfg.FACTORY_PROCESSING_RATE_WATER
            total_water = 0
        # ore
        elif resource_id == 1:
            amount += self.ore_in_time[t]
            metal_ratio = self.env_cfg.ORE_METAL_RATIO
            max_processed = self.env_cfg.FACTORY_PROCESSING_RATE_METAL
            total_metal = 0

        for t2 in range(t, TTMAX + 1):
            # don't care about ice to water conversion, just assume it will be converted
            if resource_id == 0:
                processed = self.ice_processed_in_time[t2]

                left_to_process = max_processed - processed
                assert IS_KAGGLE or left_to_process >= 0 and amount >= 0
                converted = min(amount, left_to_process) // water_ratio
                ice_processed = converted * water_ratio
                amount -= ice_processed
                total_water += converted

                self.water_in_time[t2] += total_water
                self.ice_in_time[t2] = amount
                self.ice_processed_in_time[t2] += ice_processed
            elif resource_id == 1:
                processed = self.ore_processed_in_time[t2]

                left_to_process = max_processed - processed
                assert IS_KAGGLE or left_to_process >= 0 and amount >= 0
                converted = min(amount, left_to_process) // metal_ratio
                ore_processed = converted * metal_ratio
                amount -= ore_processed
                total_metal += converted

                self.metal_in_time[t2] += total_metal
                self.ore_in_time[t2] = amount
                self.ore_processed_in_time[t2] += ore_processed
            elif resource_id == 2:
                self.water_in_time[t2] += amount
            elif resource_id == 3:
                self.metal_in_time[t2] += amount

    def transfer_power(self, t, amount_picked_up):
        # if self.id == 1:
        #     lprint(f"Transfer power {amount_picked_up} from {self} at {t}")
        for t2 in range(t, TTMAX + 1):
            # pickup of power happens before regrowth, for bookkeeping
            # we therefore subtract it from lux.the turn before
            if amount_picked_up > 0:
                self.power_in_time[t2 - 1] -= amount_picked_up
            else:
                self.power_in_time[t2] -= amount_picked_up

    def nearby_heavy_enemies(self, distance=4):
        enemies = []
        for point in self.points_within_distance(distance):
            enemy_unit = point.enemy_unit
            if enemy_unit and enemy_unit.is_heavy:
                enemies.append(enemy_unit)
        return enemies

    def nearby_heavies(self, distance=4):
        return [u for u in self.heavies if u.point.distance_to(self) <= distance]

    def dangerous_enemy_near(self, distance=4, exclude_chase=False):
        """Check if enemy is near"""
        for enemy in self.nearby_heavy_enemies(distance):
            # don't be afraid if there is an enemy factory
            if any(
                ap.factory.is_enemy
                for ap in enemy.point.adjacent_points()
                if ap.factory
            ):
                continue

            # if any(
            #     ap.unit.is_own and ap.unit.is_heavy and ap.unit.power > enemy.power
            #     for ap in enemy.point.adjacent_points()
            #     if ap.unit
            # ):
            #     continue

            if (
                enemy.is_digging
                and not enemy.digs_lichen
                and enemy.point.closest_factory.is_enemy
            ):
                continue

            return True

        return False

    def __repr__(self) -> str:
        return f"Factory({self.unit_id}, {self.center.xy})"

    def distance_to(self, other):
        """Distance to another point or factory, returns the minimum distance"""
        if isinstance(other, list):
            other = tuple(other)

        if other not in self._distance_to:
            if isinstance(other, Point):
                self._distance_to[other] = other.min_distance_to(self.tuple_points)
            elif isinstance(other, tuple):
                self._distance_to[other] = min(o.distance_to(self) for o in other)
            elif isinstance(other, Factory):
                self._distance_to[other] = min(
                    tile.min_distance_to(self.tuple_points) for tile in other.points
                )
            else:
                self._distance_to[other] = other.distance_to(self)

        return self._distance_to[other]

    def points_within_distance(self, distance):
        """Returns all points within a given distance"""
        if distance not in self._points_within_distance:
            points = []
            for point in self.edge_tiles():
                points += point.points_within_distance(distance)
            self._points_within_distance[distance] = list(set(points))
        return self._points_within_distance[distance]

    def get_lichen_tree(self):
        """Returns a lichen tree"""
        if self._lichen_tree is None:
            self.build_lichen_tree()
        return self._lichen_tree

    def filter_growth_points(self, points, visited=None, skip_rubble=False):
        return filter_growth_points(self, points, visited, skip_rubble)

    def next_growth_points(self, points):
        """Returns a list of points that will grow lichen next turn"""
        return set(flatten([p.adjacent_points() for p in points]))

    def build_lichen_tree(self, depth=0, visited=None):
        return build_lichen_tree(self, depth, visited)

    def simulate_lichen_growth(self):
        """Simulate lichen growth"""
        return simulate_lichen_growth(self)

    def has_enough_water_to(self, purpose):
        return has_enough_water_to(self, purpose)

    def get_rubble_rss_targets(self):
        """Returns the rubble points that are on the path to the rss points
        stop searching if enough found"""
        max_points = 5
        if self.rubble_rss_targets is None:
            points = []

            ore_points = self.ore_points
            ice_points = self.ice_points

            # if enough rubble cleared for lichen, no need to clear rss more
            max_rubble = 19 if self.lake_large_enough() else 0
            n = 1
            while True:
                rps = []
                if len(ore_points) >= n:
                    rps.append(ore_points[n - 1])
                if len(ice_points) >= n:
                    rps.append(ice_points[n - 1])
                if not rps:
                    break
                n += 1

                candidates_current = []
                for rp in rps:
                    rp_distance = rp.distance_to(self)

                    # add rp surrounding points to allow crowd to travel fast as well
                    candidates_current += [
                        p
                        for p in rp.surrounding_points()
                        if p.distance_to(self) <= rp_distance
                    ]

                    for factory_point in self.points:
                        candidates = get_rss_optimal_path(
                            self.game_board, factory_point, rp
                        )

                        candidates_current += candidates

                # prune the set
                candidates_current = [
                    c
                    for c in set(candidates_current)
                    if c.rubble > max_rubble
                    and c.get_rubble_at_time(TTMAX) > max_rubble
                    and c not in rps
                    and not c.next_to_enemy_factory
                    and c not in points
                ]

                points = points + candidates_current
                if len(points) > max_points:
                    break

            self.rubble_rss_targets = points

        return self.rubble_rss_targets

    def get_lichen_rubble_points(self):
        """Returns the rubble points that are on the boarder of this factory's lichen lake"""
        if self._lichen_rubble_points is None:
            is_limit_points = not self.has_enough_water_to("spread_max")

            def get_boundary_points(mask):
                board = gb.game_state.board
                clusters[~mask] = 0
                boundaries = find_boundaries(clusters, mode="outer")
                boundaries[
                    mask
                ] = False  # somehow this method considers cluster points itself as boundary
                boundaries[
                    board.ice + board.ore + board.factory_occupancy_map > -1
                ] = False

                points = []
                for x, y in np.asarray(np.where(boundaries)).T:
                    points.append(gb.get_point((x, y)))
                return points

            gb = self.game_board
            clusters = gb.lichen_clusters.copy()

            has_lichen = self.n_connected_tiles == 0
            candidates = get_boundary_points(clusters == self.lichen_lake_id)
            candidates = set(candidates + self.get_lichen_connector_rubble())

            def prune_limit(p):
                if not is_limit_points:
                    return True

                distance = p.distance_to(self)
                if distance <= 15 and p.rubble < 20:
                    return True

                if distance <= 6 and p.rubble < 60 if has_lichen else 60:
                    return True

                return distance <= 3 and p.rubble < 80 if has_lichen else 60

            # filtering
            points = [
                p
                for p in candidates
                if p.closest_factory.is_own
                and p.rubble > 0
                and p.get_rubble_at_time(TTMAX) > 0
                and prune_limit(p)
            ]
            self._lichen_rubble_points = points

        return self._lichen_rubble_points

    def get_visit_rubble_points(self):
        """Returns the rubble points that are visited often by units"""
        # now not an argument as it would break cache
        distance = 8
        threshold = 0.8
        max_points = 10
        min_visits = 5
        if self._visit_rubble_points is None:
            reference_points = [
                p
                for p in self.points_within_distance(distance)
                if not p.factory
                and not p.ice
                and not p.ore
                and not p.next_to_enemy_factory
                and p.rubble > 0
            ]

            if not reference_points:
                self._visit_rubble_points = []
                return self._visit_rubble_points
            visits = [p.visit_count for p in reference_points]
            threshold = max(np.quantile(visits, threshold), min_visits)

            points = [
                p
                for p in reference_points
                if p.visit_count > threshold
                and p.rubble > MAX_FREE_LIGHT_MOVEMENT
                and p.get_rubble_at_time(TTMAX) > MAX_FREE_LIGHT_MOVEMENT
            ]

            self._visit_rubble_points = sorted(
                points, key=lambda p: p.visit_count, reverse=True
            )[:max_points]

        return self._visit_rubble_points

    def get_transport_rubble_points(self):
        """Returns the rubble points that are on the path to closest friendly factory"""
        return get_transport_rubble_points(self)

    def get_assault_rubble_points(self):
        """Returns the rubble points that are on the path to closest enemy factory"""
        return get_assault_rubble_points(self)

    def is_crowded(self, max_factor=0.4, verbose=False):
        return is_crowded(self, max_factor=max_factor, verbose=verbose)

    def get_lichen_connector_rubble(self):
        return get_lichen_connector_rubble(self)

    def get_lichen_lake_size(self):
        if self._lichen_lake_size is None:
            self.build_lichen_tree()
        return self._lichen_lake_size

    def supported_lichen_tiles(self, n_hubs=None):
        return supported_lichen_tiles(self, n_hubs=n_hubs)

    def get_power_hub_positions(self, factory_only=False, verbose=False):
        return get_power_hub_positions(self, factory_only=factory_only, verbose=verbose)

    def lake_large_enough(self):
        return lake_large_enough(self)

    def water_for_power(self, verbose=False):
        return water_for_power(self, verbose=verbose)

    def n_lichen_tiles_div_unit_count_div_dist(self, unit, verbose=False):
        return n_lichen_tiles_div_unit_count_div_dist(self, unit, verbose=verbose)

    def is_ore_risk(self):
        return is_ore_risk(self)

    def is_ice_risk(self):
        return not self.ice_hubs

    def power_production(self):
        return (
            50
            + self.n_connected_tiles
            + 6 * len(self.power_hubs)
            + sum([u.avg_power_production_per_tick() for u in self.chargers])
        )

    def power_consumption(self):
        return sum([u.power_required_per_tick() for u in self.hub_units])

    def own_units_in_range(self):
        return own_units_in_range(self)

    def replan_power_hub(self, unit):
        PowerHub(unit).execute()

    def consider_lichen_assault(self, unit=None):
        return consider_lichen_assault(self, unit=None)

    @property
    def n_power_hub_positions(self):
        return len(self.get_power_hub_positions())

    @property
    def n_charger_positions(self):
        return len(
            set(
                [
                    self.closest_tile(p)
                    for p in list(self.heavy_ice_hubs.keys())
                    + list(self.heavy_ore_hubs.keys())
                ]
            )
        )

    def full_power_hub(self):
        capacity = self.n_power_hub_positions + self.n_charger_positions
        current = self.n_power_hubs + len([u for u in self.chargers if u.is_heavy])

        return current >= capacity

    def heavies_for_full_power_hub(self):
        capacity = self.n_power_hub_positions + self.n_charger_positions
        current = self.n_power_hubs + len([u for u in self.chargers if u.is_heavy])

        return (
            capacity
            - current
            - len(
                [
                    u
                    for u in self.heavies
                    if u.point.distance_to(self) <= 1 and u.can_be_replanned()
                ]
            )
            <= 0
        )
