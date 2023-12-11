from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np

from lux.constants import (
    COORD_MAP,
    DIRECTION_ACTION_MAP,
    DIRECTION_MAP,
    MOVES,
    NUM_FEATURES,
    TTMAX,
)


class Point:
    def __init__(
        self,
        x: int,
        y: int,
        rubble: float,
        ice: bool,
        ore: bool,
        lichen: int,
        lichen_strains: int,
        field,
    ):
        self._x = x
        self._y = y
        self._field = field
        self.xy = (x, y)
        self.is_factory = False
        self.ice = ice
        self.ore = ore
        self.rubble = rubble
        self.id = x * field._size + y

        self.lichen = lichen
        self.lichen_strains = lichen_strains
        self.distances = {}
        self._min_distance_to = {}
        self._directions = None
        self._direction = {}
        self._directions_to = {}
        self._points_within_distance = {}
        self.visit_count = 0
        self._adjacent_points = None
        self._surrounding_points = None
        self._d2_grid = None
        self._points_at_distance = {}

        self.adjacent_factories = {}
        self.next_to_enemy_factory = False
        self.factory = None
        self.factory_center = None
        self.is_factory_corner = False

        self.reset()

        self.apply = self._apply
        self.distance_penalty = self._distance_penalty
        self.nearby_points = self._nearby_points
        self.closest_factory = None
        self.closest_own_factory = None
        self.closest_enemy_factory = None

        self.closest_factory_distance = 1000
        self.closest_own_factory_distance = 1000
        self.closest_enemy_factory_distance = 1000

        self._apply_cache = {}

        # self.apply = lru_cache(maxsize=None)(self._apply)
        # self.distance_to lux.= lru_cache(maxsize=None)(self._distance_to)
        # self.distance_penalty = lru_cache(maxsize=None)(self._distance_penalty)
        # self.points_at_distance = lru_cache(maxsize=None)(self._cahce)
        # self.nearby_points = lru_cache(maxsize=None)(self._nearby_points)

    def get_rubble_at_time(self, t):
        return self._rubble_in_time[t] if t in self._rubble_in_time else self.rubble

    def rss_within_distance(self, rss_type: str, distance: int = 8):
        points = []
        points_in_range = self.points_within_distance(distance)
        is_ice = rss_type == "ice"
        is_ore = rss_type == "ore"
        is_rubble = rss_type == "rubble"
        for p in points_in_range:
            if is_ice and p.ice:
                points.append(p)
            elif is_ore and p.ore:
                points.append(p)
            elif is_rubble and p.rubble > 0:
                points.append(p)
        return points

    def reset(self, reset_factories=True):
        """Used for resetting state on a new turn"""
        self._rubble_in_time = defaultdict(int)

        self.unit = None
        self.my_unit = None
        self.enemy_unit = None
        self.visited_by = []
        self.dug_by = []
        self.is_hub_candidate = False
        self.rubble_target_value = 100
        self.choke_kills = 0
        self.own_lichen = False
        self.enemy_lichen = False
        self.tried_times = 0
        self.dug_within_2 = False

        if reset_factories:
            self.adjacent_factories = {}
            self.factory = None
            self.factory_center = None
            self.next_to_enemy_factory = False

    def __repr__(self):
        return f"Point({self._x}, {self._y})"

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def to_tuple(self) -> Tuple[int, int]:
        return self._x, self._y

    def __lt__(self, other):
        return self.id < other.id

    @property
    def field(self):
        return self._field

    def _apply(self, move: Union[int, str]) -> "Point":
        if move not in self._apply_cache:

            if isinstance(move, tuple):
                pass
            else:
                move = DIRECTION_ACTION_MAP[move]

            new_x = self.x + move[0]
            new_y = self.y + move[1]
            if 0 <= new_x < self.field.size and 0 <= new_y < self.field.size:
                self._apply_cache[move] = self.field[new_x, new_y]
            else:
                self._apply_cache[move] = self
        return self._apply_cache[move]

    def distance_to(self, point: "Point") -> int:
        if not isinstance(point, Point):
            return point.distance_to(self)

        if point in self.distances:
            return self.distances[point]
        distance = abs(self.x - point.x) + abs(self.y - point.y)
        self.distances[point] = distance
        return distance

    def get_direction(self, next_point):
        """# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)"""
        if next_point not in self._direction:
            if self == next_point:
                self._direction[next_point] = 0
            elif next_point.x > self.x:
                self._direction[next_point] = 2
            elif next_point.x < self.x:
                self._direction[next_point] = 4
            elif next_point.y > self.y:
                self._direction[next_point] = 3  # lower is higher y!
            else:
                self._direction[next_point] = 1
        return self._direction[next_point]

    def min_distance_to(self, destinations):
        if destinations in self._min_distance_to:
            return self._min_distance_to[destinations]

        min_distance = 100000
        for destination in destinations:
            distance = destination.distance_to(self)
            if distance < min_distance:
                min_distance = distance
        self._min_distance_to[destinations] = min_distance
        return min_distance

    def _distance_penalty(self, point: "Point") -> int:
        if self == point:
            return 1000000
        d = self.distance_to(point)
        d_pen = 1 / ((2 - d / 4) ** 1.1 if d < 5 else 1 / (d - 4) ** (1.5)) / 0.54
        is_same_line = self.x == point.x or self.y == point.y
        return d_pen if is_same_line else d_pen * 1.25

    # @cached_property
    def adjacent_points(self) -> List["Point"]:
        if self._adjacent_points is None:
            points = [self.apply(a) for a in MOVES]
            points = [p for p in points if p != self]
            self._adjacent_points = list(set(points))
        return self._adjacent_points

    def directions(self, add_wait=True):
        if self._directions is None:
            directions = []
            for a in MOVES:
                point = self.apply(a)
                if point != self:
                    directions.append(a)
            self._directions = directions

        if add_wait:
            return [0] + self._directions
        return self._directions

    def get_directions(self, destination):
        if self == destination:
            return [0]

        if destination in self._directions_to:
            return self._directions_to[destination]

        dx = destination.x - self.x
        dy = destination.y - self.y
        dirs = self.get_directions_xy(dx, dy)

        self._directions_to[destination] = dirs
        return dirs

    def get_directions_xy(self, dx, dy):
        dirs = []

        if dx == 0:
            pass
        elif dx > 0:
            dirs.append(DIRECTION_MAP["right"])
        elif dx < 0:
            dirs.append(DIRECTION_MAP["left"])

        # note that y starts at top with 0
        if dy == 0:
            pass
        elif dy > 0:
            dirs.append(DIRECTION_MAP["down"])
        elif dy < 0:
            dirs.append(DIRECTION_MAP["up"])
        return dirs

    def surrounding_points(self) -> List["Point"]:
        """Returns all points surrounding this point, including diagonals."""
        if self._surrounding_points is None:
            tiles = []
            size = self.field.size
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x == 0 and y == 0:
                        continue
                    new_x = self.x + x
                    new_y = self.y + y
                    if 0 <= new_x < size and 0 <= new_y < size:
                        tiles.append((new_x, new_y))
            self._surrounding_points = [self.field[tuple(t)] for t in tiles]
        return self._surrounding_points

    def d2_grid(self) -> List["Point"]:
        if self._d2_grid is None:
            tiles = []
            max_xy = self.field.size - 1

            for coord in COORD_MAP.keys():
                new_xy = np.array(self.xy) + np.array(coord)
                new_x = new_xy[0]
                new_y = new_xy[1]
                if new_x < 0 or new_x > max_xy:
                    continue
                if new_y < 0 or new_y > max_xy:
                    continue

                tiles.append((coord, new_xy))
            self._d2_grid = [(c, self.field[tuple(t)]) for c, t in tiles]
        return self._d2_grid

    # @cached_property
    def row(self) -> List["Point"]:
        return list(self._field.points[:, self.y])

    # @cached_property
    def column(self) -> List["Point"]:
        return list(self._field.points[self.x, :])

    def points_within_distance(self, distance: int) -> List["Point"]:
        if distance not in self._points_within_distance:
            points = []
            for p in self._field:
                if self.distance_to(p) <= distance:
                    points.append(p)
            self._points_within_distance[distance] = points
        return self._points_within_distance[distance]

    def points_at_distance(self, r: int) -> List["Point"]:
        if r not in self._points_at_distance:
            if r > 1:
                points = []
                for p in self._field:
                    distance = self.distance_to(p)
                    if distance == r:
                        points.append(p)
                self._points_at_distance[r] = points
            elif r == 1:
                self._points_at_distance[r] = self.adjacent_points()
            elif r == 0:
                self._points_at_distance[r] = [self]
        return self._points_at_distance[r]

    def _nearby_points(self, r: int) -> List["Point"]:
        if r > 1:
            points = []
            for p in self._field:
                distance = self.distance_to(p)
                if 0 < distance <= r:
                    points.append(p)
            return points
        elif r == 1:
            return self.adjacent_points

        raise ValueError("Radius must be more or equal then 1")

    def update_rubble_in_time(self, t, decrease):
        # dig is applied before moving --> but I will never move onto the tile before the dig..
        # so rubble effect must be applied to the same turn t
        effective_decrease = decrease

        if not self._rubble_in_time:
            # cell has never been updated
            rubble_now = self.rubble
            effective_decrease = min(decrease, rubble_now)
            if effective_decrease == 0:
                return

            new_rubble = rubble_now - effective_decrease
            for t2 in range(t, TTMAX + 1):
                self._rubble_in_time[t2] = new_rubble
            return

        # cell was updated
        min_t = min(self._rubble_in_time.keys())
        if t >= min_t:
            for t2 in range(t, TTMAX + 1):
                rubble_now = self._rubble_in_time[t2]

                effective_decrease = min(decrease, rubble_now)
                if effective_decrease == 0:
                    return

                self._rubble_in_time[t2] = rubble_now - effective_decrease
            return

        # rubble different after t
        for t2 in range(t, TTMAX + 1):
            rubble_now = (
                self._rubble_in_time[t2] if t2 in self._rubble_in_time else self.rubble
            )
            effective_decrease = min(decrease, rubble_now)

            self._rubble_in_time[t2] = rubble_now - effective_decrease

    def build_features(self):
        grid = self.d2_grid()
        my_unit = self.unit
        if not my_unit:
            return

        my_team_id = my_unit.team_id

        # features = {}
        array = np.full(400, -2)

        for _, (coord, p) in enumerate(grid):
            pidx = COORD_MAP[coord] * NUM_FEATURES
            # note, perhaps remove redundant features (degrees of freedom)
            # add distance to closest own factory?
            # add flag for unit or not?
            unit = p.unit

            # point_dict = {
            #     "rubble": p.rubble,
            #     "ice": p.ice,
            #     "ore": p.ore,
            # }
            array[pidx : (pidx + 3)] = [p.rubble, p.ice, p.ore]

            if unit:
                # point_dict["heavy"] = unit.is_heavy
                # point_dict["power"] = unit.power
                # point_dict["my_team"] = unit.team_id == my_team_id
                # point_dict["cargo_value"] = unit.cargo_value
                # point_dict["distance_factory"] = unit.factory.distance_to(unit.point)
                # point_dict[
                #     "distance_opponent_factory"
                # ] = unit.closest_opponent_factory.distance_to(unit.point)

                array[(pidx + 3) : (pidx + 9)] = [
                    unit.is_heavy,
                    unit.power,
                    unit.is_own,
                    unit.cargo_value,
                    unit.factory.distance_to(unit.point),
                    unit.closest_opponent_factory.distance_to(unit.point),
                ]

                action_queue = unit.action_queue

                if len(action_queue) == 0:
                    pass  # point_dict["no_action"] = True
                else:
                    action = action_queue[0]
                    is_move = action[0] == 0
                    is_dig = action[0] == 3
                    # point_dict["is_move"] = is_move
                    # point_dict["is_dig"] = is_dig
                    # point_dict["other_action"] = not is_move and not is_dig
                    array[(pidx + 9) : (pidx + 12)] = [
                        is_move,
                        is_dig,
                        not (is_move or is_dig),
                    ]

                    if is_move:
                        # point_dict["is_up"] = action[1] == 1
                        # point_dict["is_down"] = action[1] == 2
                        # point_dict["is_left"] = action[1] == 3
                        # point_dict["is_right"] = action[1] == 4
                        # point_dict["is_wait"] = action[1] == 0
                        array[(pidx + 12) : (pidx + 17)] = [
                            action[1] == 1,
                            action[1] == 2,
                            action[1] == 3,
                            action[1] == 4,
                            action[1] == 0,
                        ]

            if p.factory:
                # point_dict["is_factory"] = True
                # point_dict["my_team"] = p.factory.team_id == my_team_id
                # point_dict["factory_power"] = p.factory.power

                array[(pidx + 17) : (pidx + 19)] = [
                    True,
                    p.factory.power,
                ]
                array[pidx + 5] = p.factory.is_own

            # features[coord] = point_dict

        return array
