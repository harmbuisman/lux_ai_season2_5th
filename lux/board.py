from collections import defaultdict
from typing import Generator, List, Tuple

import numpy as np
from skimage.measure import label
from skimage.segmentation import expand_labels

from lux.constants import (
    DIRECTION_ACTION_MAP,
    EXECUTION_ORDER,
    IS_KAGGLE,
    MAX_EFFICIENT_LIGHT_DISTANCE,
    MIN_LICHEN_POWER_COUNT,
    POWER_ENEMY_FACTORY_MAGIC_VALUE,
    TTMAX,
    VALIDATE,
    MAX_LOW_POWER_DISTANCE,
)
from lux.globals import _BOARD
from lux.lichen import set_connected_lichen
from lux.point import Point
from lux.router import power_required
from lux.unit import Unit
from lux.utils import lprint

T_UNTIL_DONE = 2  # how many turns without a queue do we still track the position
HEAVY_TRHESHOLD = 50


class TempUnit(Unit):
    def __init__(self, unit: "Unit", player_id: int, point: "Point"):
        self.unit = unit
        self.unit_id = unit.unit_id
        self.point = point
        self.player_id = player_id

        self.action_queue = unit.action_queue.copy()
        self.t_until_done: int = T_UNTIL_DONE
        self.unit_cfg = unit.unit_cfg
        self.env_cfg = unit.env_cfg
        self.unit_type = unit.unit_type
        self.team_id = unit.team_id
        self.moved = False
        self.dies = False
        self.game_board = unit.game_board
        self.power = unit.power
        self.is_heavy = unit.is_heavy
        self.is_light = unit.is_light
        self.is_own = unit.is_own
        self.is_enemy = unit.is_enemy
        self.unit_died_last_turn = False

    def __repr__(self):
        return f"!!!TempUnit({self.unit_id} {self.point})"

    def __str__(self):
        return f"!!!TempUnit({self.unit_id} {self.point})"

    def set_dead(self, units, t):
        other_units = [u.unit for u in units if u.unit_id != self.unit_id]
        # lprint("SET DEAD", t, units)
        # do not use is_own here, as this is also run for enemy units
        dies_by_own = all([u.is_same_team(self) for u in other_units])

        if not dies_by_own and not self.unit.under_attack_dies_at_t:
            self.unit.under_attack_dies_at_t = t

        if t == 1:  # this is die next turn
            self.t_until_done = 0
            self.dies = True
            self.unit.dies_by_own = dies_by_own
            self.unit.dies_by = [u for u in other_units]
            self.unit_died_last_turn = True
            self.unit.dies = True
            self.unit.died_at_start_turn = True
            lprint(
                f"PLAYER {self.player_id}: {self.unit_id} WILL DIE at {self.point} by collision with {other_units} after moving: {self.moved} at t={t} by own: {self.unit.dies_by_own}"
            )

    def t_combat_strength(self, t, is_next_turn=False, target_point=None):
        """Determine combat strenght at time t when moving onto point"""
        return self.unit.combat_strength_move(
            t, self.moved, is_next_turn, self.point, target_point, self.power
        )

    def step(self, t, tunits):
        # get action queue_head
        # extract move
        # append if repeat
        point = self.point
        previous_point = self.point
        factory = point.factory
        unit = self.unit
        agent = unit.game_board.agent

        # step at t=0 brings state for t=1
        power = unit.power_in_time[t] + self.game_board.charge_rate[t] * unit.charge
        power = min(unit.battery_capacity, power)

        if len(self.action_queue) > 0:
            queue = list(self.action_queue).copy()
            action = queue[0].copy()

            if action[-1] == 0 and action[0] != 5:
                assert (
                    IS_KAGGLE
                ), f"t={t} {unit}: ERROR: action queue should not have 0 as last element"

            success = True
            amount = action[3]
            n_repeat = action[-2]
            if unit.is_own and n_repeat >= 8000:
                # this is a hidden lichen dig
                action = unit.dig()
                unit.digs_lichen = True

            if action[0] == 0:
                if (
                    unit.is_own and n_repeat > 1 and action[1] != 0
                ):  # this is part of the fake queue
                    # assert False, "FAKE QUEUE"
                    # could still include in next step in case it is a good action
                    success = t <= 1
                    if t > 1:
                        self.t_until_done = 0
                    target = unit.game_board.get_unit(action[-2])
                    unit.attacks_bot = target
                    if target:
                        target.is_targeted = True
                        if self.unit not in target.targeted_by:
                            target.targeted_by.append(self.unit)
                    else:
                        unit.target_gone = True
                    if t == 0:
                        unit.needs_final_attack_action = True
                if action[1] != 0 and success:
                    target_point = self.point.apply(DIRECTION_ACTION_MAP[action[1]])
                    power_needed = unit.move_power_cost(t, self.point, target_point)

                    if (
                        target_point.factory
                        and target_point.factory.team_id != unit.team_id
                    ) or target_point == point:
                        success = False
                        if t <= 0 and unit.is_own:
                            unit.illegal_move = True

                    if success and power >= power_needed:
                        self.point = target_point
                        power -= power_needed
                        if t == 1 and unit.is_own:
                            self.point.visit_count += 1
                    else:
                        success = False
            elif action[0] == 4:  # SELF-DESTRUCT
                power_needed = unit.unit_cfg.SELF_DESTRUCT_COST
                if power >= power_needed:
                    if t <= 0 and unit.is_own:
                        if not point.lichen or point.own_lichen:
                            unit.illegal_self_destruct = True
                    unit.self_destructs = True
                    power -= power_needed
                    # lprint("SELF-DESTRUCT", unit, t)
                    self.set_dead([], t)
            elif action[0] == 3:  # DIG
                power_needed = unit.unit_cfg.DIG_COST

                if power >= power_needed:
                    dig_amount = self.unit_cfg.DIG_RUBBLE_REMOVED
                    if t <= 0 and unit.is_own:
                        valid_dig = (
                            point.get_rubble_at_time(t) > 0
                            or point.lichen > 0
                            or point.ice
                            or point.ore
                        )
                        if not valid_dig:
                            unit.digs_nothing = True
                    point.update_rubble_in_time(t, dig_amount)

                    if point.ore:
                        unit.digs_ore = True
                    if point.ice:
                        unit.digs_ice = True

                    if (
                        t <= 0
                        and point.rubble > 0
                        and unit.is_own
                        and point.ice == 0
                        and point.ore == 0
                    ):
                        bordering_enemy_lichen_points = [
                            ap
                            for ap in point.adjacent_points()
                            if (ap.factory and ap.factory.is_enemy)
                            or (
                                ap.lichen
                                and ap.lichen_strains in agent.enemy_lichen_strains
                            )
                        ]
                        if len(bordering_enemy_lichen_points) > 0:
                            unit.illegal_lichen_dig = True

                            success = False
                            self.t_until_done = 0

                    if point.lichen > 0:
                        is_own_lichen = point.lichen_strains in agent.own_lichen_strains
                        if self.is_own and is_own_lichen:
                            unit.digs_own_lichen = True

                        unit.digs_lichen = True
                        if (
                            is_own_lichen
                            and unit.is_enemy
                            and point not in agent.lichen_points_under_attack
                        ):
                            # dit is de eerste kans om in te grijpen. later aanpakken is ook OK?
                            agent.lichen_points_under_attack[point] = (t, unit, power)

                    unit.digs[point] += 1
                    # unit.first_dig = min(unit.first_dig, t)
                    if t <= 1:
                        point.dug_within_2 = True
                    if unit not in point.dug_by:
                        point.dug_by.append(unit)

                        # log digging by enemy for attacking purposes
                        if unit.is_enemy and (point.ice or point.ore):
                            if (
                                point.closest_factory.is_own
                                or (
                                    point.closest_own_factory_distance <= 4
                                    and point.closest_enemy_factory_distance > 2
                                )
                            ) and point not in agent.rss_steal:
                                # allow clearing of rubble by the enemy for our tiles
                                if (
                                    point.closest_factory.is_enemy
                                    or point.get_rubble_at_time(t) == 0
                                ):
                                    agent.rss_steal[point] = (t, unit, power)
                else:
                    success = False
                if success:
                    power -= power_needed

                if t <= 1:
                    if unit.is_own and unit.is_repeating:
                        if not success or power < 40:
                            unit.lower_power_hub = True
            elif action[0] == 2:  # pickup

                if not factory:
                    lprint(
                        f"{self.unit_id} t={t} no factory to pickup from at {point}",
                    )
                elif action[2] == 4:  # power
                    assert (
                        IS_KAGGLE or amount > 0 or unit.is_enemy
                    ), f"{unit} at {point}"
                    factory_amount = factory.power_in_time[t]
                    if unit.is_own and t < 40:
                        if factory_amount < amount:
                            if t < 3:
                                lprint(
                                    f"{unit} - {t}: trying to pickup too much power from {factory}(charger started?): {unit.is_charger}"
                                )
                            if not unit.is_charger:
                                if t <= 1:
                                    lprint(
                                        "Invalidating it for picking up too much power"
                                    )
                                    success = False
                                    self.t_until_done = 0

                                    # unit.unplan(
                                    #     "trying to pickup too much power (charger started?)"
                                    # )

                                    unit.illegal_pickup = True
                                # assert (
                                #     IS_KAGGLE or factory_amount >= amount
                                # ), f"Not enough power to pickup from {factory} at {point} at t={t} amount={amount} factory_amount={factory_amount} by {self.unit}"
                        if unit.is_heavy and not unit.is_charger:
                            lprint(
                                f"{unit} HEAVY PICKED UP POWER {point} at t={t} amount={amount} pwr={power} {success}"
                            )
                        if power + amount > unit.battery_capacity:
                            if unit.is_charger:
                                msg = (
                                    f"{unit.unit_id} {point} t={t}: CHARGER "
                                    f"PICKED UP TOO MUCH POWER: PWR {power} + pickup {amount} > {unit.battery_capacity}"
                                )
                                assert (
                                    IS_KAGGLE
                                    or t > 1
                                    or power + amount - unit.battery_capacity < 5
                                ), msg
                            else:
                                msg = f">>>>>>>>>>>>Picking up too much power at {point} at t={t} amount={amount} by {self.unit} pwr={power} cap={unit.battery_capacity}"
                                assert IS_KAGGLE or t > 1, msg
                                lprint(msg)
                    if success:
                        factory.transfer_power(t, amount)
                        power += amount
                        power = min(unit.battery_capacity, power)
                else:
                    picked_up_amount = amount  # todo
                    if picked_up_amount > 0:
                        # t is updated after the step, pickup happens at t + 1
                        factory.update_resource(t + 1, action[2], -picked_up_amount)
            elif action[0] == 1:  # transfer
                direction = action[1]
                target_point = point.apply(DIRECTION_ACTION_MAP[direction])
                factory = target_point.factory

                # if factory and action[-2] >= 1 and unit.is_heavy:
                #     if action[2] == 0:  # ice
                #         factory.heavy_ice_hubs[self.point] = True
                #     if action[2] == 1:  # ore
                #         factory.heavy_ore_hubs[self.point] = True

                if not factory and not point.factory:
                    unit.is_pipeline = True

                if action[2] == 4:  # power
                    if factory:
                        factory.transfer_power(t, -amount)
                    else:
                        # this is probably to a hub.. will need to sort execution order of the unit updates
                        target_units = [
                            u for u in tunits if u.point == target_point and not u.dies
                        ]

                        # if unit.id == 110:
                        # lprint(
                        #     f"{unit} transfer power to",
                        #     target_units,
                        # )

                        if len(target_units) > 0:
                            heavies = [u for u in target_units if u.is_heavy]
                            if heavies:
                                target_units = heavies

                            if (
                                len(
                                    [
                                        tr
                                        for tr in target_units
                                        if not tr.unit.is_attacking
                                    ]
                                )
                                > 1
                            ) and t <= 2:
                                lprint(
                                    f"!!!!! multiple units on {target_point}: {target_units}, not handling transfer power"
                                )
                            else:
                                # only handle transfers to same team.
                                team_targets = [
                                    tu
                                    for tu in target_units
                                    if tu.team_id == unit.team_id
                                ]
                                # if unit.id == 110:
                                #     lprint(
                                #         f"{unit} transfer power to",
                                #         target_units,
                                #         "team_targets",
                                #         team_targets,
                                #     )

                                if team_targets:
                                    # if own units also, transfer to heaviest
                                    target_unit = sorted(
                                        team_targets, key=lambda u: u.is_light
                                    )[0]

                                    # assert (
                                    #     IS_KAGGLE
                                    #     or t >= 1
                                    #     or self.is_enemy
                                    #     or target_unit.team_id == self.team_id
                                    # ), f"trying to transfer power to enemy unit {target_unit} at {target_point} at t={t} amount={amount} by {self.unit} pwr={power}"

                                    target_unit.unit.power_in_time[t:] = (
                                        target_unit.unit.power_in_time[t:] + amount
                                    )
                                    if target_unit.unit not in unit.charges_units:
                                        unit.charges_units.append(target_unit.unit)
                        else:
                            if t <= 1:
                                lprint(
                                    f"{self.unit} t={t} transferring to nothing? to {target_point}"
                                )

                    power -= amount
                    power = max(power, 0)
                else:
                    if factory:

                        resource_id = action[2]
                        transferred_amount = (
                            amount  # min(amount, unit.get_cargo(resource_id))
                        )
                        if transferred_amount > 0:
                            # t is updated after the step, transfer happens at t + 1
                            factory.update_resource(
                                t + 1, resource_id, transferred_amount
                            )
                    else:
                        lprint(
                            f"{self.unit_id} t={t} no factory to transfer to at {point} for direction {direction}"
                        )

            elif action[0] == 5:  # recharge
                # use success flag to set charging complete
                if power < amount:
                    success = False

            self.power = power
            if t < TTMAX:
                unit.power_in_time[t + 1] = power

            if success:
                if action[-1] <= 1:
                    queue.pop(0)
                else:
                    queue[0][-1] -= 1

                # add to queue if repeat or if charging
                if action[-2] >= 1 or action[0] == 5:
                    action[-1] = action[-2]
                    queue.append(action)

            self.action_queue = queue
        else:
            self.power = unit.power_in_time[t]

            unit.power_in_time[t + 1 :] = (
                np.cumsum(self.game_board.charge_rate[t:-1]) * unit.unit_cfg.CHARGE
                + self.power
            )
            if not self.is_heavy:
                # keep heavies on the grid, they often won't move
                self.t_until_done -= 1

        self.moved = previous_point != self.point
        if self.moved:
            unit.is_moving = True
            if t == 0:
                unit.is_moving_next = True
        if t == 0:
            unit.next_point = self.point
            unit.next_point_observation = self.point
        if (
            self.point.factory
            and not unit.point.factory
            and unit.visits_factory_point is None
        ):
            unit.visits_factory_point = self.point
            unit.visits_factory_time = t
            unit.direct_path_to_factory = unit.point.distance_to(point) >= t

        if not unit.visits_factory_point and self.point == previous_point:
            unit.keeps_moving_or_factory = False

        unit.last_point = self.point
        unit.path.append(self.point)

        if unit not in self.point.visited_by:
            self.point.visited_by.append(unit)

        if t < MAX_LOW_POWER_DISTANCE:
            # to prevent chasing units that end with a endless move

            if (
                not unit.is_repeating_last_move
                and unit.is_enemy
                and self.power < 6 * unit.base_move_cost
            ):
                current_point = self.point
                enemy_factory = current_point.closest_enemy_factory
                req = min(
                    [
                        power_required(unit, current_point, tile)
                        for tile in enemy_factory.edge_tiles()
                    ]
                )

                distance = current_point.closest_enemy_factory_distance
                charge = unit.charge_from_to(t, t + distance)

                req = int(req % POWER_ENEMY_FACTORY_MAGIC_VALUE)

                comm_cost = 0 if len(self.action_queue) > 1 else unit.comm_cost

                if self.power + charge < req + comm_cost:
                    if unit not in agent.potential_targets:
                        lprint(
                            f"adding {unit} to potential targets at {t} {current_point}"
                            f"{int(self.power)}+{charge}<{req}+{comm_cost}"
                        )
                        agent.potential_targets[unit] = (
                            t,
                            current_point,
                            self.power,
                            req + comm_cost,
                        )


class Field:
    def __init__(self, size: int):
        self._size = size
        points_2d, points = self.create_array(size)

        self._points = points_2d
        self._points_1d = points
        self.n_points = size * size
        self._point_ids = np.arange(self.n_points)

    def __iter__(self) -> Generator[Point, None, None]:
        for row in self._points:
            yield from row

    def create_array(self, size: int) -> np.ndarray:
        ar2d = np.zeros((size, size), dtype=Point)
        ar1d = np.zeros(size * size, dtype=Point)
        for x in range(size):
            for y in range(size):
                point = Point(
                    x, y, rubble=0, ice=0, ore=0, lichen=0, lichen_strains=0, field=self
                )
                ar2d[x, y] = point
                ar1d[x * size + y] = point
        return ar2d, ar1d

    @property
    def points(self) -> np.ndarray:
        return self._points

    def get_row(self, y: int, start: int, size: int) -> List[Point]:
        if size < 0:
            return self.get_row(y, start=start + size + 1, size=-size)[::-1]

        ps = self._points
        start %= self._size
        out = []
        while size > 0:
            d = list(ps[slice(start, start + size), y])
            size -= len(d)
            start = 0
            out += d
        return out

    def get_column(self, x: int, start: int, size: int) -> List[Point]:
        if size < 0:
            return self.get_column(x, start=start + size + 1, size=-size)[::-1]

        ps = self._points
        start %= self._size
        out = []
        while size > 0:
            d = list(ps[x, slice(start, start + size)])
            size -= len(d)
            start = 0
            out += d
        return out

    @property
    def size(self) -> int:
        return self._size

    def swap(self, dx, dy):
        size = self._size
        if abs(dx) > size / 2:
            dx -= np.sign(dx) * size
        if abs(dy) > size / 2:
            dy -= np.sign(dy) * size
        return dx, dy

    def __getitem__(self, item) -> Point:
        x, y = item
        return self._points[x, y]

    def get_point_by_idx(self, idx: int) -> Point:
        return self._points_1d[idx]

    def get_points_by_idxs(self, idxs: np.ndarray) -> np.ndarray:
        return self._points_1d[idxs]


class GameBoard:
    def __init__(self, game_state, agent=None):
        self.agent = agent
        self.game_state = game_state
        self.board = game_state.board
        self.env_cfg = game_state.env_cfg
        self._step = game_state.real_env_steps
        game_state.game_board = self

        global _BOARD
        if _BOARD is None or self._step == 0:
            _BOARD = Field(self.env_cfg.map_size)
        elif not IS_KAGGLE:
            assert _BOARD.size == self.env_cfg.map_size

        self._field: Field = _BOARD

        for i in range(self.env_cfg.map_size):
            for j in range(self.env_cfg.map_size):
                point = self._field[i, j]
                point.reset(agent.reset_factories)

                # inverse axes
                point.rubble = game_state.board.rubble[i, j]
                point.ice = game_state.board.ice[i, j]
                point.ore = game_state.board.ore[i, j]
                point.lichen = game_state.board.lichen[i, j]
                point.lichen_strains = game_state.board.lichen_strains[i, j]
                strains = point.lichen_strains
                if strains >= 0 and strains in agent.strain_to_factory:
                    factory = agent.strain_to_factory[strains]
                    factory.lichen_points.append(point)

                    if agent.team_id == factory.team_id:
                        point.own_lichen = True
                        point.enemy_lichen = False
                    else:
                        point.own_lichen = False
                        point.enemy_lichen = True

        self._charge_up_to = {False: {}, True: {}}  # Light  # Heavy

        self._players = []

        self.unit_dict = {}
        self.factory_dict = {}
        self.ambush_positions = None

        set_connected_lichen(agent)

        self._init_factories()
        self._init_water_cost()

        for f in self.factories:
            f.init_power_in_time()

        self._init_units()

        self._init_day_schedule()

        if agent.reset_factories:
            self._set_point_closest_factory()

        self._set_unit_grids()
        self._set_unit_factory_relations()

        for f in agent.factories:
            f.set_hubs()

    def get_point(self, pos: Tuple[int, int]) -> Point:
        return self._field.points[pos[0], pos[1]]

    def get_point_by_idx(self, idx: int) -> Point:
        return self._field.get_point_by_idx(idx)

    def get_points_by_idxs(self, idxs: np.ndarray) -> np.ndarray:
        return list(self._field.get_points_by_idxs(idxs))

    def __getitem__(self, item):
        return self._field[item]

    def __iter__(self):
        return self._field.__iter__()

    def is_day(self, step):
        return (
            step % self.game_state.env_cfg.CYCLE_LENGTH
            < self.game_state.env_cfg.DAY_LENGTH
        )

    def charge_up_to(self, is_heavy, t):
        if t not in self._charge_up_to[is_heavy]:
            charged = sum(self.charge_rate[:t]) * (10 if is_heavy else 1)

            self._charge_up_to[is_heavy][t] = charged
        return self._charge_up_to[is_heavy][t]

    @property
    def field(self):
        return self._field

    @property
    def size(self):
        return self._field.size

    @property
    def step(self):
        return self._step

    @property
    def steps_left(self):
        return self.env_cfg.max_episode_length - self._step

    @property
    def units(self) -> List[Unit]:
        return self._units

    def _init_units(self):
        game_state = self.game_state
        self._units = []
        for player in game_state.units.keys():
            is_own = player == self.agent.player
            is_enemy = not is_own
            for unit in game_state.units[player].values():
                point = self.get_point(unit.pos)
                point.unit = unit
                unit.point = point
                unit.path.append(point)
                unit.game_board = self
                self._units.append(unit)
                self.unit_dict[unit.unit_id] = unit
                unit.is_own = is_own
                unit.is_enemy = is_enemy

                if is_enemy:
                    unit.is_charger = False
                    unit.is_hub = False
                    unit.is_power_hub = False

    def _init_factories(self):
        self.factories = []
        game_state = self.game_state

        for player in game_state.factories.keys():
            is_own = player == self.agent.player
            is_enemy = not is_own

            for factory in game_state.factories[player].values():
                factory.is_own = is_own
                factory.is_enemy = is_enemy
                factory._init(self)

                self.factories.append(factory)
                self.factory_dict[factory.unit_id] = factory

        self.agent.enemy_lichen_strains = [f.id for f in self.factories if f.is_enemy]
        self.agent.own_lichen_strains = [f.id for f in self.factories if f.is_own]

        no_ice_hub_factories = [
            f for f in self.factories if f.is_own and len(f.ice_hubs) == 0
        ]

        for f in no_ice_hub_factories:
            other_factories = [
                f2
                for f2 in self.factories
                if f2.is_own and f2 not in no_ice_hub_factories
            ]
            if other_factories:
                supporting_factory = sorted(
                    other_factories, key=lambda f2: f2.distance_to(f)
                )[0]
                supporting_factory.supports_factory = f

    def _init_day_schedule(self):
        # recharge
        max_t = min(self.steps_left - 1, TTMAX)

        charge_rates = [self.is_day(self.step + i) for i in range(max_t + 1)]
        charge_rates = charge_rates + [0] * (TTMAX + 1 - len(charge_rates))
        self.charge_rate = np.array(charge_rates)

    def _set_unit_grids(self, t_max=TTMAX):
        """Build a ledger for all units and their position in time"""
        t_max = min(t_max, self.steps_left - 1)

        unit_grid_in_time = defaultdict(lambda: defaultdict(list))
        attack_risk_in_time = defaultdict(lambda: defaultdict(int))

        # reset state
        tunits = []
        for point in self:
            point.unit = None
            point.my_unit = None
            point.enemy_unit = None
            point.dug_by = []
            point.visited_by = []

        # collect units
        for player in self.game_state.units.keys():
            for unit in self.game_state.units[player].values():
                point = self.get_point(unit.pos)

                tunit = TempUnit(
                    player_id=int(player[-1]),
                    unit=unit,
                    point=point,
                )
                tunits.append(tunit)

                point.unit = unit
                if unit.is_own:
                    point.my_unit = unit
                else:
                    point.enemy_unit = unit

        # step aheid in time
        t: int = 0

        while t < t_max + 1:
            temp_grid = defaultdict(list)
            # at each t:
            # - run actions
            # -

            # units that move should be handled first then transfer, then pickup
            # first heavies, such that when low power at factory pickup the light is replanned
            tunits = sorted(
                tunits,
                key=lambda u: (
                    EXECUTION_ORDER[u.action_queue[0][0]]
                    if len(u.action_queue) > 0
                    else 9999,
                    u.is_light,
                    u.unit_id,
                ),
            )

            unit: TempUnit
            for unit in tunits:
                # assume we will reschedule units when idling. but do not assume for enemy units

                # continue means stop tracking position:
                # - dead units
                # - own units that are light and are marked done
                if unit.dies and not unit.unit_died_last_turn:
                    continue
                if (
                    unit.is_own
                    and unit.is_light
                    and unit.t_until_done <= 0
                    and not unit.unit_died_last_turn
                ):
                    continue

                point = unit.point
                unit_grid_in_time[t][point].append(unit.unit)

                # update attack risk in points
                # point where the unit is now at t
                # adjacent points where the unit can be at t+1
                if unit.is_enemy:
                    # factory positions are safe
                    # own factory safe for entry
                    # enemy factory we cannot be
                    if not point.factory:
                        attack_risk_in_time[t][point] = max(
                            attack_risk_in_time[t][point],
                            unit.t_combat_strength(t),
                        )
                        # unit could decide to not move, meaning non-move risk at t+1
                        attack_risk_in_time[t + 1][point] = max(
                            attack_risk_in_time[t + 1][point],
                            unit.unit.combat_strength(
                                has_moved=False, power_left=unit.power
                            ),
                        )

                    # also update this for a dying unit, since that can try to save itself by moving onto an adjacent position
                    # so not filter on not unit.unit_died_last_turn
                    for p in point.adjacent_points():
                        if not p.factory:  # safe in factory
                            # update next time point, since that is when the unit can be there
                            # use current t for determining combat strength since we care about rubble now
                            attack_risk_in_time[t + 1][p] = max(
                                attack_risk_in_time[t + 1][p],
                                unit.unit.combat_strength_move(
                                    t, True, True, point, p, unit.power
                                ),
                                0,
                            )

                # note position changes here!
                if unit.unit_died_last_turn:
                    # unit is dead, but we needed to keep track of its position in this turn
                    unit.unit_died_last_turn = False
                else:
                    unit.step(t, tunits)
                    temp_grid[unit.point].append(unit)

            t += 1

            if t <= 20:
                for point in temp_grid.keys():
                    units = temp_grid[point]
                    if len(units) > 1:
                        max_str = max([u.t_combat_strength(t) for u in units])
                        surviving = []
                        for u in units:
                            if u.t_combat_strength(t) < max_str:
                                u.set_dead(units, t)
                            else:
                                surviving.append(u)

                        if len(surviving) == 1:
                            survivor = surviving[0]
                            kills = [
                                u.unit for u in units if u.team_id != survivor.team_id
                            ]
                            surviving_unit = survivor.unit
                            surviving_unit.kills = list(
                                set(surviving_unit.kills + kills)
                            )
                            if t == 1:
                                if surviving_unit.kills:
                                    surviving_unit.kills_next = surviving_unit.kills

                            continue

                        still_surviving = []
                        for u in surviving:
                            if not u.moved:
                                u.set_dead(units, t)
                            else:
                                still_surviving.append(u)

                        if len(still_surviving) == 1:
                            survivor = still_surviving[0]
                            kills = [
                                u.unit for u in units if u.team_id != survivor.team_id
                            ]
                            surviving_unit = survivor.unit
                            surviving_unit.kills = list(
                                set(surviving_unit.kills + kills)
                            )
                            if t == 1:
                                if surviving_unit.kills:
                                    surviving_unit.kills_next = surviving_unit.kills

                            continue
                        else:
                            # all units are dead
                            for u in still_surviving:
                                u.set_dead(units, t)

                        units_in_point_str = [
                            f"{u.team_id}: {u.unit_id}"
                            for u in unit_grid_in_time[t][unit.point]
                        ]

                        if VALIDATE:
                            lprint(
                                f"COLLISION DETECTED: t={t} at {point} between "
                                f"and {units_in_point_str}, resetting ledger",
                            )

                            for u in unit_grid_in_time[t][unit.point] + [unit]:
                                if u in self.agent.ledger:
                                    del self.agent.ledger[u.unit_id]

            if t == 1:
                # also in t==1, detect risk
                for unit in tunits:
                    if unit.is_enemy:
                        continue
                    # verkeerde point??

                    if attack_risk_in_time[t][unit.point] > unit.t_combat_strength(t):
                        lprint(
                            f"unit {unit.unit} is in danger {unit.point}",
                        )
                        unit.unit.could_die = True

        self.attack_risk_in_time = attack_risk_in_time
        self.unit_grid_in_time = unit_grid_in_time

    def get_factory(self, factory_id: int):
        return self.factory_dict[f"factory_{factory_id}"]

    def get_unit(self, unit_id: int):
        key = f"unit_{unit_id}"
        if key not in self.unit_dict:
            return None
        return self.unit_dict[key]

    def _set_point_closest_factory(self):
        if not self.factories:
            return

        for point in self:
            point.closest_factory = None
            point.closest_own_factory = None
            point.closest_enemy_factory = None

            point.closest_factory_distance = 1000
            point.closest_own_factory_distance = 1000
            point.closest_enemy_factory_distance = 1000

            # start with enemy factories such that if distance is equal, own is chosen
            for factory in sorted(self.factories, key=lambda x: x.is_own):
                distance = factory.distance_to(point)
                if distance < point.closest_factory_distance:
                    point.closest_factory = factory
                    point.closest_factory_distance = distance

                if factory.is_own:
                    if distance < point.closest_own_factory_distance:
                        point.closest_own_factory = factory
                        point.closest_own_factory_distance = distance
                else:
                    if distance < point.closest_enemy_factory_distance:
                        point.closest_enemy_factory = factory
                        point.closest_enemy_factory_distance = distance

        for factory in self.factories:
            factory.ice_points_by_distance = []
            factory.all_ice_points = []
            factory.ore_points_by_distance = []

        all_ice_points = []
        all_ore_points = []
        for point in self:
            if point.ice:
                point.closest_factory.ice_points_by_distance.append(point)
                all_ice_points.append(point)
            if point.ore:
                point.closest_factory.ore_points_by_distance.append(point)
                all_ore_points.append(point)

        def combined_distance(point, factory):
            return factory.distance_to(point), factory.center.distance_to(point)

        for factory in self.factories:
            factory.ice_points_by_distance.sort(
                key=lambda p: combined_distance(p, factory)
            )
            factory.all_ice_points = sorted(
                all_ice_points, key=lambda p: combined_distance(p, factory)
            )
            factory.ore_points_by_distance.sort(
                key=lambda p: combined_distance(p, factory)
            )
            factory.all_ore_points = sorted(
                all_ore_points, key=lambda p: combined_distance(p, factory)
            )

        for factory in self.factories:
            points_within_distance = factory.points_within_distance(
                MAX_EFFICIENT_LIGHT_DISTANCE
            )
            ice_points = [p for p in points_within_distance if p.ice]
            ore_points = [p for p in points_within_distance if p.ore]

            factory.ice_points = sorted(
                ice_points, key=lambda p: combined_distance(p, factory)
            )
            factory.ore_points = sorted(
                ore_points, key=lambda p: combined_distance(p, factory)
            )

    def _set_unit_factory_relations(self):
        for unit in self.units:

            point = unit.point if unit.is_attacking else unit.last_point

            unit.factory = point.factory

            factories = [f for f in self.factories if f.team_id == unit.team_id]
            if (not unit.factory or unit.factory.team_id != unit.team_id) and factories:
                if unit.is_enemy:
                    unit.factory = min(
                        factories,
                        key=point.distance_to,
                    )
                else:
                    unit.factory = min(
                        factories,
                        key=lambda f: (
                            power_required(unit, unit.point, f.center),
                            unit.point.distance_to(f),
                        ),
                    )

            opponent_factories = [f for f in self.factories if f.is_enemy]
            if opponent_factories:
                unit.closest_opponent_factory = min(
                    opponent_factories, key=point.distance_to
                )

            factory = unit.factory
            if factory:
                factory.units.append(unit)
                if unit.is_light:
                    factory.lights.append(unit)
                else:
                    factory.heavies.append(unit)

        for factory in self.factories:
            for unit in factory.units:
                if unit.is_hub:
                    point = unit.last_point
                    if point.ice:
                        factory.heavy_ice_hubs[point] = unit
                    elif point.ore:
                        factory.heavy_ore_hubs[point] = unit
                    factory.hub_units.append(unit)
                if unit.is_charger:
                    factory.chargers.append(unit)
                if unit.is_power_hub:
                    factory.power_hubs.append(unit)

    def grid_to_ids(self, grid):
        return self._field._point_ids[np.reshape(grid, (self._field.n_points,))]

    def grid_to_points(self, grid):
        return self.get_points_by_idxs(self.grid_to_ids(grid))

    def _init_water_cost(self):
        board = self.game_state.board

        # find strains of lichen
        strains = board.lichen_strains.copy()
        open_pos = board.factory_occupancy_map == -1
        for f in self.factories:
            # if f.power_hub_push:
            #     continue
            for p in f.points:
                strains[p.xy] = f.id

        # find positions wheren multiple lichen could grow (=invalid)
        conflict_map = np.zeros_like(strains)
        for f in self.factories:
            f_growth_map = expand_labels(strains == f.id)
            conflict_map = conflict_map + f_growth_map
        conflict_map = conflict_map > 1

        # determine valid growth positions
        valid_map = (board.ice + board.ore + board.rubble + conflict_map) == 0

        clusters = label(strains, connectivity=1, background=-1)

        # determine growth positions for each factory
        for f in self.factories:
            f.lichen_splash_id = clusters[f.center.xy]

            connected_map = (clusters == f.lichen_splash_id) & open_pos
            #     f.connected_tiles = self.get_points_by_idxs(grid_to_ids(connected_map))

            spreader_cells = np.logical_and(
                connected_map, board.lichen >= self.env_cfg.MIN_LICHEN_TO_SPREAD
            )
            for p in f.points:
                spreader_cells[p.xy] = True

            # not clear to me if after connecting an existing field whether it can also grow
            # let's check by debugging cases
            f.new_map = (valid_map * expand_labels(spreader_cells) * open_pos) > 0

            # need to add existing lichen that does not conform to the growth threshold
            f.new_map = np.logical_or(f.new_map, connected_map)

            f.n_connected_x_plus = np.sum(
                connected_map * (board.lichen >= MIN_LICHEN_POWER_COUNT)
            )
            f.n_connected_tiles = np.sum(connected_map)
            f.min_lichen_value = (
                np.min(board.lichen[connected_map]) if f.n_connected_tiles else 0
            )

            f.n_growth_tiles = np.sum(f.new_map)

            spreaders_no_threshold = np.logical_or(connected_map, spreader_cells)
            expanded = (
                valid_map * expand_labels(spreaders_no_threshold) * open_pos
            ) > 0

            f.n_expand_frontier = np.sum(
                np.logical_and(expanded, np.logical_not(spreaders_no_threshold))
            )
            # new_map could now connect to points of the same strain
            # HANDLE DISCONNECTED TILES
        #     plt.imshow(f.new_map)
        #     plt.show()
