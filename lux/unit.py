import math
import sys
from collections import defaultdict
from typing import List

import numpy as np

# from lux.factory import Factory
from lux.board import Point
from lux.cargo import UnitCargo
from lux.constants import (
    BASE_COMBAT_HEAVY,
    BASE_COMBAT_LIGHT,
    HEAVY_DIG_EFFICIENCY,
    IS_KAGGLE,
    MIN_EFFICIENCY,
    OPPOSITE_DIRECTION,
    RECHARGE,
    TMAX,
    TTMAX,
    VALIDATE,
)
from lux.router import power_required
from lux.utils import lprint, set_inverse_zero

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


class Unit:
    def __init__(self, unit_cfg=None, env_cfg=None, **f_data) -> None:
        self.team_id = f_data["team_id"]
        self.unit_id = f_data["unit_id"]
        self.unit_type = f_data["unit_type"]
        self.pos = np.array(f_data["pos"])
        self.point: Point = None
        self.power = f_data["power"]
        self.cargo = UnitCargo(**f_data["cargo"])
        self.env_cfg = env_cfg
        self.unit_cfg = unit_cfg
        self.base_move_cost = unit_cfg.MOVE_COST
        self.charge = unit_cfg.CHARGE
        self.action_queue = np.array(f_data["action_queue"])
        self.start_action_queue = self.action_queue.copy()
        action_queue = self.action_queue

        self.dig_cost = self.unit_cfg.DIG_COST
        self.cargo_space = self.unit_cfg.CARGO_SPACE
        self.collides = False
        self.battery_capacity = self.unit_cfg.BATTERY_CAPACITY
        self.full_power = self.power > self.battery_capacity * 0.9
        self.closest_opponent_factory = None
        self.init_power = self.unit_cfg.INIT_POWER
        self.dies = False
        self.kills = []
        self.kills_next = []
        self.dies_by = []
        self.could_die = False
        self.died_at_start_turn = False
        self.dies_by_own = False
        self.is_heavy = self.unit_type == "HEAVY"
        self.is_light = self.unit_type == "LIGHT"
        self.is_own = None
        self.is_enemy = None
        self.must_retreat = False
        self.under_attack_dies_at_t = False
        self.try_replan = False
        self.illegal_move = False
        self.illegal_pickup = False
        self.needs_escape = False
        self.routed_by = []
        self.visits_factory_point = (
            None  # True if starts outside factory and visits it on the way
        )
        self.visits_factory_time = (
            -1
        )  # True if starts outside factory and visits it on the way
        self.direct_path_to_factory = False

        self.illegal_lichen_dig = False
        self.illegal_self_destruct = False
        self.id = int("".join([s for s in self.unit_id if s.isdigit()]))
        self.digs_nothing = False
        self.total_cargo = (
            self.cargo.ice + self.cargo.ore + self.cargo.water + self.cargo.metal
        )
        has_queue = len(action_queue) > 0

        self.is_repeating = has_queue and action_queue[-1][-2] > 0
        self.is_digging = has_queue and any(action_queue[:, 0] == 3)
        self.is_digging_next = has_queue and action_queue[0][0] == 3
        # self.first_dig = 1000
        self.is_retreating = self.ends_recharge(RECHARGE.RETREAT)
        self.is_defending = self.ends_recharge(RECHARGE.DEFEND)

        self.rubble_factor = self.unit_cfg.RUBBLE_MOVEMENT_COST
        self.is_pipeline = False
        self.digs_own_lichen = False
        self.digs_lichen = False
        self.digs_ore = False
        self.skip_adjacent_attack = False
        self.digs_ice = False
        self.self_destructs = False
        self.is_power_hub = (
            has_queue
            and any([a[0] == 0 and a[1] == 0 and a[-2] == 20 for a in action_queue])
            and any([a[0] == 1 and a[2] == 4 and a[-2] == 30 for a in action_queue])
            # and action_queue[-1][0] == 1
            # and action_queue[-1][2] == 4
            # and action_queue[-1][-2] == 1
            # and action_queue[-1][3] == self.avg_power_production_per_tick()
        )
        self.was_power_hub = self.is_power_hub

        self.digs = defaultdict(int)
        self.is_hub = self.is_repeating and self.is_digging  # or self.is_power_hub
        self.dig_frequency = (
            max(a[-2] for a in action_queue if a[0] == 3) if self.is_hub else 0
        )
        self.is_power_hub_push_ice = (
            self.is_hub
            and self.dig_frequency
            and any([a[0] == 0 and a[1] == 0 and a[-2] > 0 for a in action_queue])
        )
        self.is_charger = (
            self.is_repeating
            and not self.is_power_hub
            and any(
                [
                    a[2] == 4
                    for a in action_queue
                    if a[0] == 1 and a[1] != 0 and a[-2] == 1
                ]
            )
            # and any([a[2] == 4 for a in action_queue if a[0] == 2])
        )

        self.charges_units = []
        self.targeted_by = []

        self.is_charging = len(action_queue) == 1 and (
            action_queue[0][0] == 5 or (action_queue[0][0] == 0 and self.is_repeating)
        )
        self.is_attacking = (
            self.ends_chase()
            or self.ends_pingpong()
            or self.ends_recharge(RECHARGE.ATTACK_NO_FOLLOW)
            or self.ends_attack_bots()
            or False
        )
        self.lower_power_hub = False
        self.accept_attack_risk = False
        self.is_moving = False
        self.is_moving_next = False
        self.next_point = None
        self.next_point_observation = None  # used to track what opponent could think
        self.has_cargo = self.total_cargo > 0
        self.factory = None
        self.has_been_reset = False
        self.game_board = None
        self.must_move = False
        self.tried_actions = []
        self.is_ambushing = has_queue and action_queue[-1][3] == RECHARGE.AMBUSH
        self.was_ambushing = self.is_ambushing
        self.is_killnet = self.ends_recharge(RECHARGE.KILLNET)
        self.is_guard = self.ends_recharge(RECHARGE.GUARD)
        self.is_shield = self.ends_recharge(
            RECHARGE.LIGHT_SHIELD if self.is_light else RECHARGE.HEAVY_SHIELD
        )
        self.is_repeating_last_move = (
            len(action_queue) == 1
            and action_queue[0][-2] > 0
            and action_queue[0][0] == 0
        )
        self.power_in_time = np.zeros(TTMAX + 1)
        self.power_in_time[0] = self.power

        self._distance_to: dict = {}
        self._combat_strength_at_t_point: dict = {}
        self._move_cost_rubble: dict = {}

        self.tried_targets = []
        self.tried_target_bots = []
        self.path = []
        self.is_replanned = False
        self.base_strength = BASE_COMBAT_HEAVY if self.is_heavy else BASE_COMBAT_LIGHT
        self.attacks_bot = None
        self.needs_final_attack_action = False
        self.is_targeted = False
        self.target_gone = False
        self.keeps_moving_or_factory = True

        self._max_return_distance = {}
        self._target_grid_at_distance = {}

        self.charges_at_factory = (
            has_queue
            and action_queue[-1][0] == 2
            and action_queue[-1][2] == 4
            and action_queue[-1][-2] == 0
        )
        self.planned_action = None

    def ends_attack_bots(self):
        if len(self.action_queue) == 0:
            return False
        last_action = self.action_queue[-1]
        if (
            last_action[0] == 0
            and last_action[1] != 0
            and self.action_queue[-1][-2] > 1
        ):
            return True

    def ends_recharge(self, amount):
        return (
            len(self.action_queue) > 0
            and self.action_queue[-1][0] == 5
            and self.action_queue[-1][3] == amount
        )

    @property
    def has_queue(self):
        return len(self.action_queue) > 0

    @property
    def is_corner_charger(self):
        return (
            self.is_charger
            and self.point.is_factory_corner
            and self.point == self.last_point
        )

    @property
    def charge_rate_now(self):
        return self.game_board.charge_rate[0]

    def combat_strength(self, has_moved, power_left=None):
        """Determine combat strenght at time t when moving onto point"""
        strength = self.base_strength  # base strength
        if power_left is None:
            power_left = self.power
        if has_moved:
            # since stationary bot doesn't use power, moving wins, + 1 to win even if power_left == 0
            strength += power_left + 1
        return strength

    # def combat_strength_at_t_point(self, t, point):
    #     """Determine combat strenght at time t when moving onto point"""
    #     key = (t, point)
    #     if key not in self._combat_strength_at_t_point:

    #         self._combat_strength_at_t_point[key] = self.combat_strength(
    #             has_moved=True, power_left=self.power_in_time[t]
    #         )

    #     return self._combat_strength_at_t_point[key]

    def enough_power_or_close(
        self, target_point=None, min_power=0.85, target_distance=8, factory_distance=8
    ):
        if self.available_power() > self.battery_capacity * min_power:
            return True

        if target_point is None:
            return self.point.distance_to(self.factory) > factory_distance
        return (
            self.point.distance_to(target_point) <= target_distance
            and self.point.distance_to(self.factory) > factory_distance
        )

    def is_other_team(self, obj):
        return self.team_id != obj.team_id

    def is_same_team(self, obj):
        return self.team_id == obj.team_id

    def combat_strength_move(
        self,
        t: int = 0,
        has_moved: bool = True,
        is_next_turn: bool = False,
        current_point: Point = None,
        target_point: Point = None,
        start_power: int = None,
    ):
        """Determine combat strenght at time t when moving onto point"""
        if is_next_turn:
            # unit moves onto target tile
            assert (
                IS_KAGGLE or target_point != current_point
            ), "should not be possible to move onto same tile"

            power_to_move = (
                self.move_power_cost(t, current_point, target_point) + self.comm_cost
            )
            if power_to_move <= start_power:
                power_left = start_power - power_to_move
                return self.combat_strength(has_moved=has_moved, power_left=power_left)
            else:
                # unit cannot move onto tile.
                return -1

        return self.combat_strength(has_moved=has_moved, power_left=start_power)

    @property
    def value(self):
        """Convert unit to power value, reference is light digging efficiency"""
        power = self.power
        ore = (
            self.cargo.metal + self.unit_cfg.METAL_COST
        ) * self.env_cfg.ORE_METAL_RATIO + self.cargo.ore
        ice = self.cargo.water * self.env_cfg.ICE_WATER_RATIO + self.cargo.ice

        light_dict = self.env_cfg.ROBOTS["LIGHT"]
        light_dig_factor = light_dict.DIG_COST / light_dict.DIG_RESOURCE_GAIN

        return power + ore * light_dig_factor + ice * light_dig_factor

    @property
    def cargo_value(self):
        """Convert unit to power value, reference is light digging efficiency"""
        ore = (self.cargo.metal) * self.env_cfg.ORE_METAL_RATIO + self.cargo.ore
        ice = self.cargo.water * self.env_cfg.ICE_WATER_RATIO + self.cargo.ice

        light_dict = self.env_cfg.ROBOTS["LIGHT"]
        light_dig_factor = light_dict.DIG_COST / light_dict.DIG_RESOURCE_GAIN

        return ore * light_dig_factor + ice * light_dig_factor

    def __repr__(self) -> str:
        return f"Unit({self.team_id}:{self.unit_id}, {self.unit_type}, {self.point.xy}, PWR={self.power})"

    @property
    def can_move(self):
        """Whether the unit can move this turn"""
        rubble = min([p.rubble for p in self.point.adjacent_points()])

        add_comm = True
        if len(self.action_queue) > 0:
            action = self.action_queue[0]
            if action[0] == 0 and action[1] != 0 and self.next_point.rubble <= rubble:
                add_comm = False

        power_cost = self.move_cost_rubble(rubble) + self.comm_cost * add_comm

        return power_cost <= self.power

    def __ge__(self, other):
        return self.unit_type <= other.unit_type  # HEAVY < LIGHT SO INVERSE

    def __gt__(self, other):
        return self.unit_type < other.unit_type  # HEAVY < LIGHT SO INVERSE

    def action_queue_cost(self, game_state):
        cost = self.env_cfg.ROBOTS[self.unit_type].ACTION_QUEUE_POWER_COST
        return cost

    @property
    def agent_id(self):
        if self.team_id == 0:
            return "player_0"
        return "player_1"

    def move_power_cost(self, t, start: Point, end: Point):
        if start == end:
            return 0

        rubble = end.get_rubble_at_time(t)
        return math.floor(self.base_move_cost + self.rubble_factor * rubble)
        # return math.floor(
        #     self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble
        # )

    def move_cost(self, game_state, direction):
        assert False, "should not use this"
        board = game_state.board
        target_pos = self.pos + move_deltas[direction]
        if (
            target_pos[0] < 0
            or target_pos[1] < 0
            or target_pos[1] >= len(board.rubble)
            or target_pos[0] >= len(board.rubble[0])
        ):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if (
            factory_there not in game_state.teams[self.agent_id].factory_strains
            and factory_there != -1
        ):
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return None
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        return math.floor(
            self.unit_cfg.MOVE_COST
            + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
        )

    def move_cost_rubble(self, rubble=0):
        if rubble not in self._move_cost_rubble:
            self._move_cost_rubble[rubble] = math.floor(
                self.unit_cfg.MOVE_COST + rubble * self.unit_cfg.RUBBLE_MOVEMENT_COST
            )
        return self._move_cost_rubble[rubble]

    @property
    def comm_cost(self):
        return self.unit_cfg.ACTION_QUEUE_POWER_COST

    def move(self, direction, repeat=0, n=1):
        # if not isinstance(direction, int):
        #     direction = direction
        # else:
        #     pass
        return np.array([0, int(direction), 0, 0, int(repeat), n])

    def transfer(
        self, transfer_direction, transfer_resource, transfer_amount, repeat=0, n=1
    ):
        assert transfer_resource < 5 and transfer_resource >= 0
        assert transfer_direction < 5 and transfer_direction >= 0

        assert (
            IS_KAGGLE or transfer_amount > 0
        ), "!!!!!!!!Warning, tried to transfer 0 or less resources"
        return np.array(
            [
                1,
                transfer_direction,
                transfer_resource,
                int(transfer_amount),
                int(repeat),
                n,
            ]
        )

    @property
    def transfers(self):
        return self.has_queue and any(self.action_queue[:, 0] == 1)

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        assert pickup_resource < 5 and pickup_resource >= 0
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, n])

    def dig(self, repeat=0, n=1):
        return np.array([3, 0, 0, 0, repeat, n])

    def self_destruct_cost(self):
        return self.unit_cfg.SELF_DESTRUCT_COST

    def self_destruct(self, repeat=0, n=1):
        return np.array([4, 0, 0, 0, repeat, n])

    def recharge(self, x, repeat=0, n=1):
        return np.array([5, 0, 0, x, repeat, n])

    def can_be_replanned(self, close_to_enemy_factory=False):
        if self.try_replan:
            return True

        if self.is_replanned:
            return False

        if len(self.action_queue) == 0:
            return True

        steps_left = self.game_board.steps_left
        if self.action_queue[0][0] == 5 and self.action_queue[0][3] in [
            RECHARGE.REPLAN,
            RECHARGE.REPLAN_HEAVY,
        ]:
            if (
                self.is_heavy
                or self.point.lichen
                or (self.power > self.init_power or self.power > steps_left)
            ):
                return True
            if self.is_light and self.point.closest_own_factory_distance < 15:
                return True

        if len(self.action_queue) == 1 and self.ends_recharge(
            RECHARGE.ATTACK_NO_FOLLOW
        ):
            return True

        if close_to_enemy_factory:

            if (
                self.action_queue[0][0] == 5
                and self.action_queue[0][3] in [RECHARGE.CLOSE_FACTORY]
                and (self.power > self.init_power or self.power > steps_left)
            ):
                return True

            if steps_left < 50:
                if (self.is_killnet or self.is_ambushing) and len(
                    self.action_queue
                ) == 1:
                    return True
                if steps_left < 20 and self.is_shield:
                    return True
        return False

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.point}"
        return out

    def fuel_max(self, actions):
        """Fuel with max available power from lux.factory"""

        current = self.power

        if current >= self.unit_cfg.BATTERY_CAPACITY - self.charge_rate_now:
            return False

        solar_charge = self.charge_rate_now
        capacity = self.unit_cfg.BATTERY_CAPACITY - solar_charge

        factory = self.point.factory

        power_available = min(factory.power_in_time[:TMAX])

        if power_available > self.init_power // 5:

            to_charge = min(capacity - current, power_available)
            action = self.pickup(4, to_charge, repeat=False)
            lprint(
                f"{self.unit_id}: fuelling {to_charge} power at {factory.unit_id} at {self.point} (current power: {current})",
                file=sys.stderr,
            )
            factory.transfer_power(0, to_charge)

            self.action_queue = [action]
            actions[self.unit_id] = [action]
            return True
        else:
            lprint(
                f"{self.unit_id} with {self.power}: Not a lot of power at {factory.unit_id}"
                f" ({power_available}), not wasting power charging",
                file=sys.stderr,
            )
            return False

    def recharge_replan(self):
        return self.recharge(
            RECHARGE.REPLAN if self.is_light else RECHARGE.REPLAN_HEAVY
        )

    def charging_units(self):
        if not self.is_hub:
            return []
        adj_points = self.point.adjacent_points()
        charging_units = [p.unit for p in adj_points if p.unit and p.unit.is_charger]
        if not charging_units:
            charging_units = [
                u
                for u in self.game_board.agent.units
                if u.is_charger and u.last_point in adj_points
            ]
        return charging_units

    def unplan(self, reason):
        """Unplan all actions for this unit"""
        if len(self.action_queue) == 0:
            return

        if (
            not self.is_moving_next
            and self.comm_cost + self.base_move_cost > self.power
        ):
            lprint(f"{self}:Not enough power to unplan", self.action_queue)
            return

        is_hub = self.is_hub
        self.has_been_reset = True
        factory = self.factory

        self.is_attacking = False
        self.is_defending = False
        self.is_ambushing = False
        self.is_digging = False
        self.next_point = None

        if self.attacks_bot:
            enemy = self.attacks_bot
            if self in enemy.targeted_by:
                enemy.targeted_by.remove(self)

            if not enemy.targeted_by:
                enemy.is_targeted = False
            if self in enemy.dies_by:
                enemy.dies_by.remove(self)
            if not enemy.dies_by:
                enemy.dies = False
            self.attacks_bot = None

        if self.is_charger:
            if self in self.factory.chargers:
                self.factory.chargers.remove(self)

        self.is_charger = False

        # as factory is set to the last point, we need to reset it to the closest point
        self.factory = min(
            self.game_board.agent.factories,
            key=lambda f: (
                power_required(self, self.point, f.center),
                self.point.distance_to(f),
            ),
        )

        agent = self.game_board.agent
        lprint(f"Unit {self} unplanned: {reason}")

        agent.actions[self.unit_id] = [self.recharge_replan()]

        point = self.point
        if self in point.dug_by:
            if point.dug_by == [self] or point.ice or point.ore:
                point._rubble_in_time = defaultdict(int)
            else:
                lprint("Warning, unit unplanned but not only unit digging")

        if self.is_moving_next and not self.is_charger:
            self.is_moving_next = False
            next_units_at_my_position = self.game_board.unit_grid_in_time[1][point]

            assert (
                IS_KAGGLE or self not in next_units_at_my_position
            ), f"{self} should not be possible."
            if len(next_units_at_my_position) == 0:
                self.dies = False
                self.dies_by_own = False

        for t, action in enumerate(self.action_queue):
            if action[0] == 2:  # pickup
                if action[2] == 4:  # power
                    lprint(
                        f"Unplanned {self.unit_id}: returning {action[3]} power to {factory.unit_id}"
                    )
                    # -1 since pickups were accounted for in the previous turn
                    factory.transfer_power(t - 1, -action[3])
            if action[0] == 1 and self.charges_units:  # transfer
                if action[2] == 4:  # power
                    target_unit = point.apply(action[1]).unit
                    if target_unit:
                        lprint(
                            f"Unplanned {self.unit_id}: claiming {action[3]} power back from {target_unit}"
                        )
                        target_unit.power_in_time[t:] = (
                            target_unit.power_in_time[t:] - action[3]
                        )
        self.charges_units = []
        self.action_queue = []

        if is_hub:
            charging_units = self.charging_units()
            for u in charging_units:
                u.unplan("charging hub")
            if point in factory.heavy_ice_hubs:
                del factory.heavy_ice_hubs[point]
            if point in factory.heavy_ore_hubs:
                del factory.heavy_ore_hubs[point]

        unit_grid_in_time = self.game_board.unit_grid_in_time
        path_len = len(self.path)
        for t, p in enumerate(self.path):
            if t > 0 and t <= TTMAX + 1:
                if self in p.dug_by:
                    p.dug_by.remove(self)
                if self in p.visited_by:
                    p.visited_by.remove(self)

                if t > TTMAX:
                    continue
                unit_grid = unit_grid_in_time[t]
                if IS_KAGGLE or t >= path_len - 1:
                    if p not in unit_grid or self not in unit_grid[p]:
                        continue
                else:
                    assert (
                        p in unit_grid
                    ), f"{t}: {p} not in unit_grid[t] path_len:{path_len}"
                    assert (
                        self in unit_grid[p]
                    ), f"{t}: {self} not in {unit_grid[t]} path_len:{path_len}"

                unit_grid[p].remove(self)
        self.path = [self.point]
        if self not in self.point.visited_by:
            self.point.visited_by.append(self)
        self.is_hub = False
        for u in self.kills:
            if self in u.dies_by:
                u.dies_by.remove(self)
            if not u.dies_by:
                u.dies = False
        self.kills = []
        self.kills_next = []

        # check safety of staying in this position
        units_at_position = unit_grid_in_time[1][point]
        attack_risk = self.game_board.attack_risk_in_time[1][point]
        combat_strength = self.combat_strength(has_moved=False, power_left=self.power)

        if len(units_at_position) > 0:
            if any(u.is_own for u in units_at_position):
                self.must_move = True
        else:
            self.dies = False
            self.dies_by_own = False
            self.could_die = attack_risk >= combat_strength

        # check risk of staying in this position
        if attack_risk >= combat_strength:
            self.dies = True
            self.dies_by_own = False
            self.could_die = True
            self.must_move = True

        self.add_to_work_queue()

        if VALIDATE:
            ledger = agent.ledger
            if self.unit_id in ledger:
                del ledger[self.unit_id]

    def add_to_work_queue(self):
        agent = self.game_board.agent
        if self not in agent.units_to_be_replanned:
            agent.units_to_be_replanned.append(self)
        if self not in agent.units_to_consider:
            agent.units_to_consider.append(self)

    def avg_power_production_per_tick(self):
        day = self.env_cfg.DAY_LENGTH
        cycle = self.env_cfg.CYCLE_LENGTH
        unit_cfg = self.unit_cfg
        production_per_tick = day / cycle * unit_cfg.CHARGE
        return production_per_tick

    def available_power_lichen_digs(self):
        return (
            self.available_power(self.point.closest_own_factory_distance)
            - self.dig_cost * 2
            - self.comm_cost * 2
        )

    def near_enemy_factory(self):
        return (
            self.point.closest_enemy_factory_distance
            < self.point.closest_own_factory_distance
            and (
                self.point.closest_enemy_factory_distance <= 6
                and self.point.closest_own_factory_distance > 6
            )
            #   or (self.point.closest_enemy_factory_distance <= 1)
        )

    def near_my_factory(self):
        return self.point.distance_to(self.factory) < 6

    def charge_from_to(self, t1, t2):
        return sum(self.game_board.charge_rate[t1:t2]) * self.unit_cfg.CHARGE

    def charge_up_to(self, t):
        return self.game_board.charge_up_to(self.is_heavy, t)

    def consider_attack_distance(self, target):
        """For distance"""
        if target.dies:
            return False

        if self < target:
            return False

        if self > target:
            return False

        point = self.point
        distance = point.distance_to(target.point)

        charge = self.charge_up_to(distance)
        move_cost = self.game_board.agent.p2p_power_cost[self.unit_type][point.id][
            target.point.id
        ]

        power_in_factory = point.factory.available_power() if point.factory else 0

        can_reach = (
            self.power + charge + power_in_factory
            >= move_cost
            + self.comm_cost
            + target.power
            + (self.comm_cost + self.base_move_cost)
        )

        if self.near_my_factory() and can_reach:
            return True

        if (
            can_reach
            and self.power > target.power * 1.25
            and target.factory.distance_to(target.point) > 5
        ):
            return True

        if target.can_move and not target.is_digging_next:
            return False

        return True

    def tried(self, action):
        """Check if action has been tried"""
        return action.name() in self.tried_actions

    def consider_lichen_attack(self):
        if self.game_board.steps_left < 100:
            return True

        point = self.point
        enemy_distance = point.closest_enemy_factory_distance
        if (
            point.closest_own_factory_distance < enemy_distance
            and not self.near_enemy_factory()
        ):
            return self.power + self.factory.available_power() >= self.battery_capacity

        return True

    # def distance_to(self, other):
    #     """Distance to another point or factory, returns the minimum distance"""
    #     if isinstance(other, list):
    #         other = tuple(other)

    #     if other not in self._distance_to:
    #         if isinstance(other, (Point, Factory)):
    #             self._distance_to[other] = other.distance_to(self.point)
    #         elif isinstance(other, tuple):
    #             self._distance_to[other] = min(o.distance_to(self) for o in other)
    #         else:
    #             raise ValueError(f"Unknown type {type(other)}")

    #     return self._distance_to[other]

    def points_within_distance(self, distance):
        """Points within distance"""
        return self.point.points_within_distance(distance)

    def cost_to_clear_lichen(self, point: Point):
        unit_point = self.point

        move_cost = (
            (unit_point.distance_to(point) - 1) * self.base_move_cost
            + self.move_cost_rubble(point.rubble)
            if point != unit_point
            else 0
        )

        dig_cost = (
            math.ceil(point.rubble / self.unit_cfg.DIG_RUBBLE_REMOVED) * self.dig_cost
        )

        return move_cost + dig_cost

    def max_return_distance(
        self, start_charge=None, min_efficiency=MIN_EFFICIENCY, verbose=False
    ):
        """Max distance to return while preserving efficiency"""
        key = (start_charge, min_efficiency)
        # if verbose:
        #     self._max_return_distance = {}
        if key not in self._max_return_distance:
            power = self.power

            if verbose:
                lprint(
                    f"Min efficiency: {min_efficiency}",
                    "MIN_EFFICIENCY",
                    MIN_EFFICIENCY,
                )

            factory = self.point.factory
            if start_charge or factory:
                if factory is None:
                    factory = self.factory
                power += factory.available_power()
                power = min(self.battery_capacity, power)

            if verbose:
                lprint(f"Available power: {power}")
            # charge = 3 / 5 * self.charge

            comm_cost = 2 * self.comm_cost

            if self.is_heavy:
                # charge_multiplier = 12
                charge = 6
                move_multiplier = 40
                turn_multiplier = 28
                dig_type_eff = HEAVY_DIG_EFFICIENCY
            else:
                # charge_multiplier = 1.2
                move_multiplier = 2
                turn_multiplier = 0.8
                dig_type_eff = 1
                charge = 0.6

            dig_turn_cost = self.dig_cost - charge

            max_d = 0
            for d in range(1, 30):
                move_cost = d * move_multiplier  # 2 * d * self.base_move_cost
                nett_move_cost = d * turn_multiplier + comm_cost
                # move_charges = d * charge_multiplier #2 * d * charge
                remaining_power = power - nett_move_cost
                # move_cost - comm_cost + move_charges
                n_digs = remaining_power // dig_turn_cost

                if n_digs <= 0:
                    break
                total_dig_cost = n_digs * self.dig_cost

                efficiency = (move_cost + comm_cost + total_dig_cost) / total_dig_cost

                ref_efficiency = dig_type_eff * efficiency
                if verbose:
                    lprint(
                        f"Distance {d}: {round(ref_efficiency, 3)}<={min_efficiency}"
                        f"  {move_cost} + {comm_cost} + {total_dig_cost} / {total_dig_cost}"
                    )
                if ref_efficiency <= min_efficiency:
                    max_d = d
                else:
                    break
            steps_left = self.game_board.steps_left
            self._max_return_distance[key] = min(max_d, steps_left - 3)
        return self._max_return_distance[key]

    def ends_chase(self):
        """Check if unit ends chase"""
        if len(self.action_queue) < 1:
            return False
        return (
            self.action_queue[-1][0] == 0
            and self.action_queue[-1][1] != 0
            and self.action_queue[-1][-2] > 0
            and self.action_queue[-1][-2] < 8000
        )

    def ends_pingpong(self):
        if len(self.action_queue) < 2:
            return False
        return (
            self.action_queue[-1][0] == 0
            and self.action_queue[-2][0] == 0
            and self.action_queue[-2][-2] > 0
            and self.action_queue[-1][-2] > 0
            and self.action_queue[-1][1] != 0
            and self.action_queue[-2][1] != 0
            and self.action_queue[-1][1] == OPPOSITE_DIRECTION[self.action_queue[-2][1]]
        )

    def starts_pingpong_attack(self):
        if len(self.path) < 3:
            return False

        target_point = self.next_point
        point = self.point

        return (
            target_point != point
            and self.path[2] == point
            and target_point.unit
            and (target_point.unit.team_id != self.team_id)
        )

    def digs_disconnected_lichen(self):
        if not self.is_digging_next or self.point.lichen == 0:
            return False

        return self.game_board.agent.connected_lichen_map[self.point.xy] == -1

    def get_cargo(self, resource_id):
        """Get cargo"""
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)

        cargo = self.cargo
        if resource_id == 0:
            return cargo.ice
        if resource_id == 1:
            return cargo.ore
        if resource_id == 2:
            return cargo.water
        if resource_id == 3:
            return cargo.metal
        if resource_id == 4:
            return self.power
        raise ValueError(f"Unknown resource id {resource_id}")

    def available_power(self, include_factory=False):
        return (
            min(self.power + self.factory.available_power(), self.battery_capacity)
            if include_factory or self.point.factory
            else self.power
        )

    # def required_from_to_via(self, from_point, to_point, via, verbose=False):

    def required_from_to(self, from_point, to_point, verbose=False):
        distance = from_point.distance_to(to_point)
        charged = self.charge_up_to(distance)
        power_needed = int(power_required(self, from_point, to_point) + self.comm_cost)
        if verbose:
            lprint(
                f">>{self}: required_from_to {from_point} --> {to_point} (d={distance}): available=charged {charged} + power {self.power} > {power_needed}"
            )

        return max(0, power_needed - charged)

    def required_to_get_home(self, verbose=False):
        center = self.factory.center

        return self.required_from_to(self.point, center, verbose=verbose)

    def go_home_power_left(self, verbose=False):
        return self.power - self.required_to_get_home(verbose=verbose)

    def remove_from_work_queue(self):
        # lprint(f"Removing {self} from work queue")
        agent = self.game_board.agent
        if self in agent.units_to_be_replanned:
            agent.units_to_be_replanned.remove(self)
        if self in agent.units_to_consider:
            agent.units_to_consider.remove(self)

    def is_target_covered(self, target, verbose=False):
        point = self.point

        covered = any(
            [
                len(
                    [
                        u
                        for u in ap.visited_by
                        if u.is_own
                        and u.last_point == ap
                        and u != self
                        and u.is_heavy == self.is_heavy
                        and u.path.index(ap) <= point.distance_to(ap) + 1
                    ]
                )
                > 0
                for ap in target.adjacent_points() + [target]
            ]
        )
        return covered

    def power_required_per_tick(self):
        if not self.is_hub:
            return 0

        production_per_tick = self.avg_power_production_per_tick()
        assert IS_KAGGLE or self.is_hub
        n_digs = [a for a in self.action_queue if a[0] == 3][-1][-2]
        waits = [a for a in self.action_queue if a[0] == 0 and a[-2] > 0]
        n_waits = waits[0][-2] if waits else 0

        consumption_per_tick = n_digs * self.dig_cost / (n_digs + n_waits + 1)
        required_per_tick = consumption_per_tick - production_per_tick
        return required_per_tick

    def dig_lichen(self):
        gb = self.game_board
        if gb.steps_left < 100:
            return True
        point = self.point

        enemy_more_units = (
            gb.agent.n_opp_heavies > gb.agent.n_heavies
            if self.is_heavy
            else gb.agent.n_opp_lights > gb.agent.n_lights
        )

        return (
            len(gb.agent.get_lichen_targets()) > 0
            and self.factory.n_heavies > 1
            and (not self.factory.power_hub_push or self.factory.full_power_hub())
            and not enemy_more_units
            and (
                self.available_power() > self.battery_capacity * 0.95
                or (
                    not point.factory
                    and (
                        (
                            self.power > self.battery_capacity * 0.5
                            and (
                                point.closest_own_factory_distance > 8
                                and point.closest_enemy_factory_distance < 6
                            )
                        )
                        or (
                            any(p.enemy_lichen for p in point.points_within_distance(2))
                            and self.power > self.battery_capacity * 0.33
                        )
                    )
                )
            )
        )

    def current_move_prevents_dig_from(self):
        next_point = self.next_point
        if next_point is None:
            return False

        prevents_dig_from = [
            ap.dug_by[0]
            for ap in next_point.adjacent_points()
            if ap.dug_within_2
            and ap.dug_by
            and ap.dug_by[0].is_enemy
            and not ap.dug_by[0].is_targeted
        ]
        return prevents_dig_from

    def get_target_grid_at_distance(self, max_distance, min_distance=None):
        key = (max_distance, min_distance)
        if key not in self._target_grid_at_distance:
            target_grid = self.game_board.agent.get_rubble_target_grid()
            area_grid, priorities = set_inverse_zero(
                target_grid.copy(),
                self.point,
                max_distance,
                min_distance=min_distance,
                get_uniques=True,
            )

            priorities = sorted(priorities, reverse=True)

            self._target_grid_at_distance[key] = (area_grid, priorities)

        return self._target_grid_at_distance[key]
