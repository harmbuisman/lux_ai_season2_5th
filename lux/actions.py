import math
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from lux.board import Point
from lux.constants import (
    ACTION,
    BASE_DIG_FREQUENCY_ICE,
    BASE_DIG_FREQUENCY_ORE,
    BREAK_EVEN_ICE_TILES,
    DIGS_TO_TRANSFER_ICE,
    DIGS_TO_TRANSFER_ORE,
    BASE_DIG_FREQUENCY_ORE_PUSH,
    IS_KAGGLE,
    MAX_COMBINE_TARGET_DISTANCE,
    MAX_EFFICIENT_LIGHT_DISTANCE,
    MIN_DONOR_WATER_FACTORY,
    MIN_EFFICIENCY,
    MIN_NORMAL_RUBBLE_PRIORITY,
    OPPOSITE_DIRECTION,
    POWER_THRESHOLD_LICHEN,
    UNPLAN_ORE_OUT_OF_WATER_TIME,
    PRIORITY,
    RECHARGE,
    REMOVE_FINAL_HIT_FROM_QUEUE,
    RESOURCE_MAP,
    STATE,
    STEPS_THRESHOLD_LICHEN,
    TMAX,
    TTMAX,
    WAIT_TIME_ICE,
    HIDE_LICHEN_DIG_DISTANCE,
)
from lux.pathfinder import Pathfinder
from lux.point import Point
from lux.router import (
    commands_from_path,
    get_optimal_path,
    get_rss_optimal_path,
    get_rss_path_queue,
    power_required,
)
from lux.unit import Unit
from lux.utils import flatten, lprint


def _get_directions(point: Point, destination: Point, add_wait: bool = True):
    # todo: add_wait can move to get_directions
    if destination is None:
        return point.directions(add_wait)
    else:
        dirs = point.get_directions(destination)
        if add_wait:
            dirs = dirs + [0]
        return dirs


def get_directions(
    point: Point,
    destinations: List[Point],
    add_wait: bool,
    allow_off_target: bool,
    optimal_dirs: list,
):
    if allow_off_target:
        dirs = point.directions(add_wait)
    else:
        dirs = optimal_dirs
        for destination in destinations:
            dirs = dirs + _get_directions(point, destination, add_wait=add_wait)

        dirs = list(set(dirs))

    return dirs


class Action:
    """Base class for all actions"""

    max_power_pickup = 3000
    resource_id = None
    penalize_move = False
    needs_to_return = True
    reserve_power_to_return = False
    max_steps_off_target = 2
    max_waits = 2
    base_max_it = TMAX
    stop_search_if_complete = False
    factory_as_destinations = False
    may_die_at_destination = False
    needs_targets = False
    post_process_recharge = False
    post_process_queue_len = 1
    min_priority = MIN_NORMAL_RUBBLE_PRIORITY  # used for rubble targets
    max_priority = 1000  # used for rubble targets
    out_of_time_threshold = 0  # used for digging operations that need to return
    can_be_planned_when_retreating = True
    can_route_own_unit_at_t1 = False
    ignore_power_req_end = False
    ignore_power_req_steps_left = 0
    ignore_attack_risk_on_unit_dominance = False
    allow_death_on_lichen = False
    track_tries = False
    disturb_is_good = False
    timed_targets_need_move = True
    beam_size = 40  # number of candidates to keep in beam search
    plan_even_if_not_complete = False

    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()

    def __init__(
        self,
        unit: Unit,
        targets: List[Point] = None,
        destinations: List[Point] = None,
        max_it: int = None,
        start_charge: bool = None,
        verbose: bool = False,
        target_bots: List[Unit] = None,
        allow_low_efficiency: bool = False,
        timed_targets: dict = None,
        debug: bool = False,
        # ignore_risk: bool = False,
        **kwargs,
    ):
        self.unit: Unit = unit
        self.replan_action = False
        self.allow_low_efficiency = allow_low_efficiency

        self.game_board = self.unit.game_board
        self.agent = self.game_board.agent
        self.actions = self.agent.actions
        self.verbose = verbose
        self.target_bots = target_bots
        self.debug = debug
        # self.ignore_risk = ignore_risk
        self.set_start_charge(start_charge)

        assert (
            timed_targets is None or targets is None
        ), "timed_targets and targets are mutually exclusive"
        self.timed_targets = timed_targets
        if timed_targets:
            targets = list(set(flatten(timed_targets.values())))

        self.targets = targets  # used in get_destinations

        if destinations is not None:  # needed for transport max_it
            self.destinations = destinations
        else:
            self.destinations = self.get_destinations(**kwargs)

        self.max_it = self.get_max_it() if max_it is None else max_it
        if self.max_it > TTMAX:
            lprint(f"Warning: max_it too high ({self.max_it}), set to {TTMAX} (TTMAX)")
            self.max_it = TTMAX

        self.targets = targets if self.targets else self.get_targets(**kwargs)

        if verbose:
            lprint(f"{self} targets: {self.targets}")

    def get_destinations(self, verbose=False, **kwargs):
        """Return destinations for action"""
        if not self.needs_to_return and not self.reserve_power_to_return:
            return None

        unit = self.unit
        factory = unit.factory
        point = unit.point

        if self.factory_as_destinations or unit.is_heavy:
            return factory.edge_tiles(unit=unit, no_chargers=True)

        if unit.cargo.metal > 0:
            factory = Metal.get_closest_low_metal_factory(unit)
            if factory:
                return factory.edge_tiles(unit=unit, no_chargers=True)
            else:
                return unit.factory.edge_tiles(unit=unit, no_chargers=True)

        if unit.cargo.water > 0:
            factory = Water.get_closest_low_water_factory(unit)
            if factory:
                return factory.edge_tiles(unit=unit, no_chargers=True)
            else:
                return unit.factory.edge_tiles(unit=unit, no_chargers=True)

        targets = self.targets or self.get_targets(**kwargs)
        reference_points = tuple(targets if targets else [self.unit.point])
        if verbose:
            lprint(f"reference_points: {reference_points}", self)
        distance_to_factory = factory.distance_to(reference_points)

        # send back to own factory if already far a way or if factory has enough power
        if distance_to_factory > 15 or (factory.available_power() >= 3000 - unit.power):
            return factory.edge_tiles(unit=unit, no_chargers=True)

        # consider any close factories
        factory_to_power_distance = {}
        for f in self.agent.factories:
            distance = f.distance_to(reference_points)
            if distance > distance_to_factory + 12:
                continue
            power = f.available_power()

            # don't consider factories that are too far away and have low power
            if f != factory and power < 500:
                continue

            factory_to_power_distance[f] = (
                power * (distance_to_factory + 1) / (distance + 1)
            )

        highest_power_factory = sorted(
            factory_to_power_distance.keys(),
            key=lambda f: factory_to_power_distance[f],
            reverse=True,
        )[0]
        return highest_power_factory.edge_tiles(unit=unit, no_chargers=True)

    def get_best_uncompleted_candidates(self, candidates):
        if not candidates:
            return []
        sorted_by_score = sorted(candidates, key=lambda c: self.prune_function(c))
        return [sorted_by_score[0]]

    def set_start_charge(self, start_charge):
        """Set start_charge based on input or default to True if unit is low on power"""
        if start_charge is None:
            unit = self.unit

            start_charge = (
                (unit.point.factory is not None or unit.is_power_hub)
                and unit.power < unit.unit_cfg.BATTERY_CAPACITY * 0.95
                and (
                    unit.factory.available_power(t_max=TTMAX)
                    > unit.unit_cfg.BATTERY_CAPACITY * 0.05
                )
            )

        self.start_charge = start_charge

    def get_next_actions(self, candidate, optimal):
        """Return action to take next"""

        if candidate.state == STATE.START:
            return self._get_preparation_actions(candidate)
        elif candidate.state == STATE.PREPARED:
            return self._get_to_target_actions(candidate, optimal=optimal)
        elif candidate.state == STATE.ON_TARGET:
            return self._get_on_target_actions(candidate)
        elif candidate.state == STATE.TARGET_ACHIEVED:
            return self._get_to_destination_actions(candidate, optimal=optimal)
        elif candidate.state == STATE.AT_DESTINATION:
            return self._get_at_destination_actions(candidate)
        assert (
            IS_KAGGLE or False
        ), f"t={candidate.pathfinder.t} {candidate.point} {self}:  Should not reach here {candidate.state.name}"

    def _get_to_points_actions(self, candidate, points: List[Point], optimal_dirs):
        """Get directions to points"""
        return get_directions(
            candidate.point,
            points,
            add_wait=candidate.waits_left > 0,
            allow_off_target=candidate.steps_off_target_left > 0,
            optimal_dirs=optimal_dirs,
        )

    def _get_preparation_actions(self, candidate):
        """Return action to take to prepare for action, e.g. charge"""
        if self.start_charge:
            if self.targets:
                optimal_dirs = candidate.pathfinder.to_target_directions[
                    candidate.point
                ]
            else:
                optimal_dirs = candidate.pathfinder.to_destination_directions[
                    candidate.point
                ]
            candidate.optimal_dirs = optimal_dirs
            dirs = self._get_to_points_actions(
                candidate, self.unit.factory.points, optimal_dirs=optimal_dirs
            )
            if candidate.point.factory and candidate.power < self.unit.battery_capacity:
                dirs = dirs + [ACTION.CHARGE]
            return dirs
        assert (
            IS_KAGGLE or False
        ), f"{self}:t={candidate.pathfinder.t}  Should not reach here {candidate.state.name}"

    def _get_to_target_actions(self, candidate, optimal):
        """Return action to take to target, e.g. move"""
        optimal_dirs = candidate.pathfinder.to_target_directions

        if optimal:
            if candidate.point in optimal_dirs:
                dirs = optimal_dirs[candidate.point]
                if candidate.waits_left > 0:
                    dirs = dirs + [0]
                return dirs
            else:
                assert (
                    IS_KAGGLE or self.timed_targets
                ), f"{self}:t={candidate.pathfinder.t} Should not reach here {candidate.state.name}"

        optimal_dirs = candidate.pathfinder.to_target_directions[candidate.point]
        candidate.optimal_dirs = optimal_dirs
        return self._get_to_points_actions(candidate, self.targets, optimal_dirs)

    def _get_on_target_actions(self, candidate):
        """Return action to take on target, e.g. dig, pickup, charge"""
        raise NotImplementedError(
            f"{self}:t={candidate.pathfinder.t} {candidate.point} Should not reach here  {candidate.state.name}{self.timed_targets}"
        )

    def _get_to_destination_actions(self, candidate, optimal):
        if optimal:
            optimal_dirs = candidate.pathfinder.to_destination_directions
            if candidate.point in optimal_dirs:
                dirs = optimal_dirs[candidate.point]
                if candidate.waits_left > 0:
                    dirs = dirs + [0]
                return dirs
            else:
                assert (
                    IS_KAGGLE or False
                ), f"{self}:  Should not reach here {candidate.state.name}"

        optimal_dirs = candidate.pathfinder.to_destination_directions[candidate.point]
        candidate.optimal_dirs = optimal_dirs
        return self._get_to_points_actions(candidate, self.destinations, optimal_dirs)

    def _get_at_destination_actions(self, candidate):
        """Return action to take at destination, e.g. transfer"""

        raise NotImplementedError(
            f"{self}:  Should not reach here {candidate.state.name}"
        )

    def update_state(self, candidate):
        if candidate.state == STATE.START:
            if self._is_prepared(candidate):
                candidate.state = STATE.PREPARED
            else:
                return

        if candidate.state == STATE.PREPARED:
            t = candidate.pathfinder.t
            point = candidate.point
            if self.timed_targets:
                if t in self.timed_targets and (
                    self.timed_targets[t] == point
                    or isinstance(self.timed_targets[t], list)
                    and point in self.timed_targets[t]
                ):
                    candidate.state = STATE.ON_TARGET
                else:
                    return
            else:
                if not self.targets or point in self.targets:
                    candidate.state = STATE.ON_TARGET
                else:
                    return

        if candidate.state == STATE.ON_TARGET:
            if self._target_achieved(candidate):
                candidate.state = STATE.TARGET_ACHIEVED
            else:
                return

        if candidate.state == STATE.TARGET_ACHIEVED:
            if not self.needs_to_return or candidate.point in self.destinations:
                candidate.state = STATE.AT_DESTINATION
            else:
                return

        if candidate.state == STATE.AT_DESTINATION:
            if self._is_complete(candidate):
                candidate.state = STATE.COMPLETE
            return

        assert (
            IS_KAGGLE or False
        ), f"{self}:  Should not reach here {candidate.state.name}"

    def _is_prepared(self, candidate):
        if self.start_charge:
            is_charged = (
                candidate.power >= self.unit.battery_capacity
                or ACTION.CHARGE in candidate.commands
            )
            return is_charged

        return True

    def _target_achieved(self, candidate):
        if not self.targets:
            return True
        assert NotImplementedError
        return False

    def _is_complete(self, candidate):
        return True

    def get_max_it(self):
        return self.base_max_it

    def prune_candidates(self, candidates):
        if len(candidates) > self.beam_size:
            candidates = sorted(candidates, key=lambda c: self.prune_function(c))
            candidates = candidates[: self.beam_size]
        return candidates

    def prune_function(self, candidate):
        return -candidate.score

    def execute(self, *args, **kwargs):
        return self.plan(**kwargs)

    def get_targets(self, **kwargs):
        return None

    def post_process_action_queue(self, candidate, action_queue):
        """Add a wait to the end of the action queue to ensure other units
        don't plan themselves into the same spot"""
        if self.post_process_recharge:
            recharge = self.unit.recharge_replan()

            return action_queue + [recharge]

        return action_queue

    def plan(self, **kwargs):
        unit = self.unit
        unit.tried_actions.append(self.name())

        if not self.targets:
            if self.needs_targets or not self.destinations:
                return False
        unit.tried_targets += self.targets if self.targets else []

        pf = Pathfinder(
            agent=self.agent,
            unit=unit,
            targets=self.targets,
            goal=self,
            destinations=self.destinations,
            max_steps_off_target=self.max_steps_off_target,
            max_it=self.max_it,
            max_waits=self.max_waits,
            verbose=self.verbose,
            target_bots=self.target_bots,
            **kwargs,
        )

        if pf.completed:
            pf.plan(self.actions, self.debug)
            return True
        else:
            if (
                self.track_tries
                and unit.available_power() > 0.9 * unit.battery_capacity
            ):
                for p in self.targets:
                    p.tried_times += 1
        if self.replan_action:
            self.replan_action = False
            self.targets = self.get_targets(**kwargs)
            return self.plan(**kwargs)
        return False

    def on_plan(self, best):
        pass

    # states
    def on_complete_valid(self, candidate, verbose=False):
        return True

    def __repr__(self):
        return f"{self.name()} - {self.unit}"

    def is_free_in_time(
        self, point, unit_grid, min_free_time=3, heavy_may_boot_light=True
    ):
        """Check if a point is free in time for e.g. mining."""
        unit = self.unit
        current_point = unit.point
        attack_risk_in_time = unit.game_board.attack_risk_in_time
        my_strength = unit.combat_strength(has_moved=False, power_left=unit.power)

        distance = current_point.distance_to(point)
        spread = self.max_waits + self.max_steps_off_target
        if distance == 0:
            spread = 0

        for start in range(distance, distance + spread + min_free_time + 1):
            end = start + min_free_time
            free = True
            for t in range(start, end + 1):
                if (
                    point in attack_risk_in_time[t]
                    and attack_risk_in_time[t][point] > my_strength
                ):
                    free = False
                    break
                if point not in unit_grid[t]:
                    continue
                units_at_t = unit_grid[t][point]
                if len(units_at_t) == 0:
                    continue
                if len(units_at_t) == 1 and unit in units_at_t:
                    continue
                else:
                    if heavy_may_boot_light and unit.is_heavy:
                        if any(u.is_heavy for u in units_at_t if u != unit):
                            free = False
                            break
                    else:
                        if any(u.is_own for u in units_at_t if u != unit):
                            free = False
                            break
                        # else: pass, we checked attack risk, so go ahead

            if free:
                return True
        return False

    def get_max_allowed_it(
        self, specific_actions, reference_points=None, verbose=False
    ):
        unit = self.unit
        if reference_points:
            end = max(reference_points, key=unit.point.distance_to)
        else:
            destinations = self.destinations
            if not destinations:
                assert IS_KAGGLE or False, f"{unit} {self}: No destinations??"
                return self.base_max_it
            # take center to allow some flexibility in choosing factory point
            end = destinations[0].closest_factory.center

        optimal_path_distance = len(
            get_optimal_path(self.game_board, unit.point, end, unit)
        )

        direct_distance = unit.point.distance_to(end)
        around_distance = direct_distance + self.max_steps_off_target

        # do not allow a path that takes twice as long
        around_distance = min(direct_distance * 2, around_distance)

        if verbose:
            lprint(
                f"Direct distance: {direct_distance} - around distance: {around_distance} - optimal path: {optimal_path_distance}"
            )

        return (
            max(around_distance, optimal_path_distance)
            + self.max_waits
            + specific_actions
            + self.start_charge * 2
        )

    def filter_rss_points(
        self,
        candidates: List[Point],
        unit: Unit,
        max_points=5,
        verbose=False,
    ):
        """Filter out points that are not reachable by the unit"""
        point: Point = unit.point
        step = self.game_board.step
        factory = unit.factory
        if not candidates:
            return []

        efficiency_factor = (
            2
            if (
                (
                    candidates[0].ore
                    and factory.is_ore_risk()
                    and factory.available_metal() < 10
                )
                or (
                    candidates[0].ice
                    and factory.is_ice_risk()
                    and factory.available_water() < 50
                )
            )
            and factory.n_heavies == 1
            else 1.5
        )

        min_efficiency = (
            MIN_EFFICIENCY
            if not self.allow_low_efficiency
            and (unit.power + factory.available_power() < 3000)
            else efficiency_factor * (MIN_EFFICIENCY - 1) + 1
        )
        if verbose:
            lprint(
                f"Min efficiency: {min_efficiency} efficiency_factor={efficiency_factor} start_charge={self.start_charge}"
            )

        max_distance = unit.max_return_distance(
            self.start_charge, min_efficiency=min_efficiency
        )

        # don't mine too far if there is still a lot of rubble to dig for power
        if (
            len(factory.ice_hubs) > 0
            and factory.get_lichen_lake_size() < BREAK_EVEN_ICE_TILES
        ):
            max_distance = min(
                math.ceil(MAX_EFFICIENT_LIGHT_DISTANCE / 2), max_distance
            )

        max_unit_distance = max_distance

        # limit travel distance if unit is out in the field
        if unit.point.distance_to(unit.factory) > 4 and unit.point not in candidates:
            max_unit_distance = 4
        elif not unit.factory:
            max_unit_distance = 8  # don't go to other side of the factory

        if (
            factory.power_hub_push
            and not factory.full_power_hub()
            and not (unit.dies or unit.could_die)
            and step < 150
        ):
            max_distance = min(max_distance, 2)

        available_power = unit.available_power(include_factory=self.start_charge)
        if verbose:
            lprint(
                "max_distance:",
                max_distance,
                "max_unit_distance:",
                max_unit_distance,
                "min_efficiency:",
                min_efficiency,
                "available_power:",
                available_power,
            )
            lprint("input candidates:", candidates)

        def is_efficient(p: Point):
            required_to = power_required(unit, unit.point, p) + unit.comm_cost
            charged_to = unit.charge_up_to(unit.point.distance_to(p))
            n_digs = (available_power + charged_to - required_to) // unit.dig_cost

            if verbose:
                lprint(
                    "p:",
                    p,
                    "required_to:",
                    required_to,
                    "charged_to:",
                    charged_to,
                    "n_digs:",
                    n_digs,
                )  # noqa

            if n_digs <= 0:
                return False
            value_gained = unit.dig_cost * n_digs

            required_back = (
                power_required(unit, p, factory.closest_tile(p)) + unit.comm_cost
            )
            required = required_to + required_back + value_gained
            ratio = required / value_gained

            if verbose:
                lprint(
                    "p:",
                    p,
                    "required_back:",
                    required_back,
                    "required:",
                    required,
                    "ratio:",
                    ratio,
                )  # noqa
            return ratio <= min_efficiency

        step = self.game_board.step
        candidates = [
            p
            for p in candidates
            if p not in unit.tried_targets
            and (factory.distance_to(p) <= max_distance)
            and (unit.point.distance_to(p) <= max_unit_distance)
            and (  # don't go there if there is an enemy factory nearby and another own is closer
                p.closest_own_factory == factory
                or p.closest_enemy_factory_distance > p.distance_to(factory)
            )
            and (
                p.rubble == 0 or p.closest_factory.is_own or factory.is_ore_risk()
            )  # dont clear rubble for enemy, unless we need ore
            and not (p.unit and p.unit.is_hub)
            and not (p.is_hub_candidate and unit.is_light and unit.game_board.step < 20)
            and (unit.is_heavy or p.unit or len(p.dug_by) == 0)
        ]
        if verbose:
            lprint("First pass candidates:", candidates)

        candidates = [
            p
            for p in candidates
            if (  # don't rubble mine if there is an enemy factory nearby
                p.closest_factory.is_own
                or p.rubble == 0
                or (
                    p.rubble < 30
                    and p.closest_enemy_factory.n_lights == 0
                    and self.game_board.step > 75
                )
                or factory.distance_to(p) <= p.closest_factory_distance
                or (
                    p.closest_factory_distance > 2
                    and factory.distance_to(p) < 2 * p.closest_factory_distance
                )
            )
        ]
        if verbose:
            lprint("enemy factory nearby candidates:", candidates)

        candidates = [p for p in candidates if is_efficient(p)]
        if verbose:
            lprint("Is efficient candidates:", candidates)

        unit_grid = unit.game_board.unit_grid_in_time
        heavy_may_boot_light = (
            unit.is_heavy and (unit.power + unit.factory.available_power()) >= 1000
        )
        candidates = [
            c
            for c in candidates
            if self.is_free_in_time(c, unit_grid, heavy_may_boot_light)
        ]

        if verbose:
            lprint("is_free candidates:", candidates)

        # don't go to a point that is closer to an enemy and enemy has bots near
        candidates = [
            c
            for c in candidates
            if c.closest_factory.is_own
            or len(
                [
                    p
                    for p in c.points_within_distance(3)
                    if p.unit and p.unit.is_enemy and unit <= p.unit and p.unit.can_move
                ]
            )
            == 0
        ]

        move_cost_candidates = {}

        for c in candidates:
            path_to_rss = get_optimal_path(unit.game_board, point, c, unit)
            move_cost = sum([unit.move_cost_rubble(p.rubble) for p in path_to_rss[:-1]])

            path_to_factory = get_optimal_path(unit.game_board, c, factory.center, unit)
            move_cost += sum(
                [unit.move_cost_rubble(p.rubble) for p in path_to_factory[:-1]]
            )

            # drop some inefficient routes already
            if move_cost <= (unit.base_move_cost * max(0, point.distance_to(c))) * 5:
                move_cost_candidates[c] = move_cost

        candidates = move_cost_candidates.keys()

        if verbose:
            lprint("move_cost candidates:", candidates)

        if candidates:
            n_before = len(candidates)
            closest_to_self = [c for c in candidates if c.closest_factory.is_own]
            if len(closest_to_self) > 0:
                candidates = closest_to_self

            candidates = sorted(candidates, key=lambda p: move_cost_candidates[p])[
                :max_points
            ]
            closest = candidates[0]

            candidates = [
                c
                for c in candidates
                if c.distance_to(closest) <= MAX_COMBINE_TARGET_DISTANCE
            ]
            if len(candidates) != n_before:
                self.replan_action = True  # make sure to replan if failed to find path

            candidates = sorted(candidates, key=closest.distance_to)[:max_points]

        if verbose:
            lprint("Final:", candidates)

        return candidates

    def update_targets(self, pathfinder):
        # do nothing by default, used for timed actions
        return


class DigAction(Action):
    penalize_move = True
    needs_targets = True
    track_tries = True
    dig_19_20_rubble = False
    can_be_planned_when_retreating = False

    def _get_on_target_actions(self, candidate):
        """Target is somewhere to dig and this action is dig until complete"""
        return [ACTION.DIG]

    def prune_function(self, candidate):
        goal_achieved = candidate.state >= STATE.TARGET_ACHIEVED

        min_distance = (
            0
            if goal_achieved
            else candidate.point.min_distance_to(candidate.pathfinder.targets)
        )
        return (min_distance, -candidate.score)

    def _target_achieved(self, candidate):
        enough_digs = (
            ACTION.DIG in candidate.commands
        )  # candidate.on_goal_action >= self.min_goal_action
        if not enough_digs:
            return False

        # stop digging if out of power or cargo full or goal achieved or out of time
        dig_goal_achieved = (
            candidate.ice + candidate.ore >= candidate.pathfinder.goal_amount
            or candidate.is_full
        )

        if dig_goal_achieved:
            if self.verbose:
                lprint(f"{self}: _target_achieved dig_goal_achieved")
            return True

        distance = candidate.distance_to_destination() if self.needs_to_return else 0
        almost_out_of_time = (
            distance >= candidate.time_left() - self.out_of_time_threshold
        )

        if almost_out_of_time:
            if self.verbose and enough_digs:
                lprint(f"{self}: _target_achieved almost_out_of_time and enough_digs")
            return enough_digs

        power_for_dig = self.unit.dig_cost

        # add one to compensate waiting
        power_needed_at_end = candidate.power_needed_at_end()
        almost_out_of_power = power_for_dig + power_needed_at_end >= candidate.power

        if almost_out_of_power:
            if self.verbose and enough_digs:
                lprint(
                    f"{self}: _target_achieved almost_out_of_power and enough_digs power: {candidate.power}power_needed_at_end: {power_needed_at_end}"
                )
            return enough_digs

        return False

    def get_rubble_targets(self, verbose=False):
        """Get rubble targets for this action, which is based on several priorities
        1. shortest path to rss
        2. often visited points
        2. clear rubble around factory
        3. dig path to neighbour factory
        4. clear lichen lake

        needs to add: much visited points
        """
        unit = self.unit
        factory = unit.factory
        point = unit.point

        max_points = 5

        max_dist = unit.max_return_distance()

        if verbose:
            lprint(f"max_return_distance:{max_dist}")

        # don't mine too far if there is still a lot of rubble to dig for power
        if (
            len(factory.ice_hubs) > 0
            and factory.get_lichen_lake_size() < BREAK_EVEN_ICE_TILES
        ):
            max_dist = min(math.ceil(MAX_EFFICIENT_LIGHT_DISTANCE / 2), max_dist)

        plenty_of_power = unit.power > unit.battery_capacity * 0.9

        def prune_candidates(candidates, priority):
            # filter out points that not suitable
            unit_grid = unit.game_board.unit_grid_in_time
            dig_removed = unit.unit_cfg.DIG_RUBBLE_REMOVED
            step = unit.game_board.step
            if verbose:
                lprint(
                    f"in prune_candidates:",
                    len(candidates),
                    candidates,
                )

            candidates = [
                p
                for p in candidates
                if p.rubble > p.rubble_target_value
                and p.get_rubble_at_time(TTMAX) > p.rubble_target_value
                and (
                    plenty_of_power
                    or factory.clear_rubble_for_lichen()
                    or p.closest_factory == factory
                )
            ]

            if verbose:
                lprint(
                    f"after prune_candidates:",
                    len(candidates),
                    candidates,
                )

            candidates = [
                p
                for p in candidates
                if p.distance_to(point) <= max_dist
                and (
                    p.closest_own_factory == factory
                    or p.distance_to(factory) < 4
                    or (
                        priority >= MIN_NORMAL_RUBBLE_PRIORITY
                        and p.closest_own_factory.n_lights < 5
                    )
                )
                and self.is_free_in_time(p, unit_grid, min(3, p.rubble // dig_removed))
                and not (p.is_hub_candidate and step < 11)
                and len(
                    [
                        ap
                        for ap in p.adjacent_points() + [p]
                        if ap.rubble and len(ap.dug_by) > 0
                    ]
                )  # make sure there are not too many bots digging in the vicinity
                < 4
                and len([ap for ap in p.adjacent_points() + [p] if ap.unit]) < 5
                and (
                    p.closest_own_factory.clear_rubble_for_lichen()
                    or (p.distance_to(point) <= 6 and p.closest_own_factory == factory)
                )
            ]
            return candidates

        if factory:
            gb = self.unit.game_board

            candidates = []

            tried = set(unit.tried_targets)
            prev_d = None
            distances = (
                [6]
                if factory.power_hub_push
                and not factory.full_power_hub()
                and not (unit.dies or unit.could_die)
                else [8, MAX_EFFICIENT_LIGHT_DISTANCE]
            )
            for d in distances:
                area_grid, priorities = unit.get_target_grid_at_distance(
                    d, min_distance=prev_d
                )
                prev_d = d
                # area_grid, priorities = set_inverse_zero(
                #     target_grid.copy(), unit.point, d, get_uniques=True
                # )

                # priorities = sorted(priorities, reverse=True)

                if verbose:
                    plt.imshow(area_grid.T)
                    plt.title(f"area_grid {d}")
                    plt.show()
                # for is_own in [True, False]:
                for priority in priorities:
                    if verbose:
                        lprint(
                            f"d: {d} priority {priority} min: {self.min_priority} max: {self.max_priority}"
                        )

                    if (
                        gb.steps_left < 200 or factory.available_power() < 500
                    ) and priority <= 4:
                        continue

                    if priority >= self.min_priority and priority <= self.max_priority:
                        pts = set(gb.grid_to_points(area_grid == priority))
                        if verbose:
                            lprint(f"priority {priority} pts: {pts} - tried:{tried}")
                        candidates = prune_candidates(pts - tried, priority)
                        if verbose:
                            lprint(f"priority {priority} pts: {pts}")
                            lprint(f"after pruning candidates: {candidates}")

                        if candidates:
                            if verbose:
                                lprint(f"candidates before: {candidates}")
                            n_before = len(candidates)
                            # take only a few, cluster them together
                            candidates = sorted(candidates, key=point.distance_to)
                            closest = candidates[0]
                            distance = closest.distance_to(point)

                            candidates = [
                                c
                                for c in candidates
                                if c.distance_to(closest) <= MAX_COMBINE_TARGET_DISTANCE
                                and c.distance_to(point) == distance
                            ]
                            if verbose:
                                lprint(f"candidates after: {candidates}")

                            targets = sorted(candidates, key=closest.distance_to)[
                                :max_points
                            ]

                            # try closest first
                            if len(targets) < n_before:
                                self.replan_action = True

                            if (
                                priority >= self.min_priority
                                and priority < self.max_priority
                                # and priority > PRIORITY.SURROUNDING_NON_ICE
                            ):
                                self.replan_action = True

                            if targets:
                                return targets
        return


class MiningAction(DigAction):
    factory_as_destinations = True
    check_efficiency = True

    def on_complete_valid(self, candidate, verbose=False):
        if not self.check_efficiency:
            return True

        n_digs = candidate.on_goal_action
        dig_cost = self.unit.dig_cost
        value_gained = n_digs * dig_cost

        consumed = candidate.power_consumed

        if consumed > value_gained * MIN_EFFICIENCY:
            if verbose:
                lprint(
                    f"Unit {self.unit.unit_id} consumed more surplus power {consumed} than value gained {value_gained} ({100*consumed/value_gained:.0f}%) route_len={len(candidate.path)}, {candidate.commands} {candidate.path}",
                )
            return False
        return True


class ReturnMiningAction(MiningAction):
    out_of_time_threshold = 1
    post_process_queue_len = 0

    def _get_at_destination_actions(self, candidate):
        dirs = [ACTION.TRANSFER]
        if candidate.power <= self.unit.comm_cost:
            dirs.append(ACTION.CHARGE)
        return dirs

    def _is_complete(self, candidate):
        return super()._is_complete(candidate) and (
            candidate.ice == 0
            and candidate.ore == 0
            and ACTION.TRANSFER in candidate.commands
        )


class Rubble(Action):
    max_distance = 10
    max_waits = 1
    factory_as_destinations = True
    check_efficiency = False

    def get_targets(self, **kwargs):
        return self.get_rubble_targets(verbose=verbose)


class RubbleRaid(DigAction):
    max_steps_off_target = 1
    max_waits = 0
    needs_to_return = False
    max_distance = 12
    base_max_it = 25
    check_efficiency = False

    def get_targets(self, verbose=False, **kwargs):
        return self.get_rubble_targets(verbose=verbose)

    # def post_process_action_queue(self, candidate, action_queue):
    #     target_power = self.unit.unit_cfg.BATTERY_CAPACITY * 0.75

    #     if candidate.power > target_power:
    #         return action_queue

    #     action_queue.append([5, 0, 0, int(target_power), 0, 1])

    #     return action_queue

    def _is_complete(self, candidate):
        t = candidate.pathfinder.t
        return (
            t == candidate.pathfinder.max_it
            or candidate.power < self.unit.dig_cost + 2 * self.unit.unit_cfg.MOVE_COST
        ) and candidate.get_rubble_left(
            t
        ) <= 0  # don't wait on a rubble tile


class UniMining(MiningAction):
    max_steps_off_target = 2
    max_waits = 2

    needs_to_return = False
    reserve_power_to_return = True
    factory_as_destinations = (
        False  # will check a close factory with most power to return to
    )

    base_max_it = TTMAX
    stop_search_if_complete = True
    beam_size = 15


class UniIce(UniMining):
    resource_id = RESOURCE_MAP["ice"]

    def get_max_it(self):
        out_of_water_time = self.unit.factory.out_of_water_time()
        if out_of_water_time < TTMAX:
            return min(self.base_max_it, self.unit.factory.out_of_water_time() - 1)
        return self.base_max_it

    def get_targets(self, verbose=False, **kwargs):
        ice_points = self.unit.factory.ice_points

        # don't consider further points that the amount of bots
        candidates = ice_points[: len(self.unit.factory.units) + 1]

        return self.filter_rss_points(candidates, self.unit, verbose=verbose)


class UniOre(UniMining):
    resource_id = RESOURCE_MAP["ore"]

    def get_targets(self, verbose=False, **kwargs):
        factory = self.unit.factory

        if factory.available_power() > 3000 and factory.available_metal() < 100:
            target_candidates = [
                p for p in factory.all_ore_points if p.distance_to(factory) <= 24
            ]
        else:
            target_candidates = factory.ore_points

        return self.filter_rss_points(target_candidates, self.unit, verbose=verbose)


def is_skimmed(rubble):
    return rubble % 20 in [0, 15, 16, 17, 18, 19]


class SkimRubble(UniMining):
    factory_as_destinations = True
    check_efficiency = False

    def get_targets(self, **kwargs):
        return [
            p
            for p in self.unit.factory.hubs
            if not is_skimmed(p.rubble) and not is_skimmed(p.get_rubble_at_time(TTMAX))
        ]

    def _target_achieved(self, candidate):
        return is_skimmed(candidate.get_rubble_left(candidate.pathfinder.t))


class UniRubble(UniMining):
    max_distance = 15
    reserve_power_to_return = True  # must be able to flee back to factory
    post_process_recharge = True
    check_efficiency = False
    max_priority = PRIORITY.LAKE_CONNECTOR_PRIO - 1

    def get_targets(self, **kwargs):
        return self.get_rubble_targets()

    def _target_achieved(self, candidate):
        return candidate.get_rubble_left(
            candidate.pathfinder.t
        ) <= candidate.point.rubble_target_value or super()._target_achieved(candidate)

    def _get_on_target_actions(self, candidate):
        """Target is somewhere to dig and this action is dig until complete"""
        t = candidate.pathfinder.t
        rubble_left = candidate.get_rubble_left(t)
        if rubble_left > 0:
            return [ACTION.DIG]
        return []  # this fails if we unplan a bot and don't update the rubble
        # assert (
        #     IS_KAGGLE or False
        # ), f"{self}: {t} {candidate.point}:  on_target, but {rubble_left > 0} rubble={rubble_left} != 0 {candidate.state.name}, type={type(rubble_left)} {self._target_achieved(candidate)} valid={candidate.is_valid}"


class UniRubbleTopPrio(UniRubble):
    min_priority = UniRubble.max_priority + 1
    max_priority = 1000


class UniRubbleLowPrio(UniRubble):
    min_priority = 1
    max_priority = UniRubble.min_priority - 1


class UniLichen(UniMining):
    base_max_it = TTMAX
    reserve_power_to_return = True
    factory_as_destinations = True
    check_efficiency = False
    ignore_power_req_steps_left = 25
    ignore_attack_risk_on_unit_dominance = True

    def get_max_it(self):
        targets = self.targets if self.targets else self.get_targets()
        if targets:
            return self.get_max_allowed_it(0, targets)
        return self.base_max_it

    def get_targets(self, factory=None, verbose=False, **kwargs):
        # need to get the list of the closest lichen points
        unit = self.unit
        steps_left = self.game_board.steps_left
        point = unit.point
        agent = self.agent

        if factory and factory.n_all_lichen_tiles == 0:
            return []

        # don't try unilichen if at home and not a lot of power
        if not unit.consider_lichen_attack():
            return []

        all_candidates = agent.get_lichen_targets()
        if factory is None:
            # prevent traveling all over the map and between factories, so start with nearby factories
            factory_candidates = [
                f
                for f in agent.opponent_factories
                if (unit.is_heavy or (f.lichen > 0 and f.distance_to(point) < 14))
                and f.consider_lichen_assault()
                and (len(f.own_units_in_range()) <= max(f.n_lichen_tiles * 1.5, 10))
            ]

            if verbose:
                lprint(f"factory_candidates: {factory_candidates}")
            if not factory_candidates:
                factory_candidates = agent.opponent_factories

            # sort by # lichen tiles / number of units of same type
            enemy_factories = sorted(
                [
                    f
                    for f in factory_candidates
                    if f.consider_lichen_assault()
                    # (f.n_lichen_tiles >= BREAK_EVEN_ICE_TILES)
                    # or f.available_water() > 200
                    # or steps_left < 75
                    # or len(agent.units) / max(1, len(agent.opponent_units)) > 2
                    # or agent.n_opp_heavies > agent.n_opp_lights
                ],
                key=lambda f: f.n_lichen_tiles_div_unit_count_div_dist(
                    unit
                ),  # f.lichen if unit.is_light else -f.distance_to(point),
                reverse=True,
            )
        else:
            enemy_factories = [factory]

        if verbose:
            lprint(f"enemy_factories: {enemy_factories}")

        # for now assume just stationary unit with same power
        combat_strength = unit.combat_strength(has_moved=False, power_left=unit.power)

        # need at least two digs with safe escape
        av_unit_power = (
            unit.available_power_lichen_digs()
            if steps_left > self.ignore_power_req_steps_left
            else unit.available_power()
        )

        max_distance = min(self.base_max_it, steps_left - 1)
        unit_dig_remove = unit.unit_cfg.DIG_LICHEN_REMOVED

        def check_power(c):
            if steps_left < self.ignore_power_req_steps_left:
                return power_required(unit, point, c) + unit.comm_cost + math.ceil(
                    c.lichen / unit_dig_remove
                ) * unit.dig_cost < av_unit_power + unit.charge_up_to(
                    point.distance_to(c) + c.closest_own_factory_distance
                )

            return power_required(unit, point, c) + power_required(
                unit, c, c.closest_own_factory.center
            ) < av_unit_power + unit.charge_up_to(
                point.distance_to(c) + c.closest_own_factory_distance
            )

        # try highest prio factory first
        risk = self.game_board.attack_risk_in_time[1]
        for enemy_factory in enemy_factories:
            unit_near_factory = point.distance_to(enemy_factory) <= 4

            def check_distance(p):
                n_digs = math.ceil(p.lichen // 10) if unit.is_light else 1

                distance = unit.point.distance_to(p)
                if distance + n_digs >= steps_left:
                    return False

                if distance + n_digs >= TTMAX:
                    return False
                return True

            if verbose:
                lprint(enemy_factory, f"before candidates: {all_candidates}")

            candidates = [
                c
                for c in all_candidates
                if c.lichen_strains == enemy_factory.id
                and c.tried_times < 5
                and (
                    (unit.is_light or unit_near_factory)
                    or c.lichen > 79
                    or c.choke_kills > 4
                    or c.adjacent_factories
                    or steps_left < 15
                )
                and (
                    len(c.dug_by) == 0
                    or min(u.point.distance_to(c) for u in c.dug_by)
                    > (
                        point.distance_to(c)
                        + 1
                        + (c.lichen // 10 if unit.is_light else 0)
                    )
                )
                and point.distance_to(c) < max_distance
                and check_distance(c)
                and check_power(c)
                and (
                    # (agent.unit_dominance and point.distance_to(c) <= 20)
                    unit.game_board.attack_risk_in_time[point.distance_to(c)][c]
                    < combat_strength
                )
            ]
            if verbose:
                lprint(enemy_factory, f"after candidates: {candidates}")

            if not candidates:
                continue

            # prioritize effect for heavy and greed for light (closest, low lichen)
            # prefer to dig at choke points
            # prefer high lichen for heavy, low for light

            candidates = sorted(
                candidates,
                key=lambda x: (
                    risk[x],
                    point.distance_to(x) + x.lichen // 10,
                    -x.choke_kills,
                    x.lichen,
                )
                if unit.is_light
                else (
                    (
                        risk[x] >= 10000,
                        point.distance_to(x) + enemy_factory.distance_to(x)
                        if point.factory
                        else point.distance_to(x),
                    ),
                    -x.choke_kills,
                    -x.lichen,
                ),
            )
            if verbose:
                lprint(f"sorted: {candidates}")

            # make sure a choke point is targeted if any
            if candidates[0].choke_kills > 0:
                targets = [
                    c
                    for c in candidates[:4]
                    if c.choke_kills == candidates[0].choke_kills
                ]
            else:
                if unit.is_light:
                    targets = candidates[:6]
                    optimal = targets[0]
                    targets = [t for t in targets if optimal.distance_to(t) <= 3]

                    if unit.point.distance_to(targets[0]) <= 2:
                        safest = [
                            t
                            for t in targets
                            if not any(
                                p.unit and p.unit.is_enemy
                                for p in t.points_within_distance(3)
                            )
                        ]
                        if safest:
                            targets = safest
                else:
                    closest = candidates[0]
                    distance_to_factory = closest.distance_to(enemy_factory)

                    targets = [
                        c
                        for c in candidates[:10]
                        if c.distance_to(closest) <= 1
                        and c.distance_to(enemy_factory) <= distance_to_factory
                    ]

            return targets
        return []

    def _target_achieved(self, candidate):
        n_digs = candidate.on_goal_action
        t = candidate.pathfinder.t

        if self.unit.is_heavy:
            return n_digs >= 1

        return (
            n_digs
            >= min(candidate.point.lichen + t, 110)
            / self.unit.unit_cfg.DIG_LICHEN_REMOVED
        )

    def get_random_number(self):
        return np.random.randint(8000, 8500)

    def post_process_action_queue(self, candidate, action_queue):
        # todo:
        dig_point = candidate.point
        if self.unit.point.distance_to(dig_point) > HIDE_LICHEN_DIG_DISTANCE:
            new_queue = [a for a in action_queue if a[0] != 3]

            # return home
            random_nr = self.get_random_number()
            queue_to_factory = get_rss_path_queue(
                self.agent, dig_point, self.unit.factory.center, random_nr
            )

            action_queue = new_queue + queue_to_factory
        else:
            recharge = self.unit.recharge(RECHARGE.CLOSE_FACTORY)
            action_queue = action_queue + [recharge]

        # assert IS_KAGGLE or len(action_queue) <= 20
        return action_queue[:20]


class UniChokeLichen(UniLichen):
    base_max_it = TTMAX
    reserve_power_to_return = True

    def get_random_number(self):
        return np.random.randint(8501, 8999)

    def get_targets(self, **kwargs):
        # need to get the list of the closest lichen points
        all_candidates = self.game_board.agent.choke_points_enemy
        if not all_candidates:
            return []

        unit = self.unit

        # for now assume just stationary unit with same power
        combat_strength = unit.combat_strength(has_moved=False, power_left=unit.power)

        # need at least two digs with safe escape
        av_unit_power = unit.available_power_lichen_digs()
        steps_left = self.game_board.steps_left
        max_distance = min(self.base_max_it, steps_left - 1)

        candidates = [
            c
            for c in all_candidates.keys()
            if len(c.dug_by) == 0
            and unit.point.distance_to(c) < max_distance
            and power_required(unit, c, unit.factory.center) * 2
            < av_unit_power + unit.charge_up_to(2 * unit.point.distance_to(c))
            and (
                unit.game_board.attack_risk_in_time[unit.point.distance_to(c)][c]
                <= combat_strength
            )
        ]

        if not candidates:
            return []

        candidates = sorted(candidates, key=unit.point.distance_to)
        closest = candidates[0]
        targets = sorted(candidates, key=closest.distance_to)[:5]
        return targets


class Ice(ReturnMiningAction):
    base_max_it = 50
    resource_id = RESOURCE_MAP["ice"]

    def get_max_it(self):
        out_of_water_time = self.unit.factory.out_of_water_time()
        if out_of_water_time < TTMAX:
            return min(self.base_max_it, self.unit.factory.out_of_water_time() - 1)
        return self.base_max_it

    def get_targets(self, **kwargs):
        ice_points = self.unit.factory.ice_points
        return self.filter_rss_points(ice_points, self.unit)


class Ore(ReturnMiningAction):
    base_max_it = 50
    resource_id = RESOURCE_MAP["ore"]

    def get_targets(self, **kwargs):
        return self.filter_rss_points(self.unit.factory.ore_points, self.unit)


class Dump(Action):
    stop_search_if_complete = True
    max_waits = 2
    beam_size = 20
    ignore_power_req_end = True  # added since we added a charge option

    def get_max_it(self):
        unit = self.unit
        cargo = unit.cargo
        if cargo.water > 0 or cargo.metal > 0:
            return TTMAX

        max_allowed_it = self.get_max_allowed_it(1)  # transfer

        return max_allowed_it

    def _is_complete(self, candidate):
        pf = candidate.pathfinder
        unit = self.unit
        return (
            candidate.ice == 0
            and candidate.ore == 0
            and ACTION.TRANSFER in candidate.commands
            and candidate.power + pf.get_charge_power(pf.t, 1)
            >= unit.comm_cost + unit.base_move_cost
        )

    def _get_at_destination_actions(self, candidate):
        dirs = []
        if ACTION.TRANSFER not in candidate.commands:
            dirs = [ACTION.TRANSFER]

        unit = self.unit
        if candidate.power <= unit.comm_cost + unit.base_move_cost:
            dirs.append(ACTION.CHARGE)
        return dirs


class TransportAction(Action):
    needs_targets = True
    base_max_it = TTMAX
    beam_size = 15
    penalize_move = True

    def _target_achieved(self, candidate):
        charged_enough = (
            candidate.power >= self.unit.unit_cfg.BATTERY_CAPACITY * 0.9
            or ACTION.CHARGE in candidate.commands
        )

        return charged_enough and ACTION.PICKUP in candidate.commands

    def _get_on_target_actions(self, candidate):
        return [ACTION.PICKUP, ACTION.CHARGE]

    def _get_at_destination_actions(self, candidate):
        dirs = [ACTION.TRANSFER]
        if candidate.power <= self.unit.comm_cost:
            dirs.append(ACTION.CHARGE)
        return dirs

    def _is_complete(self, candidate):
        pf = candidate.pathfinder
        return (
            ACTION.TRANSFER in candidate.commands
            and (candidate.power + pf.get_charge_power(pf.t, 1)) >= self.unit.comm_cost
        )


class Water(TransportAction):
    resource_id = RESOURCE_MAP["water"]

    @staticmethod
    def get_closest_low_water_factory(unit):
        low_water_factories = sorted(
            [
                f
                for f in unit.game_board.agent.factories
                if (
                    f.out_of_water_time() < TTMAX
                    or (f.available_water() < 50 and len(f.ice_hubs) == 0)
                )
            ],
            key=lambda f: f.cargo.water,
        )
        if not low_water_factories:
            return None
        return sorted(low_water_factories, key=unit.point.distance_to)[0]

    @staticmethod
    def get_high_water_factories(unit):
        return [
            f
            for f in unit.game_board.agent.factories
            if f.available_water() > MIN_DONOR_WATER_FACTORY
            and f.out_of_water_time() >= TTMAX
        ]

    def get_max_it(self):
        factory = self.destinations[0].factory
        return min(
            self.base_max_it,
            factory.out_of_water_time() - 2,
        )

    def get_targets(self, **kwargs):
        targets = self.unit.factory.edge_tiles(unit=self.unit, no_chargers=True)
        return targets

    def execute(self, low_water_factories=None, *args, **kwargs):
        unit = self.unit
        factory = unit.factory

        current_cargo = unit.cargo.water
        capacity = unit.cargo_space
        if current_cargo >= capacity:
            lprint("Cargo already full, changing to dump instead")
            return Dump(unit=unit, destinations=self.destinations).execute()

        amount = min(capacity - current_cargo, factory.available_water() - 65)
        return self.plan(goal_amount=amount)


class Metal(TransportAction):
    resource_id = RESOURCE_MAP["metal"]

    @staticmethod
    def get_closest_low_metal_factory(unit):
        low_metal_factories = [
            f
            for f in unit.game_board.agent.factories
            if max(f.metal_in_time) < 100 and f.is_ore_risk()
            # and f.available_power() > 500
            and unit.factory.available_metal() - 50 > max(f.metal_in_time)
            and unit.factory.distance_to(f) < 35
        ]

        if not low_metal_factories:
            return None
        return sorted(low_metal_factories, key=unit.point.distance_to)[0]

    @staticmethod
    def get_high_metal_factories(unit):
        return [
            f
            for f in unit.game_board.agent.factories
            if f.available_metal() > (150 if f.available_power() > 500 else 125)
        ]

    def get_targets(self, **kwargs):
        targets = self.unit.factory.points
        return targets

    def execute(self, low_water_factories=None, *args, **kwargs):
        unit = self.unit
        current_cargo = unit.cargo.metal
        capacity = unit.cargo_space
        if current_cargo >= capacity:
            lprint("Cargo already full, changing to dump instead")
            return Dump(unit=unit, destinations=self.destinations).execute()

        amount = min(
            capacity - current_cargo,
            unit.factory.available_metal()
            - (100 if unit.factory.available_power() > 500 else 50),
        )
        # need minimum cargo
        if amount < 40:
            return False

        lprint(
            f"Metal: {amount} current cargo: {current_cargo} available at factory: {unit.factory.available_metal()}"
        )
        return self.plan(goal_amount=amount)


class Rebase(Action):
    base_max_it = TTMAX
    beam_size = 15

    def on_plan(self, best):
        last_point_factory = best.point.factory
        if last_point_factory:
            self.unit.factory = last_point_factory


class Move(Action):
    max_steps_off_target = 0
    needs_to_return = False
    penalize_move = True
    max_waits = 0
    post_process_recharge = True

    def _target_achieved(self, candidate):
        # If we are on the target, we are done
        return True


class Offense(Action):
    needs_to_return = False
    needs_targets = True
    penalize_move = True
    max_waits = 0
    max_steps_off_target = 0
    post_process_queue_len = 1
    reserve_power_to_return = True
    disturb_is_good = True

    def prune_function(self, candidate):
        return (candidate.distance_to_target(), -candidate.score)

    def post_process_action_queue(self, candidate, action_queue):
        if self.unit.cargo.ice or self.unit.cargo.ore:
            return action_queue

        target_power = self.unit.unit_cfg.BATTERY_CAPACITY - self.unit.charge

        if candidate.power > target_power:
            return action_queue

        action_queue.append([5, 0, 0, int(target_power), 0, 1])

        return action_queue


class Kamikaze(Offense):
    needs_to_return = False
    needs_targets = True
    reserve_power_to_return = False
    max_steps_off_target = 1
    may_die_at_destination = True
    ignore_power_req_steps_left = 25
    ignore_attack_risk_on_unit_dominance = True
    beam_size = 15
    stop_search_if_complete = True
    allow_death_on_lichen = True
    disturb_is_good = False

    def _get_on_target_actions(self, candidate):
        unit = self.unit
        return [ACTION.SELF_DESTRUCT] if unit.is_light else [ACTION.DIG]

    def _target_achieved(self, candidate):
        # arrive at target is enough
        return candidate.died_on_lichen or ACTION.DIG in candidate.commands


class SolarHit(Offense):
    penalize_move = False
    max_steps_off_target = 1
    max_waits = 1
    stop_search_if_complete = True

    def _target_achieved(self, candidate):
        # arrive at target is enough
        return True

    def get_targets(self, **kwargs):
        unit = self.unit
        point = unit.point
        av_power = unit.available_power()

        candidates = [
            c
            for c, total_heavy_power in self.agent.get_solar_targets().items()
            if c.distance_to_point(point) <= 15
            and total_heavy_power < av_power
            and power_required(unit, point, c) < unit.battery_capacity * 0.4
        ]

        if not candidates:
            return []

        targets = sorted(candidates, key=point.distance_to)
        closest = targets[0]
        return [t for t in targets if t.distance_to_point(closest) <= 4]


class Harass(Offense):
    max_steps_off_target = 1
    beam_size = 10
    reserve_power_to_return = True
    may_die_at_destination = True  # will be handled by unplan logic
    factory_as_destinations = True
    can_be_planned_when_retreating = False
    stop_search_if_complete = True

    def _target_achieved(self, candidate):
        # arrive at target is enough
        return True

    def get_targets(self, **kwargs):
        unit = self.unit
        point = unit.point
        diggers = [
            p
            for p in point.points_within_distance(5)
            if p.unit
            and p.unit.is_enemy
            and len(p.dug_by) > 0
            and p.dug_by[0].is_enemy
            and p.dug_by[0] <= unit
        ][:3]

        candidates = flatten([p.adjacent_points() for p in diggers])
        candidates = [p for p in candidates if p.factory is None]
        if not candidates:
            return []
        closest = sorted(candidates, key=point.distance_to)[0]
        return sorted(candidates, key=closest.distance_to)[:5]


class Defend(Offense):
    stop_search_if_complete = True

    def _target_achieved(self, candidate):
        # achieved if target reached, todo: add timing
        return True

    def post_process_action_queue(self, candidate, action_queue):
        recharge = self.unit.recharge(RECHARGE.DEFEND, n=1)
        # self.unit.move(0, n=1)

        return action_queue + [recharge]

    def on_plan(self, best):
        self.unit.is_defending = True


class Attack(Offense):
    max_steps_off_target = 0
    max_waits = 0
    base_max_it = 7
    post_process_queue_len = 1
    can_route_own_unit_at_t1 = True

    def execute(self, is_intercept=False, **kwargs):
        # TODO check that we are not already attacking this unit
        unit = self.unit
        safe_destinations = []
        self.is_intercept = is_intercept

        assert IS_KAGGLE or self.target_bots, "No target bots provided in attack"
        assert (
            IS_KAGGLE or self.target_bots[0].is_enemy
        ), f"Target {self.target_bots} is not enemy"

        # only need destination if we keep camping
        targets = self.targets
        # for p in targets:
        #     for p2 in p.adjacent_points() + [p]:
        #         if (not p2.factory or p2.factory.is_own) and (
        #             not p2.adjacent_factories
        #             or all(
        #                 unit.game_board.factory_dict[f_id].is_own
        #                 for f_id in p2.adjacent_factories.keys()
        #             )
        #         ):
        #             safe_destinations.append(p2)

        # self.destinations = list(set(safe_destinations))
        self.target_bots = (
            [p.unit for p in targets] if self.target_bots is None else self.target_bots
        )

        return self.plan()

    def _target_achieved(self, candidate):
        # achieved if target reached, todo: add timing
        return True

    def post_process_action_queue(self, candidate, action_queue, skip_ping_pong=False):
        unit = self.unit
        pathfinder = candidate.pathfinder
        t = pathfinder.t

        # if self.is_intercept:
        #     recharge = unit.recharge(
        #         RECHARGE.REPLAN if unit.is_light else RECHARGE.REPLAN_HEAVY, n=1
        #     )
        #     action_queue.append(recharge)
        #     return action_queue

        # don't do special pattern if attacking light bot with heavy
        if self.target_bots and unit.is_heavy and self.target_bots[0].is_light:
            recharge = unit.recharge(RECHARGE.ATTACK_NO_FOLLOW, n=1)
            action_queue.append(recharge)
            return action_queue

        # check if we are the stronger bot
        target_bot = self.target_bots[0] if self.target_bots else None
        do_ping_pong = False
        if target_bot:
            enemy_power = (
                target_bot.power_in_time[t]
                if len(target_bot.power_in_time) > t
                else target_bot.power_in_time[-1]
            )
            if enemy_power > candidate.power:
                # enemy stronger than me, expect it to target me, so do pingpong pattern
                do_ping_pong = True

        own_center_id = unit.factory.center.id
        last_point = candidate.point
        power_left = candidate.power
        point = candidate.point

        power_to_return = pathfinder.p2p_power[point.id][own_center_id] + unit.comm_cost

        if not do_ping_pong:
            # check if deimos pattern
            if target_bot.is_heavy and target_bot.is_repeating_last_move:
                # check movement options for target bot
                aps = point.adjacent_points()

                valid = [
                    p
                    for p in aps
                    if len(
                        [
                            p2
                            for p2 in p.adjacent_points()
                            if p2.unit and p2.unit.is_heavy and p2.unit.is_own
                        ]
                    )
                    == 0
                    and p != candidate.parent.point
                    and p.closest_own_factory_distance > 1
                ]
                if valid:
                    min_rubble = min([p.rubble for p in valid])
                    options = [p for p in valid if p.rubble <= min_rubble + 10]
                    path_point = (
                        target_bot.path[t - 1]
                        if len(target_bot.path) > t
                        else target_bot.next_point
                    )
                    next_option = [p for p in options if p == path_point]

                    if next_option:
                        predicted_point = next_option[0]
                    else:
                        predicted_point = options[0]
                    direction = point.get_direction(predicted_point)
                    append_action = unit.move(direction, n=1)
                    append_action[3] = target_bot.id  # will be reset, but used later
                    point_after = predicted_point.apply(direction)
                    for i in range(10):
                        if point_after.rubble <= 20:
                            append_action[-1] = append_action[-1] + 1
                            point_after = point_after.apply(direction)

                    append_action[-2] = target_bot.id
                    action_queue.append(append_action)

                    return action_queue

            # use direction to enemy factory as direction to move
            enemy_factory = point.closest_enemy_factory
            # direction = point.get_directions(closest_tile)[0]
            # last_action[1] = direction

            # next_point = point.apply(direction)

            # find the optimal route that an optimal bot would take to retreat
            # for now assume bot is of same unit type, but if it isn't, let's just
            # pressure accordingly
            lowest_cost = 999999
            best_tile = None
            for tile in enemy_factory.edge_tiles():
                cost = pathfinder.p2p_power[point.id][tile.id]
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_tile = tile

            # don't include final point as it is factory
            optimal_path = get_optimal_path(self.game_board, point, best_tile, unit)[
                :-1
            ]
            if any(p.closest_own_factory_distance <= 3 for p in optimal_path):
                # I don't want this route near my factory
                optimal_path = get_rss_optimal_path(self.game_board, point, best_tile)[
                    :-1
                ]

                if any(p.closest_own_factory_distance <= 3 for p in optimal_path):
                    # I don't want this route near my factory
                    optimal_path = []
            commands = commands_from_path(optimal_path)

            # build up the post queue while checking if we can make it back
            for i, p in enumerate(
                optimal_path[1:]
            ):  # skip first point as it is the start
                rubble = p.rubble  # work with current rubble
                move_cost = unit.move_cost_rubble(rubble)
                charged = unit.charge_from_to(t + i, t + i + 1)
                power_left -= move_cost + charged

                power_to_return = (
                    pathfinder.p2p_power[p.id][own_center_id] + unit.comm_cost
                )

                if power_left <= power_to_return:
                    do_ping_pong = False
                    break

                direction = commands[i]
                if (
                    action_queue[-1][1] == direction
                    and action_queue[-1][3] == target_bot.id
                ):
                    action_queue[-1][-1] += 1
                else:
                    if len(action_queue) >= 20:
                        do_ping_pong = False
                        break
                    action_queue.append(
                        np.array([0, int(direction), 0, target_bot.id, 0, 1])
                    )
                last_point = p

        # pingpong if illegal next point, factory also exclude own factory
        # (next_point == candidate.point or next_point.factory)
        if do_ping_pong and not skip_ping_pong:
            last_action = action_queue[-1].copy()
            last_action[3] = target_bot.id  # encode attacking

            opposite_direction = OPPOSITE_DIRECTION[last_action[1]]
            next_point = last_point.apply(opposite_direction)
            assert IS_KAGGLE or next_point != last_point, "next point is same as last?"

            # ignore charging for now
            pingpong_cost = unit.move_cost_rubble(
                last_point.rubble
            ) + unit.move_cost_rubble(next_point.rubble)
            n_ping_pongs = int((power_left - power_to_return) // pingpong_cost)

            pong_space = (20 - len(action_queue)) // 2
            if pong_space <= 0:
                return action_queue

            # need to fit in the action queue.
            if n_ping_pongs > pong_space and n_ping_pongs > 3:
                last_action[-2] = 1  # make repeating
                last_action[-1] = 1  # make sure it is a single step

                inverse_action = last_action.copy()
                inverse_action[1] = OPPOSITE_DIRECTION[inverse_action[1]]

                action_queue.append(inverse_action)
                action_queue.append(last_action)
            else:
                for i in range(n_ping_pongs):
                    last_action[-2] = target_bot.id
                    last_action[-1] = 1  # make sure it is a single step

                    inverse_action = last_action.copy()
                    inverse_action[1] = OPPOSITE_DIRECTION[inverse_action[1]]

                    action_queue.append(inverse_action)
                    action_queue.append(last_action)
        # else:
        #     # continue in same direction
        #     last_action[-2] = 100  # make repeating
        #     last_action[-1] = 100  # used to count down and able to abort chase
        #     action_queue.append(last_action)
        action_queue = action_queue[:20]
        if action_queue[-1][-2] == 0:
            action_queue[-1][-2] = target_bot.id
        return action_queue

    def on_plan(self, best):
        assert IS_KAGGLE or self.target_bots, "No target bots provided in attack"
        if self.target_bots:
            for enemy_unit in self.target_bots:
                enemy_unit.dies = True
                enemy_unit.is_targeted = True
                if self.unit not in enemy_unit.targeted_by:
                    enemy_unit.targeted_by.append(self.unit)
                if self.unit not in enemy_unit.dies_by:
                    enemy_unit.dies_by.append(self.unit)
        self.unit.is_attacking = True

    def on_complete_valid(self, candidate, verbose=False):
        target_bots = self.target_bots
        if not target_bots:
            lprint("?????????????? No target bots")
            return True
        min_target_power = min([t.power for t in target_bots])
        target_bot = target_bots[0]
        if candidate.point != target_bot.point:
            if candidate.power < min_target_power:
                lprint(
                    f"Not enough power {candidate.power} left to pingpong with target ({min_target_power})"
                )
                return False
        return True


class Suicide(Action):
    max_waits = 0
    max_steps_off_target = 0
    reserve_power_to_return = False
    needs_to_return = False
    may_die_at_destination = True

    def post_process_action_queue(self, candidate, action_queue):
        action_queue[-1][-2] = 1
        return action_queue

    def _target_achieved(self, candidate):
        # achieved if target reached, todo: add timing
        return True


class TimedAttack(Attack):
    max_waits = 3
    max_steps_off_target = 1
    # disturb_is_good = False

    def update_targets(self, pathfinder):
        # do nothing by default, used for timed actions
        timed_targets = self.timed_targets
        t = pathfinder.t
        current_targets = pathfinder.targets
        pathfinder.targets = tuple(
            sorted([p for tt, p in timed_targets.items() if tt > t])
        )
        if current_targets != pathfinder.targets:
            pathfinder.min_distance_to_targets = {}
        return

    def _target_achieved(self, candidate):
        t = candidate.pathfinder.t
        timed_targets = self.timed_targets
        if t not in timed_targets:
            return False
        return candidate.point == timed_targets[t]

    def get_max_it(self):
        return max(self.timed_targets.keys()) if self.timed_targets else 0

    def post_process_action_queue(self, candidate, action_queue):
        unit = self.unit
        point = candidate.point
        pathfinder = candidate.pathfinder
        t = pathfinder.t

        if len(action_queue) <= 1:  # or not REMOVE_FINAL_HIT_FROM_QUEUE:
            return super().post_process_action_queue(
                candidate, action_queue, skip_ping_pong=True
            )

        # REMOVE_FINAL_HIT_FROM_QUEUE -- adds a fake queue instead
        last_action = action_queue[-1].copy()
        last_direction = last_action[1]
        point_before = candidate.path[-2]
        closest_enemy_factory_center = point.closest_enemy_factory.center

        enemy_unit_id = self.target_bots[0].id
        queue_to_factory = get_rss_path_queue(
            self.agent, point_before, closest_enemy_factory_center, repeat=enemy_unit_id
        )[
            :-2
        ]  # exclude last two commands which are at the factory

        random_dir = np.random.choice(
            [
                num
                for num in [1, 2, 3, 4]
                if num not in [last_direction, OPPOSITE_DIRECTION[last_direction]]
            ]
        )
        if not queue_to_factory:
            fake_queue = [
                np.array([0, random_dir, 0, 0, enemy_unit_id, np.random.randint(5, 10)])
            ]
        elif queue_to_factory[0][1] == last_direction:
            queue_to_factory[0][1] = random_dir
            fake_queue = queue_to_factory
        else:
            fake_queue = queue_to_factory

        # remove last action
        if REMOVE_FINAL_HIT_FROM_QUEUE:
            action_queue = action_queue[:-1]
        combined_queue = action_queue + fake_queue
        return combined_queue[:20]
        # return action_queue + fake_queue


class PipelineAttack(Attack):
    max_waits = 1
    max_steps_off_target = 2
    may_die_at_destination = True  # handled by unplan logic


class Ambush(Offense):
    max_steps_off_target = 2
    base_max_it = TTMAX
    stop_search_if_complete = True
    ambush_distance = 1
    beam_size = 10
    post_process_queue_len = 1
    reserve_power_to_return = True
    can_be_planned_when_retreating = False
    track_tries = True

    def get_max_it(self):
        targets = self.targets if self.targets else self.get_filtered_ambush_targets()
        if targets:
            return self.get_max_allowed_it(0, targets)
        return self.base_max_it

    def _target_achieved(self, candidate):
        # achieved if target reached, todo: add timing
        return True

    def post_process_action_queue(self, candidate, action_queue):
        """Add a wait to the end of the action queue to ensure other units
        don't plan themselves into the same spot"""
        recharge = self.unit.recharge(RECHARGE.AMBUSH)

        point = candidate.point

        if point.lichen and point.lichen_strains not in self.agent.own_lichen_strains:
            action_queue.append(self.unit.dig(n=1 if self.unit.is_heavy else 11))

        return action_queue + [recharge] if len(action_queue) < 20 else action_queue

    def get_filtered_ambush_targets(self, factory=None, verbose=False, **kwargs):
        lprint("get_filtered_ambush_targets, factory:", factory)
        if not hasattr(self, "_filtered_ambush_targets") or factory:
            self._filtered_ambush_targets = []

            ambush_targets = self.get_ambush_targets()
            unit = self.unit
            point = unit.point

            if verbose:
                lprint("ambush_targets", ambush_targets)

            if unit.is_light:
                ambush_targets = [
                    p
                    for p in ambush_targets
                    if not any(u.is_heavy and u.is_enemy for u in p.dug_by)
                ]

                if verbose:
                    lprint("ambush_targets", ambush_targets)

            if factory is not None:
                ambush_targets = [
                    p for p in ambush_targets if p.closest_enemy_factory == factory
                ]

            if verbose:
                lprint("b ambush_targets", ambush_targets)

            if len(ambush_targets) == 0:
                return []

            if any(
                p in ambush_targets
                for p in unit.last_point.points_within_distance(self.ambush_distance)
            ):
                # remove current point if it is at risk next turn
                if point in ambush_targets:
                    if self.game_board.attack_risk_in_time[1][
                        point
                    ] > unit.combat_strength(
                        True, unit.power
                    ):  # True, meaning under pressure of stronger target
                        ambush_targets = [p for p in ambush_targets if p != point]
                    # el
                    elif not (unit.dies or unit.could_die or unit.must_move):
                        lprint(f"{unit.unit_id}: keep ambushing {unit.point}")
                        unit.remove_from_work_queue()
                        return [unit.point]
                else:
                    if (
                        unit.point == unit.last_point
                        and not unit.dies
                        and not unit.could_die
                    ):
                        # we are already in a good ambush position
                        lprint(f"{unit.unit_id}: keep ambushing {unit.point}")
                        if not (unit.dies or unit.could_die or unit.must_move):
                            unit.remove_from_work_queue()
                        return [unit.point]

            point = unit.point
            ambush_targets = [
                target
                for target in ambush_targets
                if not unit.is_target_covered(target)
            ]

            camp_positions = flatten(
                [p.points_within_distance(self.ambush_distance) for p in ambush_targets]
            )
            if verbose:
                lprint("camp_positions before", camp_positions)

            camp_positions = sorted(
                [
                    p
                    for p in camp_positions
                    if p.factory is None
                    and len(
                        [
                            u
                            for u in p.visited_by
                            if u.is_own
                            and u.is_heavy
                            and u.last_point == p
                            and p in u.path
                            and u.path.index(p) <= point.distance_to(p) + 1
                            and (
                                power_required(unit, point, p)
                                < unit.battery_capacity * 0.4
                            )
                        ]
                    )
                    == 0
                ],
                key=lambda p: p.distance_to(point),
            )
            if verbose:
                lprint("camp_positions", camp_positions)
            if not camp_positions:
                return []

            distance = point.distance_to(camp_positions[0])

            if distance > self.base_max_it:
                return []

            # exclude cells adjacent to enemy factories, too risky, unless no risk or already there
            tiles = [
                p
                for p in camp_positions
                if not p.factory
                and (
                    not p.adjacent_factories
                    or (self.agent.opponent_heavy_bot_potential == 0 and unit.is_heavy)
                    or unit.point == p
                    # or (p.rubble == 0 and p in unit.point.adjacent_points())
                )
            ]
            if verbose:
                lprint("camp_positions tiles", tiles)

            # exclude the ambushtargets themselves
            tiles = list(set(tiles) - set(ambush_targets))
            self._filtered_ambush_targets = tiles

        return self._filtered_ambush_targets

    def execute(self, **kwargs):
        unit = self.unit
        point = unit.point
        # self.start_charge = point.factory is not None and not unit.full_power

        tiles = self.targets if self.targets else self.get_filtered_ambush_targets()
        if tiles:
            closest_factory = tiles[0].closest_own_factory
            self.destinations = closest_factory.edge_tiles(unit=unit, no_chargers=True)
            self.targets = tiles
            distance = point.distance_to(tiles[0])
            min_power = (distance + 4) * (100 if unit.is_heavy else 2)
            return self.plan(goal_amount=min_power, **kwargs)
        return False

    def get_ambush_targets(self):
        """Get the positions to target for an ambush, = ice spots"""
        game_board = self.game_board
        if game_board.ambush_positions is not None:
            return game_board.ambush_positions

        tiles = []
        heavy_cost = game_board.env_cfg.ROBOTS["HEAVY"].METAL_COST
        light_cost = game_board.env_cfg.ROBOTS["LIGHT"].METAL_COST
        ref_cost = (
            heavy_cost
            if len([u for u in self.agent.opponent_units if not u.is_heavy]) == 0
            else light_cost
        )

        for op_factory in game_board.agent.opponent_factories:
            bot_potential = (
                len(op_factory.units) > 0 or op_factory.cargo.metal >= ref_cost
            )
            if not bot_potential:
                continue

            closest = op_factory.all_ice_points[0]
            distance = closest.distance_to(op_factory)

            ice_candidates = set(op_factory.all_ice_points[:9])

            candidates = [
                p for p in ice_candidates if p.distance_to(op_factory) <= distance
            ]
            c_candidates = [c for c in candidates]  # if c.unit and c.unit.is_enemy]
            r_candidates = []
            if len(c_candidates) > 0:
                candidates = c_candidates
            else:
                r_candidates = [c for c in candidates if c.rubble == 0]
                if len(r_candidates) > 0:
                    candidates = r_candidates

            candidates = sorted(candidates, key=lambda p: p.rubble)

            tiles += candidates

        game_board.ambush_positions = tiles
        return tiles


class SentryAction(Action):
    stop_search_if_complete = True
    max_steps_off_target = 2
    needs_to_return = False
    needs_targets = True
    free_time = 8
    track_tries = True

    def on_complete_valid(self, candidate, verbose=False):
        point = candidate.point
        pf = candidate.pathfinder
        unit = self.unit
        for i in range(self.free_time):
            other_units = set(
                [
                    u
                    for u in set(pf.unit_grid_in_time[pf.t + i][point]) - set([unit])
                    if u >= unit
                ]
            )

            if len(other_units - set(pf.target_bots)) > 0:
                if verbose:
                    lprint(
                        f"Position {candidate.point} not free ({other_units}) at t={pf.t + i -1}",
                    )
                return False
        return True

    def _target_achieved(self, candidate):
        return True

    def filter_sentry_targets(self, candidates, verbose=False):
        if verbose:
            lprint("filter_sentry_targets before:", candidates)
        unit = self.unit
        point = unit.point

        risk_grid = self.game_board.attack_risk_in_time

        my_strength = unit.combat_strength(has_moved=True, power_left=unit.power)

        candidates = [
            p
            for p in candidates
            # must not be camped by a unit that is stronger than me
            if len([u for u in p.visited_by if u >= unit and u.last_point == p]) == 0
            and len(  # must not be adjacent to enemy
                [
                    ap
                    for ap in p.adjacent_points()
                    if (
                        ap.unit
                        and ap.unit.last_point == ap
                        and ap.unit.is_enemy
                        and ap.unit >= unit
                    )
                ]
            )
            == 0
            and len(  # must be able to get there
                [
                    ap
                    for ap in p.adjacent_points()
                    if risk_grid[ap.distance_to(point)][point] < my_strength
                ]
            )
            > 0
        ]
        if verbose:
            lprint("filter_sentry_targets after:", candidates)

        return candidates


class Guard(SentryAction):
    def post_process_action_queue(self, candidate, action_queue):
        action_queue.append(self.unit.recharge(RECHARGE.GUARD, n=1))
        return action_queue

    def get_targets(self):
        factory = self.unit.factory
        if not factory.ice_hubs:
            return []

        if not factory.full_power_hub():
            return []

        n_guards = len([u for u in factory.heavies if u.ends_recharge(RECHARGE.GUARD)])
        if n_guards:
            return []

        target = factory.ice_hubs[0]
        targets = sorted(
            [
                p
                for p in target.points_within_distance(2)
                if not p.ice and not p.ore and not p.unit and not p.factory
            ],
            key=lambda p: p.closest_enemy_factory_distance,
        )
        return targets[:2]


class Net(SentryAction):
    max_steps_off_target = 1
    needs_to_return = False
    reserve_power_to_return = True
    beam_size = 10
    recharge_encoding_light = RECHARGE.CLOSE_FACTORY
    recharge_encoding_heavy = RECHARGE.CLOSE_FACTORY

    def get_max_it(self):
        targets = self.targets if self.targets else self.get_targets()
        if targets:
            return self.get_max_allowed_it(0, targets)
        return self.base_max_it

    def post_process_action_queue(self, candidate, action_queue):
        recharge_target = (
            self.recharge_encoding_light
            if self.unit.is_light
            else self.recharge_encoding_heavy
        )
        action_queue.append(self.unit.recharge(recharge_target, n=1))
        return action_queue


class Killnet(Net):
    recharge_encoding_light = RECHARGE.KILLNET
    recharge_encoding_heavy = RECHARGE.KILLNET

    def get_targets(self):
        unit = self.unit

        candidates = self.agent.get_killnet_targets()
        candidates = self.filter_sentry_targets(candidates)
        candidates = sorted(candidates, key=lambda p: p.distance_to(unit.point))

        if not candidates:
            return []

        closest = candidates[0]

        close_candidates = sorted(
            candidates[:10], key=lambda p: p.distance_to(closest)
        )[:4]

        return close_candidates

    def on_plan(self, best):
        self.agent.update_killnet_action(best.point)


class Shield(Net):
    recharge_encoding_light = RECHARGE.LIGHT_SHIELD
    recharge_encoding_heavy = RECHARGE.HEAVY_SHIELD

    def get_targets(self, verbose=False):
        unit = self.unit
        candidates = sorted(
            self.agent.get_shield_targets(),
            key=lambda p: p.distance_to(unit.point)
            + 1.5 * p.closest_enemy_factory_distance,
        )

        if not candidates:
            if verbose:
                lprint("No shield candidates")
            return []

        closest = sorted(candidates, key=lambda p: p.distance_to(unit.point))[0]

        if unit.point.distance_to(closest) > 30:
            return []

        close_candidates = sorted(
            [c for c in candidates[:10] if c.distance_to(closest) <= 3],
            key=lambda p: p.distance_to(closest),
        )[:4]

        return close_candidates

    def on_plan(self, best):
        self.agent.update_shield_action(best.point)


class LichenShield(Net):
    recharge_encoding_light = RECHARGE.LIGHT_SHIELD
    recharge_encoding_heavy = RECHARGE.HEAVY_SHIELD

    def get_targets(self):
        unit = self.unit
        all_candidates = self.agent.get_lichen_shield_targets(unit=unit)
        candidates = [p for p in all_candidates if p.lichen_strains == unit.factory.id]

        if not candidates:
            candidates = all_candidates

        candidates = sorted(
            all_candidates,
            key=lambda p: p.closest_enemy_factory_distance,
        )

        if not candidates:
            return []

        closest = candidates[0]

        if unit.point.distance_to(closest) > 20:
            return []

        close_candidates = sorted(
            candidates[:10], key=lambda p: p.distance_to(closest)
        )[:4]

        return close_candidates

    def on_plan(self, best):
        self.agent.update_lichen_shield_action(best.point, self.unit)


class Camp(SentryAction):
    base_max_it = TTMAX
    stop_search_if_complete = True
    # may_die_at_destination = True
    max_steps_off_target = 1
    beam_size = 10
    can_be_planned_when_retreating = False
    reserve_power_to_return = True

    def get_max_it(self):
        targets = self.targets if self.targets else self.get_targets()
        if targets:
            return self.get_max_allowed_it(0, targets)
        return self.base_max_it

    def get_targets(self, **kwargs):
        opponent_factories = sorted(
            self.agent.opponent_factories, key=self.unit.point.distance_to
        )
        unit = self.unit
        point = unit.point
        unit_grid = self.game_board.unit_grid_in_time

        def is_free(p):
            distance = p.distance_to(point)

            for dt in range(4):
                if p not in unit_grid[distance + dt]:
                    continue
                units_at_t = unit_grid[distance + dt][p]
                if len(units_at_t) == 1 and unit in unit_grid[distance + dt][p]:
                    return True
                if len(units_at_t) > 0 and any(u.is_own for u in units_at_t):
                    return False
            return True

        def point_camped(point):
            return (
                point.unit
                and point.unit.is_repeating
                and point.unit.last_point == point
                and point.unit.is_own
            )

        steps_left = self.game_board.steps_left
        targets = []
        for f in opponent_factories:
            candidates = [
                p
                for p in f.neighbour_tiles()
                if not p.ice
                and not p.ore
                and not p.lichen
                and p.rubble <= 2
                and not point_camped(p)
                and (is_free(p) or steps_left < 25)
            ]
            targets += candidates
            if candidates:
                break
        targets = sorted(targets, key=lambda p: p.distance_to(unit.point))[:5]
        return targets

    def post_process_action_queue(self, candidate, action_queue):
        marking = (
            RECHARGE.CLOSE_FACTORY
        )  # if self.unit.is_light else RECHARGE.CLOSE_FACTORY
        action_queue.append(self.unit.recharge(marking, n=1))

        return action_queue


class AmbushRSS(SentryAction):
    may_die_at_destination = False  # will be handled by unplan logic
    reserve_power_to_return = True
    beam_size = 15
    max_waits = 1
    needs_targets = True
    post_process_queue_len = 1
    base_max_it = TTMAX
    penalize_move = True

    def get_targets(self, **kwargs):
        rss_targets = []
        unit = self.unit
        # if unit.is_heavy:
        #     return []

        point = unit.point

        # select target rss
        for f in self.agent.opponent_factories:
            f_scarce = []
            if f.is_ore_risk():
                f_scarce += f.ore_points
            if f.is_ice_risk():
                f_scarce += f.ore_points

            candidates = sorted(
                [
                    p
                    for p in f_scarce
                    if p.closest_factory.is_enemy
                    and (
                        p.distance_to(f) <= 12
                        and (p.distance_to(f) > 3 or unit.point.distance_to(p) <= 4)
                    )
                    and (p.rubble == 0 or p.dug_by)
                    and not any(u > unit for u in p.dug_by)
                ],
                key=lambda p: p.distance_to(f),
            )

            # just don't block rss if enemy has plenty of options
            if (
                len(
                    [
                        c
                        for c in candidates
                        if c.distance_to(f) <= 7 and c.closest_factory.is_enemy
                    ]
                )
                > 2
            ):
                return []

            if len(candidates) > 2:  # cannot block too much
                candidates = [p for p in candidates if p.distance_to(f) <= 6]

            if len(candidates) <= 2:  # cannot block too much
                rss_targets += sorted(candidates[:2], key=lambda p: p.distance_to(f))

        stay_combat_strength = unit.combat_strength(False, unit.power)

        def candidate_score(p):
            adj_points_p = p.adjacent_points() + [p]
            n_covered = len([ap for ap in adj_points_p if ap.ore or ap.ice])
            distance = p.distance_to(point)
            distance_to_enemy = p.closest_enemy_factory_distance
            safe = (p != point) and self.game_board.attack_risk_in_time[1][
                p
            ] < stay_combat_strength
            return (safe, n_covered, -distance, distance_to_enemy)

        # only if my factory is closest, or if we are already close
        def is_covered(p, adj_points_p):
            covered = any(
                [
                    len(
                        [
                            u
                            for u in ap.visited_by
                            if u.is_own
                            and u.last_point == ap
                            and u != unit
                            and u.is_heavy != unit.is_heavy
                        ]
                    )
                    > 0
                    for ap in adj_points_p
                ]
            )
            if covered:
                return True

            nearby_points = p.points_within_distance(3)
            unit_nearby = (
                len(
                    [
                        p
                        for p in nearby_points
                        if p.unit
                        and p.unit.is_own
                        and p.unit != unit
                        and p.unit.is_heavy != unit.is_heavy
                        and p.unit.power > p.unit.init_power * 2
                    ]
                )
                > 0
            )

            return unit_nearby

        candidates = []
        for p in rss_targets:
            if p.distance_to(point) < 5 or p.closest_own_factory == unit.factory:
                adj_points_p = p.adjacent_points() + [p]
                if is_covered(p, adj_points_p):
                    continue  # already camping
                else:
                    # only select the best from the adjacent points
                    candidates.append(
                        sorted(adj_points_p, key=candidate_score, reverse=True)[0]
                    )

        candidates = sorted(candidates, key=candidate_score, reverse=True)

        if not candidates:
            return []

        # search only in same area
        closest = candidates[0]
        return [p for p in candidates if p.distance_to(closest) <= 4][:2]

    def post_process_action_queue(self, candidate, action_queue):

        action_queue.append(self.unit.recharge(RECHARGE.AMBUSH_RSS, n=1))

        return action_queue


class Hub(SentryAction):
    base_max_it = 25
    light_dig_count = None
    heavy_dig_count = None
    needs_to_return = False
    post_process_queue_len = 2
    max_power_pickup = 400
    reserve_power_to_return = True
    needs_targets = True
    digs_to_transfer = None
    base_dig_frequency = None
    base_dig_frequency_push = None
    penalize_move = True
    free_time = 2

    def current_cargo(self):
        return 0

    def get_n_digs(self, candidate):
        unit = self.unit
        factory = candidate.point.closest_own_factory

        if factory.power_hub_push:
            if factory.heavies_for_full_power_hub():
                return self.base_dig_frequency
            return self.base_dig_frequency_push

        if (
            unit.is_light
            or factory.n_connected_x_plus >= BREAK_EVEN_ICE_TILES
            or (len(factory.ice_hubs) > 0 and factory.lake_large_enough())
            or candidate.power > 1000
            or factory.available_power() > 1000
            or (
                factory.available_water() < 50
                and any(
                    u.point.distance_to(factory) < 8
                    for u in self.agent.opponent_heavies
                )
            )
        ):
            return self.digs_to_transfer
        return self.base_dig_frequency  # such that one ore cycle can create a light bot

    def post_process_action_queue(self, candidate, action_queue):
        factory = candidate.point.closest_own_factory

        n_digs = self.get_n_digs(candidate)

        dig = self.unit.dig(repeat=n_digs, n=n_digs)
        transfer_direction = candidate.point.adjacent_factories[
            candidate.point.closest_own_factory.unit_id
        ]
        dig_amount = self.unit.unit_cfg.DIG_RESOURCE_GAIN
        amount = dig_amount * n_digs
        transfer = self.unit.transfer(
            transfer_direction, self.resource_id, amount, repeat=1
        )

        if candidate.point.ice:
            dig_ten = self.unit.dig(n=10)
            transfer_ten = self.unit.transfer(
                transfer_direction,
                self.resource_id,
                10 * dig_amount,
            )

            if factory.out_of_water_time() - candidate.pathfinder.t - n_digs - 1 < 0:
                dig_once = self.unit.dig(n=1)
                transfer_once = self.unit.transfer(
                    transfer_direction,
                    self.resource_id,
                    dig_amount,
                )
                dig_twice = self.unit.dig(n=2)
                transfer_twice = self.unit.transfer(
                    transfer_direction,
                    self.resource_id,
                    2 * dig_amount,
                )
                dig_three = self.unit.dig(n=3)
                transfer_three = self.unit.transfer(
                    transfer_direction,
                    self.resource_id,
                    3 * dig_amount,
                )
                action_queue += (
                    [dig_once]
                    + [transfer_once]
                    + [dig_twice]
                    + [transfer_twice]
                    + [dig_three]
                    + [transfer_three]
                    + [dig_ten]
                    + [transfer_ten]
                )
            elif n_digs > self.base_dig_frequency:
                if factory.available_water() < 100:
                    dig_ten = self.unit.dig(n=10)
                    transfer_ten = self.unit.transfer(
                        transfer_direction,
                        self.resource_id,
                        10 * dig_amount,
                    )

                    action_queue += (
                        [dig_ten]
                        + [transfer_ten]
                        + [dig_ten]
                        + [transfer_ten]
                        + [dig_ten]
                        + [transfer_ten]
                    )

            if factory.power_hub_push and self.game_board.step < 75:
                # transfer excess power to the hub
                action_queue += [
                    self.unit.transfer(transfer_direction, 4, 150, repeat=0)
                ]

        current_cargo = self.current_cargo()
        if current_cargo:
            action_queue += [
                self.unit.transfer(
                    transfer_direction, self.resource_id, current_cargo, repeat=0
                )
            ]

        action_queue += [dig] + [transfer]

        if candidate.point.ice and candidate.point.closest_own_factory.power_hub_push:
            action_queue.append(np.array([0, 0, 0, 0, WAIT_TIME_ICE, WAIT_TIME_ICE]))

        assert (
            IS_KAGGLE or len(action_queue) <= self.unit.env_cfg.UNIT_ACTION_QUEUE_SIZE
        ), f"len(action_queue)={len(action_queue)} for {self.unit} {self}"
        return action_queue

    def is_my_resource(point, self):
        return None

    def get_targets(self, **kwargs):
        step = self.game_board.step
        point = self.unit.point
        targets = [
            p
            for p in self.unit.factory.hubs
            if self.is_my_resource(p)
            # - 1 since dig effect is administrated one turn before
            and (
                step > 25
                or not p.dug_by
                or is_skimmed(
                    p.get_rubble_at_time(point.distance_to(p) + self.start_charge - 2)
                )
            )
            and len([u for u in p.dug_by if u.is_heavy]) == 0
            and (
                p.ice
                or len(p.closest_own_factory.heavy_ice_hubs) > 0
                or p.closest_own_factory.out_of_water_time()
                > UNPLAN_ORE_OUT_OF_WATER_TIME
            )
            and len(
                [
                    p
                    for p in p.points_within_distance(3)
                    if p.unit
                    and p.unit.is_heavy
                    and p.unit.is_enemy
                    and not (not p.unit.is_digging or p.unit.digs_lichen)
                ]
            )
            == 0
        ]

        return targets


class IceHub(Hub):
    resource_id = RESOURCE_MAP["ice"]
    digs_to_transfer = DIGS_TO_TRANSFER_ICE
    base_dig_frequency = BASE_DIG_FREQUENCY_ICE
    base_dig_frequency_push = BASE_DIG_FREQUENCY_ICE

    def is_my_resource(self, point):
        return point.ice

    def current_cargo(self):
        return self.unit.cargo.ice


class OreHub(Hub):
    resource_id = RESOURCE_MAP["ore"]
    digs_to_transfer = DIGS_TO_TRANSFER_ORE
    base_dig_frequency = BASE_DIG_FREQUENCY_ORE
    base_dig_frequency_push = BASE_DIG_FREQUENCY_ORE_PUSH

    def is_my_resource(self, point):
        return point.ore

    def current_cargo(self):
        return self.unit.cargo.ore


class PowerHub(Hub):
    resource_id = RESOURCE_MAP["power"]
    post_process_queue_len = 1
    free_time = 5

    def get_targets(self, **kwargs):
        point = self.unit.point
        targets = [
            p
            for p in self.unit.factory.get_power_hub_positions()
            if p.rubble == 0
            and (
                not p.unit
                or p.unit.last_point != p
                or p.unit.can_be_replanned()
                or p.unit == self.unit
                or p.unit < self.unit
            )
            and not any(
                u
                for u in p.visited_by
                if u.is_power_hub and u.last_point == p and u != self.unit
            )
            # and is_free(self, self.unit, p, point.distance_to(p) + 1)
        ]
        outside_targets = [t for t in targets if t.factory is None]
        if outside_targets:
            targets = outside_targets
        return targets

    def post_process_action_queue(self, candidate, action_queue):
        gb = self.unit.game_board
        transfer_direction = (
            0
            if candidate.point.factory
            else candidate.point.adjacent_factories[
                candidate.point.closest_own_factory.unit_id
            ]
        )

        # amount = self.unit.avg_power_production_per_tick()
        # transfer = self.unit.transfer(
        #     transfer_direction, self.resource_id, amount, repeat=1
        # )

        power_at_bot = candidate.power
        # don't leave more than 250 power in bot
        transfer_first_amount = power_at_bot - (60 if candidate.point.factory else 120)
        if transfer_first_amount > 0:
            transfer_first = self.unit.transfer(
                transfer_direction, self.resource_id, transfer_first_amount, repeat=0
            )
            action_queue.append(transfer_first)

        shift = candidate.pathfinder.t

        schedule = gb.charge_rate
        n_nights = np.argmax(schedule)

        if shift > n_nights:
            shift -= n_nights
            n_nights = 0
        else:
            n_nights -= shift
            shift = 0

        n_days = 0
        amount = 10
        if n_nights:
            action_queue.append(np.array([0, 0, 0, 0, 20, int(n_nights)]))
            transfer = self.unit.transfer(
                transfer_direction, self.resource_id, amount, repeat=30, n=30
            )
            action_queue.append(transfer)
        else:
            n_days = np.argmax(np.logical_not(schedule))
            if shift > n_days:
                shift -= n_days
                n_days = 0
            else:
                n_days -= shift
                shift = 0

            if n_days:
                transfer = self.unit.transfer(
                    transfer_direction, self.resource_id, amount, repeat=30, n=n_days
                )
                action_queue.append(transfer)

            else:
                transfer = self.unit.transfer(
                    transfer_direction, self.resource_id, amount, repeat=30, n=30
                )
                action_queue.append(transfer)
            action_queue.append(np.array([0, 0, 0, 0, 20, 20]))

        # action_queue.append(transfer)

        assert (
            IS_KAGGLE or len(action_queue) <= self.unit.env_cfg.UNIT_ACTION_QUEUE_SIZE
        ), f"len(action_queue)={len(action_queue)} for {self.unit} {self}"
        return action_queue

    def on_plan(self, best):
        self.unit.is_power_hub = True
        self.unit.was_power_hub


def get_active_hub_units(factory):
    active_hubs = [
        p.dug_by[-1]
        for p in factory.ice_hubs + factory.ore_hubs
        for digger in ((p.dug_by[-1],) if p.dug_by else [])
        if p.dug_by
        if (
            p.dug_by
            and digger.is_own
            and digger.is_hub
            and digger.last_point == p
            and not digger.dies
            and not digger.could_die
            and digger.power < digger.init_power * 1.5
        )  # to prevent too much useless power in hub
    ]
    return active_hubs


class Charger(SentryAction):
    base_max_it = 10
    penalize_move = True
    post_process_queue_len = 10
    needs_targets = True
    needs_to_return = False
    reserve_power_to_return = False
    timed_targets_need_move = False
    max_waits = 4

    def get_max_it(self):
        targets = self.targets if self.targets else self.get_targets()
        if targets:
            return self.get_max_allowed_it(0, targets)
        return self.base_max_it

    @classmethod
    def get_valid_charger_positions(
        cls, factory, allow_heavy_overtake=False, verbose=False
    ):
        units = get_active_hub_units(factory)

        if not units:
            return None, None

        active_hubs = [u.last_point for u in units]
        min_t = min([u.path.index(u.last_point) for u in units])

        targets = [p.apply(p.adjacent_factories[factory.unit_id]) for p in active_hubs]

        if verbose:
            lprint(f"Charger active_hubs:{active_hubs}, targets: {targets}")
        targets = [
            p
            for p in targets
            if not p.unit
            or not (p.unit.is_repeating and p == p.unit.last_point)
            or not p.unit.has_queue
            or (allow_heavy_overtake and p.unit.is_light)
        ]

        if min_t:
            timed_targets = {t: targets for t in range(min_t, TMAX)}
        else:
            timed_targets = None

        return targets, timed_targets

    def get_targets(self, **kwargs):
        factory = self.unit.factory
        return self.get_valid_charger_positions(factory)[0]

    def power_required_per_tick(self, hub_unit):
        return hub_unit.power_required_per_tick()

    def post_process_action_queue(self, candidate, action_queue):
        # detect active hubs
        hub_units = []
        for ap in candidate.point.adjacent_points():
            if ap.ice or ap.ore:
                hub_diggers = [
                    u
                    for u in ap.visited_by
                    if u.is_own
                    and u.is_hub
                    and u.last_point == ap
                    and not u.dies
                    and not u.could_die
                ]
                if hub_diggers:
                    hub_units.append(hub_diggers[-1])

        # hub_units = [
        #     p.dug_by[-1]
        #     for p in
        #     if (p.ice or p.ore)
        #     and p.dug_by
        #     and p.dug_by[-1].is_hub
        #     and p == p.dug_by[-1].last_point
        #     and not p.dug_by[-1].dies
        #     and not p.dug_by[-1].could_die
        # ]

        factor = 1 + len(hub_units)
        total = 0
        actions = []
        unit = self.unit

        # charger_full = unit.full_power
        need_power_hub = None
        need_power_hub_direction = None
        # lprint(">>hub_units", hub_units)
        for target_unit in hub_units:
            power_required = self.power_required_per_tick(target_unit)

            # lprint(
            #     ">>target_unit",
            #     candidate.point,
            #     target_unit,
            #     target_unit.last_point,
            #     target_unit.last_point.adjacent_factories,
            # )
            opp_direction = target_unit.last_point.adjacent_factories[
                candidate.point.factory.unit_id
            ]
            direction = OPPOSITE_DIRECTION[opp_direction]

            amount = factor * power_required
            total += amount

            lprint(
                f"need_power_hub {need_power_hub} -> {target_unit} "
                f"PWR:{target_unit.power_in_time[candidate.pathfinder.t]}"
            )

            if (
                self.unit.is_heavy
                or target_unit.power_in_time[candidate.pathfinder.t]
                < 0.5 * target_unit.init_power
            ):
                if (
                    need_power_hub is None
                    or target_unit.power_in_time[candidate.pathfinder.t]
                    < need_power_hub.power_in_time[candidate.pathfinder.t]
                ) and (
                    self.game_board.step > 75
                    or unit.is_heavy
                    or not candidate.point.closest_own_factory.power_hub_push
                ):
                    # lprint(
                    #     f"need_power_hub {need_power_hub} -> {target_unit} "
                    #     f"PWR:{target_unit.power_in_time[candidate.pathfinder.t]}"
                    # )
                    need_power_hub = target_unit
                    need_power_hub_direction = direction

            # if charger_full and hub.unit.power < (
            #     hub.unit.unit_cfg.BATTERY_CAPACITY - 4 * amount
            # ):
            #     actions.append(np.array([1, direction, 4, math.ceil(2 * amount), 0, 1]))
            actions.append(np.array([1, direction, 4, math.ceil(amount), 1, 1]))

        # pickup power from factory, then transfer
        unit_charge = unit.avg_power_production_per_tick()
        base_charger_power = (
            unit.init_power / 2 if unit.is_light else unit.init_power * 0.4
        )

        candidate_power = int(candidate.power)
        if factor * unit_charge < total:
            total = total - factor * unit_charge
            pickup = np.array([2, 0, 4, math.ceil(total), 1, 1])
            # if candidate_power > (total + base_charger_power):
            actions.append(pickup)
            # else:
            #     action_queue.append(pickup)
        else:
            # charging is more than consumption. in case of heavy
            for action in actions:
                action[3] = math.ceil(action[3] / factor * (factor - 1))

        if (
            need_power_hub
            and need_power_hub.power_in_time[candidate.pathfinder.t] < 120
        ):
            if candidate_power > 120:
                action_queue.append(
                    np.array(
                        [1, need_power_hub_direction, 4, candidate_power - 1, 0, 1]
                    )
                )
                candidate_power = 1
            else:
                action_queue.append(np.array([2, 0, 4, 149 - candidate_power, 0, 1]))
                action_queue.append(
                    np.array([1, need_power_hub_direction, 4, 148, 0, 1])
                )
                candidate_power = 1

        # we always start with transfer, but could be that bot already has some power so we need to pickup first
        amount_needed = (total + base_charger_power) - candidate_power
        if amount_needed > 0:
            pickup_amount = min(
                unit.battery_capacity - candidate_power,
                math.ceil(amount_needed),
            )
            action_queue.append(
                np.array(
                    [
                        2,
                        0,
                        4,
                        pickup_amount,
                        0,
                        1,
                    ]
                )
            )
            candidate_power += pickup_amount

        # if have too much, transfer to factory
        overflow_direction = need_power_hub_direction or 0
        lprint(
            f"overflow_direction: PWR ={candidate_power} total={total} {overflow_direction} {need_power_hub_direction} "
            f"{amount_needed} {base_charger_power // 3} {-amount_needed + base_charger_power // 3}"
        )
        overflow_transfer = 0
        if amount_needed < 0:
            # remove a bit more from the charger to make sure we don't overflow
            overflow_transfer = math.ceil(-amount_needed + base_charger_power // 3)
            action_queue.append(
                np.array(
                    [
                        1,
                        overflow_direction,
                        4,
                        overflow_transfer,
                        0,
                        1,
                    ]
                )
            )
            candidate_power -= overflow_transfer

        def get_reserve_power():
            schedule = self.agent.game_board.charge_rate
            n_nights = np.argmax(schedule)
            n_days = np.argmax(np.logical_not(schedule))
            if n_nights:
                return 120 + 10 * n_nights
            return 310 - 6 * n_days

        # power sitting in a hub is wasted, burn it first to a minimum level
        burn_turns = 0
        if unit.is_light:
            assert IS_KAGGLE or hub_units, "light unit without hub??"
            if hub_units:
                min_power = min(
                    [u.power_in_time[candidate.pathfinder.t] for u in hub_units]
                )
                RESERVE_POWER = get_reserve_power()
                if min_power > RESERVE_POWER:
                    burn_turns = (min_power - RESERVE_POWER) // power_required  # 60
        else:
            burn_power = overflow_transfer - 150
            if overflow_transfer and burn_power > 0:
                burn_turns = burn_power // power_required - 1

        burn_turns = int(burn_turns)
        if burn_turns > 0:

            charged = sum(self.agent.game_board.charge_rate[: burn_turns + 1])
            charged_power = int(charged * unit.charge)

            if unit.is_heavy:
                transfer_now = min(charged_power, candidate_power - 150)
                charged_power -= max(transfer_now, 0)

                if transfer_now > 0:
                    # we have some power due to waiting, transfer it to the factory
                    action_queue.append(np.array([1, 0, 4, int(transfer_now), 0, 1]))

            action_queue.append(np.array([0, 0, 0, 0, 0, int(burn_turns) + 1]))
            if unit.is_heavy:
                if charged_power > 0:
                    action_queue.append(np.array([1, 0, 4, int(charged_power), 0, 1]))

        # deal with difference in production of light and charge in turns,
        # which is 1 diff, but 1.2 charge, so every 5 turns we need to reduce 1

        if unit.is_light and len(hub_units) > 1:
            queue = actions * 4
            if actions[-1][0] == 1:  # transfer
                pickup = actions[-2].copy()
                transfer = actions[-1].copy()
                queue = queue + [pickup, transfer]
            else:
                pickup = actions[-1].copy()
                transfer = actions[-2].copy()
                queue = queue + [transfer, pickup]
            pickup[-3] = pickup[-3] - 1

            action_queue += queue
        else:
            action_queue += actions

        assert (
            IS_KAGGLE or len(action_queue) <= self.unit.env_cfg.UNIT_ACTION_QUEUE_SIZE
        ), f"len(action_queue)={len(action_queue)} for {self.unit} {self}"
        return action_queue


def is_free(self, unit: Unit, p: Point, t_max=TMAX):
    unit_grid_in_time = self.game_board.unit_grid_in_time

    # can be occupied at t=0
    for t in range(1, t_max + 1):
        if p in unit_grid_in_time[t] and (
            len([u for u in unit_grid_in_time[t][p] if u != unit and u >= unit]) > 0
        ):
            return False
    return True


class Retreat(Action):
    max_waits = 0
    factory_as_destinations = True
    stop_search_if_complete = True
    base_max_it = TTMAX
    beam_size = 5
    max_steps_off_target = 1
    ignore_power_req_end = True
    post_process_queue_len = 1
    plan_even_if_not_complete = True

    def _is_complete(self, candidate):
        assert (
            IS_KAGGLE or self.unit.point.factory is None
        ), f"{self.unit}: retreat from factory!?"
        return (
            candidate.point in self.destinations
            or candidate.pathfinder.t >= self.max_it
        )

    def post_process_action_queue(self, candidate, action_queue):
        recharge = self.unit.recharge(RECHARGE.RETREAT)
        if len(action_queue) >= 20:
            action_queue = action_queue[:19]
        return action_queue + [recharge]


class ChargeAtFactory(Action):
    needs_targets = True
    stop_search_if_complete = True
    factory_as_destinations = False  # will check for factory with most power
    max_charges = 4
    needs_to_return = False
    penalize_move = True
    beam_size = 15
    max_waits = 2
    min_charge_factor = 7.5  # charging less is not worth it
    max_steps_off_target = 1
    ignore_power_req_end = True

    def get_max_it(self):
        targets = tuple(self.get_targets())
        if not targets:
            return 0
        return self.get_max_allowed_it(self.max_charges, targets)

    def prune_function(self, candidate):
        return (candidate.distance_to_target(), -candidate.score)

    def get_targets(self, **kwargs):
        unit = self.unit
        factory = unit.factory
        if not factory:
            return []

        steps_left = self.agent.game_board.steps_left

        if factory.available_power() < unit.battery_capacity:
            distance = 6 if unit.is_light else 4
            other_factories = sorted(
                [
                    f
                    for f in self.agent.factories
                    if f != factory
                    and f.available_power() > unit.battery_capacity
                    and f.available_power() > 500
                    and f.distance_to(unit.point) <= distance
                    and (not factory.power_hub_push or f.full_power_hub())
                ],
                key=unit.point.distance_to,
            )
            if other_factories:
                factory = other_factories[0]

        if (
            not unit.dies
            and unit.is_light
            and factory.available_power() < unit.init_power // 2
        ):
            return []

        if factory.distance_to(unit.point) >= steps_left:
            return []

        return factory.edge_tiles(unit=unit, no_chargers=True)

    def _get_on_target_actions(self, candidate):
        dirs = self._get_to_target_actions(candidate, optimal=False)
        if candidate.point.factory:
            dirs = dirs + [ACTION.CHARGE]
        return dirs

    def _target_achieved(self, candidate):
        # find number of occurrence of CHARGE in commands
        charge_count = candidate.on_goal_action
        charged_enough = (charge_count >= 1) and (
            (
                sum(candidate.pickups.values())
                >= self.min_charge_factor * self.unit.unit_cfg.MOVE_COST
            )
            or candidate.power >= self.unit.unit_cfg.BATTERY_CAPACITY - 1
            or self.unit.under_attack_dies_at_t
        )

        return charged_enough and (
            candidate.power >= self.unit.unit_cfg.BATTERY_CAPACITY * 0.9
            or candidate.pathfinder.t >= candidate.pathfinder.max_it
            or charge_count >= self.max_charges
            or (not candidate.point.factory or candidate.get_power_at_factory() <= 0)
        )


class Recharge(Action):
    needs_to_return = False
    max_waits = 100
    penalize_move = True

    def get_targets(self, **kwargs):
        unit = self.unit
        point = unit.point
        points = point.points_within_distance(10)
        points = [
            p
            for p in points
            if (p.closest_factory_distance > 5)
            and not p.ice
            and not p.ore
            and not p.unit
            and p.closest_factory.is_own
            and is_free(self, self.unit, p)
            and len(
                [
                    p2
                    for p2 in p.surrounding_points()
                    if p2.unit and (p2.unit.is_repeating or p2.unit.is_charging)
                ]
            )
            == 0
        ]

        # first without rubble
        targets = [p for p in points if not p.rubble]

        if not targets:
            targets = points

        targets = sorted(targets, key=lambda p: p.closest_factory_distance)[:5]
        return targets

    def post_process_action_queue(self, candidate, action_queue):
        target_power = self.unit.unit_cfg.BATTERY_CAPACITY * 0.95

        if candidate.power > target_power:
            return action_queue

        action_queue.append([5, 0, 0, int(target_power), 0, 1])

        return action_queue
