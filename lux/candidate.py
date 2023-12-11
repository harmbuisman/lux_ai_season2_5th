import sys
from collections import defaultdict

from lux.board import Point
from lux.constants import (
    ACTION,
    DIRECTION_ACTION_MAP,
    IS_KAGGLE,
    MOVES,
    RESOURCE_MAP,
    STATE,
    TTMAX,
    VALIDATE,
)
from lux.unit import Unit
from lux.utils import lprint
from lux.validations import (
    is_attack_risk,
    is_enemy_factory,
    is_out_of_power,
    is_power_at_factory,
    is_attack_risk_no_queue,
    invalid_on_target,
)

ACTION_MAP = {
    0: "move",
    1: ACTION.TRANSFER,
    2: ACTION.TRANSFER,
    3: ACTION.DIG,
    4: "self_destruct",
    5: "recharge",
}
MAP_SIZE = 48
GOAL_DIG_SCORE = 17


def move(point: Point, direction: tuple):
    dx, dy = DIRECTION_ACTION_MAP[direction]

    return point.apply((dx, dy))


class Candidate:
    def __init__(
        self,
        direction,
        pathfinder,
        start: Point = None,
        parent: "Candidate" = None,
        verbose: bool = False,
    ):
        c = parent
        self.state = c.state if c is not None else STATE.START
        self.died_on_lichen = c.died_on_lichen if c is not None else False
        self.back_and_forth_quotum = c.back_and_forth_quotum if c is not None else 1
        self.parent = parent
        self.pathfinder = pathfinder

        goal = pathfinder.goal
        unit: Unit = pathfinder.unit
        self.routed_units = (
            c.routed_units
            if c is not None  # and (goal.can_route_own_unit_at_t1 or unit.needs_escape)
            else []
        )
        self.optimal_dirs = []  # empty, will be set by pathfinder

        t = pathfinder.t

        self.direction = direction
        self.queue_len = (
            0
            if c is None
            else c.queue_len
            + (
                direction != parent.direction
                or direction == ACTION.CHARGE
                or direction == ACTION.TRANSFER
                # or direction == ACTION.PICKUP
            )
        )

        point: Point = start if parent is None else move(parent.point, direction)
        self.point = point
        self.is_valid: bool = False
        self.power_consumed = 0 if c is None else c.power_consumed
        self.power_wasted = 0 if c is None else c.power_wasted

        if invalid_on_target(self, goal, verbose):
            return

        # check if it took too long to get to the target
        if (
            not goal.start_charge
            and self.state == STATE.PREPARED
            and not pathfinder.is_optimal
        ):
            if point in pathfinder.targets:
                if (
                    t
                    > point.distance_to(unit.point)
                    + goal.max_waits
                    + goal.max_steps_off_target
                ):
                    if verbose:
                        lprint(f"{t}: target reached too late: {self.path}")
                    return

        if direction == ACTION.PICKUP and self.commands.count(ACTION.PICKUP) > 1:
            if verbose:
                lprint(f"{t}: can only do one pickup: {self.commands}")
            return

        if (
            not goal.plan_even_if_not_complete
            and self.queue_len + goal.post_process_queue_len > 20
        ):
            if verbose:
                lprint(f"{t}: queue too long: {self.queue_len}")
            return

        # TODO: find out if point is same as grandparent point but parent is different
        if t > 3:
            parent_point = parent.point
            if parent_point != point:
                grand_parent_point = parent.parent.point
                if grand_parent_point == point:
                    self.back_and_forth_quotum -= 1
                    if self.back_and_forth_quotum < 0:
                        if verbose:
                            lprint(
                                f"!!!!!!!!!!{t}: back and forth quotum exceeded!: {self.path}"
                            )
                        return

        if is_enemy_factory(self, verbose=verbose):
            return

        units = pathfinder.unit_grid[point]
        self.score = c.score if c is not None else 0

        self.rubble_from_target = (
            defaultdict(int) if c is None else c.rubble_from_target
        )

        self.rubble_from_point = (
            0
            if c is None
            else c.rubble_from_point
            if c.point == point
            else self.rubble_from_target[point]
            if point in self.rubble_from_target
            else 0
        )

        self.power = c.power if c is not None else unit.power
        self.on_goal_action = c.on_goal_action if c is not None else 0

        self.ice: int = c.ice if c is not None else unit.cargo.ice
        self.ore: int = c.ore if c is not None else unit.cargo.ore

        self.ice_before_transfer = c.ice_before_transfer if c is not None else self.ice
        self.ore_before_transfer = c.ore_before_transfer if c is not None else self.ore

        self.waits_left = pathfinder.max_waits if c is None else c.waits_left
        if direction == 0:
            self.waits_left -= 1

        self.pickups = c.pickups if c is not None else defaultdict(int)

        unit_cfg = unit.unit_cfg
        self.is_full = self.ice + self.ore >= unit_cfg.CARGO_SPACE

        # penalty for blocking resource
        if point.ore:
            self.score -= 1
        if point.ice:
            self.score -= 1

        # penalty for being in a factory spot
        if t > 0:
            if point.factory_center:
                self.score -= 1

        #     if point.factory:
        #         self.score -= 1
        #     if point.adjacent_factories and not (point.ore or point.ice):
        #         self.score -= 1 * len(point.adjacent_factories)

        # update power/energy
        move_power = 0
        if direction in MOVES:  # [0, ACTION.DIG, ACTION.TRANSFER]:
            # move cost is based on start cell
            move_power = self.power_move_cost(unit, t)

        self.power -= move_power
        self.power_consumed += move_power

        moved = direction in MOVES
        self.own_combat_strength = unit.combat_strength(moved, self.power)

        if is_out_of_power(self, verbose=verbose):
            return

        if is_attack_risk(self, verbose=verbose):
            return

        if units:
            other_units = set(units) - set([unit]) - set(pathfinder.target_bots)

            if other_units:
                has_enemies = any(u.is_enemy for u in other_units)
                may_die = (
                    goal.may_die_at_destination
                    and (
                        point in pathfinder.destinations or point in pathfinder.targets
                    )
                    and has_enemies
                )

                may_die_lichen = (
                    has_enemies
                    and unit.is_light
                    and point.enemy_lichen
                    and self.pathfinder.gb.steps_left - t <= 3
                    and point.lichen > self.pathfinder.gb.steps_left
                )
                may_die = may_die or may_die_lichen

                if not may_die:
                    my_units = [u for u in other_units if u.is_own and u != unit]
                    # lprint(f"{t} {point}: my units", my_units, file=sys.stderr)

                    if my_units:
                        can_bump_charger = unit.is_heavy and goal.name() == "charger"
                        can_bump_power_hub = (
                            unit.is_heavy and goal.name() == "powerhub"
                        ) or (unit.was_power_hub and (unit.dies or unit.could_die))
                        # allow heavy bumping into light, but not the first turn and not if there is a charger
                        bumps_own_charger = (
                            unit.is_hub
                            and len(
                                [
                                    u
                                    for u in my_units
                                    if u.is_charger
                                    and unit in u.charges_units
                                    and u.is_light
                                ]
                            )
                            > 0
                        )

                        bumps_powerhub = (
                            can_bump_power_hub
                            and len(
                                [
                                    u
                                    for u in my_units
                                    if u.is_power_hub and u.point == u.last_point
                                ]
                            )
                            > 0
                        )

                        routed_by_unit = any(u for u in my_units if u in unit.routed_by)
                        if (
                            any(
                                u
                                for u in my_units
                                if unit <= u or (u.is_charger and not can_bump_charger)
                            )
                            or (
                                t < 2
                                and not (
                                    unit.is_heavy
                                    and (
                                        unit.dies
                                        or unit.could_die
                                        or unit.needs_escape
                                        or unit.died_at_start_turn
                                    )
                                )
                            )
                            or routed_by_unit
                        ):
                            if not bumps_own_charger:
                                if bumps_powerhub and not routed_by_unit:
                                    if verbose:
                                        lprint(
                                            "ROUTING OWN UNIT", my_units, "at", point
                                        )
                                    self.routed_units = my_units
                                elif not routed_by_unit and (
                                    t == 1
                                    and (
                                        (
                                            goal.can_route_own_unit_at_t1
                                            and point in pathfinder.targets
                                            and all(unit >= u for u in my_units)
                                        )
                                        or unit.needs_escape
                                    )
                                    and not any(
                                        [u.died_at_start_turn for u in my_units]
                                    )
                                ):
                                    if verbose:
                                        lprint(
                                            "ROUTING OWN UNIT", my_units, "at", point
                                        )
                                    self.routed_units = my_units
                                else:
                                    if verbose:
                                        lprint(
                                            "collision with own units",
                                            [u.unit_id for u in other_units],
                                            "at",
                                            point,
                                            "t=",
                                            pathfinder.t,
                                            unit.dies,
                                            unit.could_die,
                                            unit.needs_escape,
                                        )
                                    return
                        else:
                            if t < 10:
                                if unit.is_heavy and all(
                                    [
                                        u.is_light
                                        and u.is_replanned
                                        and not u.is_attacking
                                        for u in my_units
                                    ]
                                ):
                                    # if self.routed_units:
                                    #     self.routed_units = self.routed_units + my_units
                                    # else:
                                    self.routed_units = self.routed_units + my_units

                            self.score -= 5  # 2xcomm cost and move cost of a light + some extra overhead cost

                    opponent_units = [u for u in other_units if u.is_enemy]
                    if opponent_units:

                        max_dmg = max(
                            u.combat_strength(u.is_moving, u.power_in_time[t])
                            for u in opponent_units
                        )
                        if self.own_combat_strength > max_dmg:
                            if unit.is_heavy:
                                if all(
                                    u.is_light
                                    and u.is_enemy
                                    and not u.is_targeted
                                    and not u.dies
                                    for u in other_units
                                ):
                                    self.score += 4
                        else:
                            if may_die_lichen:
                                self.died_on_lichen = True
                            else:
                                if verbose:
                                    lprint(
                                        "collision with",
                                        [u.unit_id for u in opponent_units],
                                        "at",
                                        point,
                                        "t=",
                                        pathfinder.t,
                                    )
                                return

        if goal.penalize_move:
            self.score -= move_power
            if direction == 0:
                self.score -= unit.unit_cfg.MOVE_COST / 2

        if direction == ACTION.DIG:
            rubble_left = self.get_rubble_left(t)
            assert (
                IS_KAGGLE
                or point.ice == 1
                or point.ore == 1
                or rubble_left
                or point.lichen
            ), (
                f"{point} {t} {direction}rubble left: {rubble_left}, rubble:"
                f" {point.rubble}, ice: {point.ice}, ore: {point.ore}, {self.rubble_from_point}"
            )

            self.on_goal_action += 1

            if rubble_left > 0:
                rubble_unit_dig = unit_cfg.DIG_RUBBLE_REMOVED
                digged = min(rubble_left, rubble_unit_dig)
                self.update_rubble_from_target(point, digged)

                # use dig amount for scoring to prevent rubble search
                self.score += rubble_unit_dig * GOAL_DIG_SCORE
            else:
                # then remove ICE or ORE
                dig_amount = unit_cfg.DIG_RESOURCE_GAIN
                if point.ice == 1:
                    self.ice += dig_amount
                elif point.ore == 1:
                    self.ore += dig_amount
                if self.ice + self.ore >= unit_cfg.CARGO_SPACE:
                    self.is_full = True
                self.score += dig_amount * GOAL_DIG_SCORE

            dig_cost = pathfinder.dig_cost
            self.power -= dig_cost
            self.power_consumed += dig_cost

            # do again after diggig
            if is_out_of_power(self, verbose=verbose):
                return

        if direction == ACTION.TRANSFER:
            if self.ice > 0:
                self.ice_before_transfer = self.ice
                self.score += self.ice * 10
                self.ice = 0
            elif self.ore > 0:
                self.ore_before_transfer = self.ore
                self.score += self.ore * 10
                self.ore = 0
            else:
                # transport
                self.score += unit.unit_cfg.CARGO_SPACE

        if direction == ACTION.PICKUP:
            self.score += unit.unit_cfg.CARGO_SPACE

        # todo: this should happen for any collide
        if (
            goal.name() == "attack"
            and self.state == STATE.ON_TARGET
            and direction in MOVES
        ):
            assert IS_KAGGLE or opponent_units, "attacking but no opponent units?"
            # self.killed_units = opponent_units
            # rubble_dropped = target_bot.unit_cfg.RUBBLE_AFTER_DESTRUCTION
            # self.update_rubble_from_target(point, -rubble_dropped)

            self.score += 10008

        # CHARGE
        if t > 0:
            # charged on previous turn
            charged = pathfinder.charge_rate[t - 1]
            self.power += charged

            capacity = unit_cfg.BATTERY_CAPACITY
            if self.power > capacity:
                self.power_wasted += self.power - capacity
                self.power = capacity

        if direction == ACTION.CHARGE:
            assert (
                IS_KAGGLE or point.factory
            ), f"{point} {t} {direction} {goal.start_charge}"
            power_at_factory = self.get_power_at_factory()

            if is_power_at_factory(self, power_at_factory, verbose=verbose):
                return

            charge_solar = pathfinder.charge_rate[t]
            to_charge = min(
                unit_cfg.BATTERY_CAPACITY - self.power - charge_solar, power_at_factory
            )

            to_charge = min(to_charge, goal.max_power_pickup)

            if self.power > unit_cfg.BATTERY_CAPACITY * 0.99 * unit_cfg.CHARGE:
                if verbose:
                    lprint(
                        f"{t}: already charged power={self.power} {self.on_goal_action}",
                    )
                return

            assert (
                IS_KAGGLE or to_charge > 0
            ), f"{unit} {goal} Should not charge 0. pwr: {self.power} fac:{power_at_factory} start_charge: {goal.start_charge} state:{self.state.name}, {self.path}, {self.commands}"

            if to_charge <= 0:
                return

            self.power += max(to_charge, 0)
            self.update_pickups(t, to_charge)

            if goal.name() == "chargeatfactory":
                self.on_goal_action += 1

                self.score += 1000

        # communication cost is removed at the start of turn 1,
        # but for the sake of power validation, we add it at end of t=0
        if t == 0:
            comm_cost = unit.action_queue_cost(pathfinder.game_state)
            self.power -= comm_cost
            self.power_consumed += comm_cost

            if is_out_of_power(self, verbose=verbose):
                return

        self.steps_off_target_left = self.get_steps_off_target_left()

        if (
            direction == ACTION.SELF_DESTRUCT
            and point.enemy_lichen
            and point.lichen > self.pathfinder.gb.steps_left
        ):
            self.power -= unit.unit_cfg.SELF_DESTRUCT_COST
            self.power_consumed += unit.unit_cfg.SELF_DESTRUCT_COST
            self.died_on_lichen = True

        goal.update_state(self)

        self.is_valid = self._is_valid(verbose=verbose)

    def get_steps_off_target_left(self):
        parent = self.parent
        if parent is None:
            return self.pathfinder.max_steps_off_target

        steps_left = parent.steps_off_target_left
        if steps_left == 0:
            return 0

        if self.direction in parent.optimal_dirs:
            return steps_left

        if self.direction in MOVES:
            if self.state <= STATE.PREPARED:
                my_distance = self.distance_to_target()
                parent_distance = parent.distance_to_target()
            elif self.state == STATE.TARGET_ACHIEVED:
                my_distance = self.distance_to_destination()
                parent_distance = parent.distance_to_destination()
            else:
                return steps_left

            if my_distance >= parent_distance:
                steps_left -= 1

        return steps_left

    @property
    def path(self):
        if self.parent is None:
            return [self.point]
        return self.parent.path + [self.point]

    @property
    def commands(self):
        if self.parent is None:
            return []
        return self.parent.commands + [self.direction]

    @property
    def powers(self):
        if self.parent is None:
            return [self.power]
        return self.parent.powers + [self.power]

    @property
    def scores(self):
        if self.parent is None:
            return [self.score]
        return self.parent.scores + [self.score]

    @property
    def ices(self):
        if self.parent is None:
            return [self.ice]
        return self.parent.ices + [self.ice]

    @property
    def ores(self):
        if self.parent is None:
            return [self.ore]
        return self.parent.ores + [self.ore]

    def get_power_at_factory(self):
        return min(
            self.point.factory.power_in_time[
                self.pathfinder.t
                - 1 : TTMAX  # -1 since power becomes available at end of turn
            ]
        ) - sum(self.pickups.values())

    def power_move_cost(self, unit: Unit, t: int):
        rubble_at_target = self.get_rubble_left(t)

        return unit.move_cost_rubble(rubble_at_target)

    def update_rubble_from_target(self, point: Point, amount: int):
        # amount is positive if dig, negative if rubble created
        old = self.rubble_from_target
        new = old.copy()
        new_value = new[point] + amount
        new[point] = new_value
        self.rubble_from_target = new
        self.rubble_from_point = new_value

    def update_pickups(self, t: int, amount: int):
        old = self.pickups
        new = old.copy()
        new[t] = amount
        self.pickups = new

    def get_rubble_left(self, t: int):
        point = self.point

        rubble_at_time = point.get_rubble_at_time(t)
        if self.rubble_from_point:
            rubble_left = rubble_at_time - self.rubble_from_point
            # todo: reenable at some point
            # assert IS_KAGGLE or skip_validation or rubble_left >= 0, (
            #     f"{t} {point}: {rubble_left} {rubble_at_time} {self.rubble_from_point} "
            #     f"rubble: {point.rubble}, ice: {point.ice}, ore: {point.ore} {self.path} {self.commands}"
            # )
        else:
            rubble_left = rubble_at_time

        return rubble_left

    def distance_to_destination(self):
        if not hasattr(self, "_distance_to_destination"):
            self._distance_to_destination = self.point.min_distance_to(
                self.pathfinder.destinations
            )
        return self._distance_to_destination

    def distance_to_target(self):
        point = self.point
        pf = self.pathfinder
        if point not in pf.min_distance_to_targets:
            pf.min_distance_to_targets[point] = point.min_distance_to(pf.targets)

        return pf.min_distance_to_targets[point]

    def power_needed_at_end(self):
        """
        Returns the power needed to execute the command, including the power needed to return to the factory.

        power to return + 1 move + comm - charged

        """
        if not hasattr(self, "_power_needed_at_end"):
            pf = self.pathfinder
            goal = pf.goal
            t = pf.t
            unit = goal.unit

            if goal.ignore_power_req_steps_left > pf.steps_left:
                self._power_needed_at_end = 0
                return self._power_needed_at_end

            if (
                (unit.point.factory and t < 3)
                or ((unit.dies or unit.must_move) and t < 5)
            ) and goal.can_be_planned_when_retreating:
                self._power_needed_at_end = 0
                return self._power_needed_at_end

            if not goal.needs_to_return and not goal.reserve_power_to_return:
                self._power_needed_at_end = (
                    0 if goal.ignore_power_req_end else pf.comm_cost + pf.move_cost
                )
                return self._power_needed_at_end

            point = self.point
            if point in pf.min_move_power_to_destinations:
                min_move_power = pf.min_move_power_to_destinations[point]
            else:
                powers = pf.p2p_power[self.point.id][pf.destination_ids]
                min_move_power = min(powers)
                pf.min_move_power_to_destinations[point] = min_move_power

            if point in pf.min_distance_to_destinations:
                min_distance = pf.min_distance_to_destinations[point]
            else:
                min_distance = self.distance_to_destination()
                pf.min_distance_to_destinations[point] = min_distance

            # add 1 for charging the robot when return at factory, needed if it returns on a night
            if min_distance in pf.charge_in_time:
                charge_in_time = pf.charge_in_time[min_distance]
            else:
                charge_in_time = pf.get_charge_power(t, min_distance + 1)
                pf.charge_in_time[min_distance] = charge_in_time

            reserve_end = 0
            if not goal.ignore_power_req_end:
                reserve_end = pf.comm_cost + pf.move_cost

            self._power_needed_at_end = min_move_power - charge_in_time + reserve_end
        return self._power_needed_at_end

    def time_left(self):
        return self.pathfinder.max_it - self.pathfinder.t

    def _is_valid(self, verbose=False):
        pf = self.pathfinder
        goal = pf.goal

        if not goal.plan_even_if_not_complete:
            if (
                goal.needs_to_return
                and self.time_left() < self.distance_to_destination()
            ):
                if verbose:
                    lprint(
                        self.point,
                        pf.t,
                        "cannot make it back in time",
                        self.time_left(),
                        self.distance_to_destination(),
                    )
                return False

        point = self.point

        if self.power < self.power_needed_at_end():
            if self.state > STATE.PREPARED:
                if verbose:
                    lprint(
                        f"{pf.t}: {point} no power to make it back, PWR={int(self.power)}, NEEDED={self.power_needed_at_end()}",
                        f"start_charge={goal.start_charge}",
                    )
                return False

        if self.state == STATE.COMPLETE:
            unit = goal.unit

            # repeat actions starting at the point should still be safe
            if not self.commands:
                if is_attack_risk_no_queue(self, verbose=verbose):
                    return False

            # either enough power, or the spot needs to be free for two turns for recharging
            if self.power < unit.init_power and not goal.may_die_at_destination:
                for i in [1]:
                    other_units = set(
                        [
                            u
                            for u in set(pf.unit_grid_in_time[pf.t + i][point])
                            - set([unit])
                            if u >= unit
                        ]
                    )
                    if len(other_units - set(pf.target_bots)) > 0:
                        if verbose:
                            lprint(
                                f"Other incoming units ({other_units}) prevent "
                                f"recharging at the factory at t={pf.t + i -1}",
                            )
                        return False
            if not goal.on_complete_valid(self, verbose=verbose):
                return False

        return True

    def __repr__(self):
        goal = self.pathfinder.goal
        stats = ""
        if self.ice:
            stats += f"ICE:{self.ice} "
        elif self.ore:
            stats += f"ORE:{self.ore} "
        if self.rubble_from_target:
            stats += f"RUBBLE:{sum(self.rubble_from_target.values())} "

        pretty_path = [str(self.path[0].xy)]
        dig_count = 0
        wait_count = 0
        charge_count = 0
        for i, command in enumerate(self.commands + ["END"]):
            if command == ACTION.DIG:
                dig_count += 1
            elif command == 0:
                wait_count += 1
            elif command == ACTION.CHARGE:
                charge_count += 1
            else:
                if dig_count:
                    pretty_path.append(f"dig x {dig_count}")
                    dig_count = 0
                if wait_count:
                    pretty_path.append(f"wait x {wait_count}")
                    wait_count = 0
                if charge_count:
                    pretty_path.append(f"charge x {charge_count}")
                    charge_count = 0

                if command == "END":
                    pass
                elif command == ACTION.TRANSFER:
                    if goal.resource_id == RESOURCE_MAP["ore"]:
                        pretty_path.append(
                            f"transfer {self.ore_before_transfer} {goal.name()}"
                        )
                    else:
                        pretty_path.append(
                            f"transfer {self.ice_before_transfer} {goal.name()}"
                        )
                else:
                    pretty_path.append(str(self.path[i + 1].xy))

        if dig_count:
            pretty_path.append(f"dig x {dig_count}")
        if wait_count:
            pretty_path.append(f"wait x {wait_count}")

        route = "->".join(pretty_path)

        return (
            f"{self.pathfinder.unit.unit_id}: {goal.name()} SCORE:{self.score} "
            + f"t={self.pathfinder.t} "
            + f"state: {self.state.name} "
            + stats
            + f"PWR:{int(self.power)} "
            + f"len={len(self.commands)} "
            + f"on_goal={self.on_goal_action} "
            + route
        )
