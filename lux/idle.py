import sys

from lux.actions import (
    Move,
    Recharge,
    is_free,
    ChargeAtFactory,
    Retreat,
    PowerHub,
    Charger,
    Rebase,
    Suicide,
)

from lux.unit import Unit
from lux.utils import lprint
from typing import List
from lux.constants import MAX_CHARGE_FACTORY_DISTANCE


def set_power_hubs(self):
    for f in self.factories:
        idle_heavies = [
            u
            for u in f.units
            if u.is_heavy
            and u.point.closest_own_factory_distance <= 1
            and u.can_be_replanned()
        ]

        if len(f.ice_hubs) >= 1 and f.n_heavies == 1:  # or max(f.metal_in_time) < 100:
            # only one heavy, should not use it as power hub
            continue
        if (
            f.power_hub_push
            and len(f.heavy_ore_hubs) == 0
            and f.available_metal() < 150
        ):
            continue

        if idle_heavies:
            for unit in sorted(idle_heavies, key=lambda u: u.power):
                light_charger_points = [
                    u.last_point
                    for u in f.units
                    if u.is_light
                    and u.is_charger
                    and u.last_point == u.point
                    and u.charges_units
                    and not any([h for h in u.charges_units if h.dies or h.could_die])
                ]
                if light_charger_points:
                    # and (
                    #     not f.power_hub_push
                    #     # only release chargers if no power hub positions available
                    #     or len(f.get_power_hub_positions(factory_only=True))
                    #     <= f.n_power_hubs
                    # ):
                    lprint("Charger from set_power_hubs", light_charger_points)
                    if Charger(
                        unit=unit, targets=light_charger_points, start_charge=False
                    ).execute():
                        break
                start_charge = unit.power < 250 and (
                    unit.point.closest_own_factory_distance <= 1
                )

                action = PowerHub(unit=unit, start_charge=start_charge)
                targets = action.get_targets()
                if targets:
                    has_factory_options = any(p.factory for p in targets)
                    if not has_factory_options:
                        has_free_outside_options = any(
                            p
                            for p in targets
                            if not p.closest_factory.closest_tile(p).unit
                            and (
                                self.game_board.attack_risk_in_time[1][p]
                                < unit.combat_strength(
                                    has_moved=False, power_left=unit.power
                                )
                            )
                        )
                        if not has_free_outside_options:
                            for p in targets:
                                if p.factory:
                                    continue

                                other_unit = p.closest_factory.closest_tile(p).unit
                                if other_unit and other_unit.is_power_hub:
                                    # try to move other unit
                                    lprint(f"{unit} BUMPING OTHER CHARGER")
                                    if PowerHub(
                                        unit=other_unit, start_charge=True, targets=[p]
                                    ).execute():
                                        break

                if PowerHub(unit=unit, start_charge=start_charge).execute():
                    break


def handle_idle(self, units: List[Unit], at_risk=False):
    huge_power_factories = [f for f in self.factories if f.available_power() > 3100]
    full_power_hub_factories = [f for f in self.factories if f.full_power_hub()]
    for unit in units:
        if unit.factory in full_power_hub_factories and unit.is_heavy:
            other_factories = sorted(
                [
                    f
                    for f in self.factories
                    if f.distance_to(unit.point) <= 15 and not f.full_power_hub()
                ],
                key=lambda f: f.distance_to(unit.point),
            )
            for f2 in other_factories:
                old_factory = unit.factory
                unit.factory = f2
                lprint(
                    f"IDLE POWER FULL: TRYING REBASE CHARGE AT FACTORY {unit} --> {f2}"
                )
                if ChargeAtFactory(unit=unit).execute():
                    continue
                unit.factory = old_factory

        if (
            unit.is_light
            and unit.factory.power_hub_push
            and unit.factory
            and (unit.dies or unit.could_die or unit.point == unit.factory.center)
        ):
            # blocking center, should move away
            lprint(f"IDLE 0: TRYING MOVE AWAY FROM CENTER {unit}")
            if Move(
                unit,
                start_charge=False,
                targets=unit.factory.edge_tiles(unit, no_chargers=True),
            ).execute():
                continue
        point = unit.point
        other_units = set(self.game_board.unit_grid_in_time[1][point]) - set([unit])

        if (unit.dies and not unit.dies_by_own) or unit.could_die:
            if (
                unit.point.closest_own_factory_distance <= MAX_CHARGE_FACTORY_DISTANCE
                and unit.power < 0.95 * unit.battery_capacity
            ):
                lprint(f"IDLE 0: TRYING CHARGE AT FACTORY {unit}")
                if ChargeAtFactory(unit=unit).execute():
                    continue

            if not unit.point.factory and Retreat(unit).execute():
                continue

        if (
            not (unit.dies or unit.could_die)
            and unit.is_light
            and point.closest_own_factory_distance > 12
            and point.closest_factory.is_enemy
        ):
            # light units not at risk out in the field should just wait for recharge
            unit.remove_from_work_queue()
            continue

        tried_charge = False

        if huge_power_factories:
            closest = min(huge_power_factories, key=lambda f: point.distance_to(f))
            if closest != unit.factory:
                if (
                    unit.point.distance_to(closest) > 20
                    or closest.available_power() > 6000
                ):
                    continue
                old_factory = unit.factory
                unit.factory = closest
                lprint(f"IDLE 1: TRYING REBASE CHARGE AT FACTORY {unit} --> {closest}")
                if ChargeAtFactory(unit=unit).execute():
                    continue
                unit.factory = old_factory

        if (
            not unit.point.factory
            and unit.point.closest_own_factory_distance <= MAX_CHARGE_FACTORY_DISTANCE
            and (
                unit.power < unit.init_power
                or unit.must_move
                or (unit.is_heavy and unit.factory.n_heavies == 1)
            )
        ):
            if not unit.full_power:
                lprint(f"IDLE 1: TRYING CHARGE AT FACTORY {unit}")
                tried_charge = True
                if ChargeAtFactory(unit=unit).execute():
                    continue

        if other_units or unit.must_move or at_risk:
            if move_away(self, unit):
                continue

        # if still nothing, try a factory recharge
        if (
            unit.near_my_factory()
            and not unit.point.factory  # charge should be part of the action
            and not unit.full_power
            and not unit.tried(ChargeAtFactory)
            and (not unit.is_heavy or unit.factory.available_power() > 500)
            and not tried_charge
        ):
            lprint(f"IDLE 2: TRYING CHARGE AT FACTORY {unit}")
            if ChargeAtFactory(unit=unit).execute():
                continue


def rebase_closest_enemy(self, units: List[Unit], verbose=False):
    if self.game_board.step < 700:
        return

    closest_to_enemy_factory = min(
        self.factories, key=lambda f: f.center.closest_enemy_factory_distance
    )

    if len([p for p in closest_to_enemy_factory.points if p.unit]) > 6:
        # crowded at factory
        return

    for unit in units:
        if verbose:
            lprint(
                f"REBASE ENEMY: {unit} avPWR: {unit.available_power()}, dist enemy:{unit.point.closest_enemy_factory_distance}"
            )
        if (
            unit.available_power() > 0.95 * unit.battery_capacity
            and unit.point.closest_enemy_factory_distance > 25
        ):
            # factory is far from closest enemy
            # rebase to the closest to center
            # center_factory = self.game_board.get_point(
            #     (24, 24)
            # ).closest_own_factory

            if closest_to_enemy_factory != unit.factory:

                rebase = Rebase(
                    unit,
                    destinations=closest_to_enemy_factory.edge_tiles(no_chargers=True),
                )
                lprint(
                    f"REBASE ENEMY: {unit} avPWR: {unit.available_power()}, {closest_to_enemy_factory}: {unit.factory.center.closest_enemy_factory_distance} -> {closest_to_enemy_factory.center.closest_enemy_factory_distance}"
                )
                if rebase.execute():
                    continue


def any_nearby_points(unit, distance=3):
    point = unit.point
    points = point.points_within_distance(distance)
    return points


def move_away(self, unit):
    distance = 3
    points = any_nearby_points(unit, distance)
    charge = sum(self.game_board.charge_rate[:3]) * unit.unit_cfg.CHARGE

    p2p_power = self.p2p_power_cost[unit.unit_type]
    power_available = charge + unit.power - unit.comm_cost
    point = unit.point

    destinations = [
        p
        for p in points
        if p != point
        and is_free(self, unit, p, t_max=3)
        and p2p_power[point.id][p.id] <= power_available
    ]

    if not destinations or (
        unit.is_light
        and (unit.dies or unit.could_die)
        and unit.power <= 3 * unit.base_move_cost
    ):
        lprint(f"{unit}: NO POINTS to move to")
        if (unit.dies or unit.could_die) and not unit.dies_by_own:
            unit_grid = self.game_board.unit_grid_in_time[1]
            # also exclude any own units next tick, prefer if not unit next tick
            points = [
                p
                for p in point.adjacent_points()
                if (not p.unit or (p.unit and p.unit.is_enemy))
                and not p.factory
                and not p.own_lichen
                and not any([u.is_own for u in unit_grid[p]])
            ]
            if points:
                best = sorted(
                    points,
                    key=lambda p: (
                        -p.lichen,
                        p.unit is not None,
                        p.unit and p.unit.is_heavy,
                        p.unit and p.unit.power,
                    ),
                )[0]
                lprint(f"{unit}: ATTACKING {best}")
                if Suicide(unit, targets=[best], target_bots=[best.unit]).execute():
                    return True
        if not destinations:
            return False

    factory = point.closest_own_factory
    distance = point.distance_to(factory)
    primary_destinations = [
        p for p in destinations if p.distance_to(factory) < distance
    ]
    secondary_destinations = [
        p for p in destinations if p.distance_to(factory) == distance
    ]
    tertiary_destinations = [
        p for p in destinations if p.distance_to(factory) > distance
    ]

    for dests in [primary_destinations, secondary_destinations, tertiary_destinations]:
        if dests:
            destinations = dests
            lprint(
                "MOVE AWAY: ",
                unit,
                destinations,
                "accept_attack_risk",
                unit.accept_attack_risk,
            )
            start_charge = None
            if unit.point.factory and unit.point.factory.power_hub_push:
                start_charge = unit.power < 5 * unit.comm_cost
            move = Move(unit, targets=destinations, start_charge=start_charge)
            if move.execute():
                return True


def recharge_outside(self, unit: Unit):
    recharge = Recharge(unit)
    if recharge.execute():
        return True
    return False
