from typing import List

import numpy as np
import math
from lux.actions import (
    Ambush,
    AmbushRSS,
    Attack,
    Camp,
    Defend,
    Dump,
    Harass,
    Killnet,
    Shield,
    LichenShield,
    UniChokeLichen,
    UniIce,
    UniLichen,
    TimedAttack,
    UniOre,
    Kamikaze,
    PipelineAttack,
    Guard,
)
from lux.constants import (
    BASE_COMBAT_HEAVY,
    BREAK_EVEN_ICE_TILES,
    IS_KAGGLE,
    MAX_LOW_POWER_DISTANCE,
    MAX_UNIT_PER_FACTORY,
    MAX_ADJACENT_CHASE_INTERCEPT,
    RECHARGE,
    TTMAX,
    RADAR_DISTANCE,
)
from lux.sentry import get_lichen_sentry_points

from lux.point import Point
from lux.router import power_required
from lux.unit import Unit
from lux.utils import flatten, lprint


def action_if_possible(
    agent,
    action,
    unit: Unit,
    point: Point,
    min_t: int,
    max_t: int,
    enemy_strength: int,
    enemy_unit: Unit,
):
    if unit.is_attacking or unit.is_defending:
        return False
    distance = unit.point.distance_to(point)

    offset = 1 if unit.point.factory else 0  # charge
    if (
        min_t - 1 - offset <= distance <= max_t + 1 - offset
    ):  # +1, because if the enemy decides to dig, we can kill it, also, do not wat to send to soon
        on_factory = unit.point.factory
        my_power = (
            unit.power
            + np.sum(unit.game_board.charge_rate[:distance]) * unit.unit_cfg.CHARGE
            - unit.comm_cost
        )
        if on_factory:
            my_power += on_factory.available_power()
            my_power = min(my_power, unit.battery_capacity)
        power_needed = agent.p2p_power_cost[unit.unit_type][unit.point.id][point.id]
        own_strength = unit.combat_strength(
            has_moved=unit.point != point, power_left=my_power - power_needed
        )

        if own_strength > enemy_strength:
            return action(
                unit=unit, targets=[point], target_bots=[enemy_unit]
            ).execute()
    return False


# need general purpose timed attack
def target_if_possible(
    unit: Unit, enemy_unit: Unit, max_distance=100, pipeline=False, verbose=False
):
    if enemy_unit in unit.tried_target_bots and not verbose:
        return False

    if unit < enemy_unit:
        # don't attack heavy with light
        return False
    unit.tried_target_bots.append(enemy_unit)
    allow_charge = (
        unit.power < unit.battery_capacity * 0.975
        and unit.point.closest_own_factory_distance <= 1
    )

    # don't bother if enemy has more power and self lowish on power
    if (
        not allow_charge
        and unit.power < enemy_unit.power
        and unit.power < unit.battery_capacity * 0.4
    ):
        return False

    start_charge, timed_targets = determine_timed_targets(
        unit,
        enemy_unit,
        allow_charge=allow_charge,
        max_distance=max_distance,
        verbose=verbose,
    )
    if verbose:
        lprint(f"{unit}: start_charge: {start_charge} timed_targets: {timed_targets}")
    if not timed_targets:
        return False

    action_type = PipelineAttack if pipeline else TimedAttack

    action = action_type(
        unit=unit,
        timed_targets=timed_targets,
        target_bots=[enemy_unit],
        start_charge=start_charge,
    )
    return action.execute()


def handle_cargo_transport_intercepts(agent, units: List[Unit], max_distance=12):
    # give priority to high water cargo
    for enemy_unit in sorted(
        agent.enemy_transports, key=lambda u: u.cargo.water, reverse=True
    ):
        lprint(
            f"INTERCEPT CARGO {enemy_unit}: water: {enemy_unit.cargo.water} metal: {enemy_unit.cargo.metal}"
        )
        if enemy_unit.dies or enemy_unit.is_targeted:
            continue

        candidates = sorted(units, key=lambda u: u.point.distance_to(enemy_unit.point))
        for unit in candidates:
            if (
                unit.is_attacking
                or unit.is_defending
                or unit.has_cargo
                or unit.is_charger
                or unit.unit_type != enemy_unit.unit_type
            ):
                continue

            distance = unit.point.distance_to(enemy_unit.point)
            if distance > max_distance:
                break  # sorted

            lprint(f"TRY INTERCEPT CARGO {unit} against {enemy_unit}")
            if target_if_possible(unit, enemy_unit, max_distance=max_distance):
                break


def handle_adj_attack_opportunities(units: List[Unit], verbose=False):
    for unit in sorted(units, key=lambda u: u.is_attacking, reverse=True):
        if verbose:
            lprint(f"ADJ ATTACK {unit}")
        if unit.kills_next:
            unit.remove_from_work_queue()
            if verbose:
                lprint(f"KILLS NEXT {unit}")
            continue

        if unit.must_retreat or unit.is_charger:
            if verbose:
                lprint(f"MUST RETREAT {unit}")
            continue

        if intercept(unit):
            if verbose:
                lprint(f"INTERCEPT {unit}")
            continue

        if not unit.skip_adjacent_attack and handle_adjacent_enemies(
            unit, verbose=verbose
        ):
            if verbose:
                lprint(f"HANDLE ADJ ENEMIES {unit}")
            continue


def ambush_rss(agent, units: List[Unit], no_actions=False):
    workers = units.copy()

    # prevent enemy ambush
    for f in agent.factories:
        if f.is_ore_risk() and f.ore_points:
            rp = f.ore_points[0]
            if rp.closest_factory.is_own:
                for p in [rp] + rp.adjacent_points():
                    enemy_campers = [
                        u
                        for u in p.visited_by
                        if u.is_enemy and not u.is_targeted and u.last_point == p
                    ]

                    if not enemy_campers:
                        continue

                    lprint(f"CAMPERS detected {f} {p} {enemy_campers}")

                    enemy = enemy_campers[0]
                    for unit in sorted(f.units, key=lambda u: u.point.distance_to(p)):
                        if unit.is_heavy != enemy.is_heavy:
                            continue
                        if not unit.is_shield and not unit.can_be_replanned():
                            continue
                        target_if_possible(unit, enemy, max_distance=12)

    for target in agent.pipeline_targets:
        if (
            target.dies
            or target.is_targeted
            or any(
                [
                    p.unit.dies
                    or p.unit.is_targeted
                    or [u.is_own for u in p.visited_by]
                    for p in target.last_point.adjacent_points()
                    if p.unit and p.unit.is_enemy
                ]
            )
        ):
            continue

        my_units = sorted(workers, key=lambda u: u.point.distance_to(target.last_point))

        for unit in my_units:
            if unit.is_heavy:
                continue
            # plenty of power, or close to target
            if unit.enough_power_or_close(target.last_point):
                lprint(f"TRY HIT PIPELINE {unit} against {target}")
                if target_if_possible(unit, target, max_distance=25):
                    lprint(f"TARGETING PIPELINE {unit} --> {target}")
                    workers.remove(unit)
                    continue

    workers = sorted(workers, key=lambda u: u.point.closest_enemy_factory_distance)
    if agent.n_opp_lights > 5:
        for unit in [u for u in workers if (no_actions or u.factory.n_lights > 4)]:
            if unit.enough_power_or_close():
                if AmbushRSS(unit=unit).execute():
                    continue


def handle_factory_defense(agent, units: List[Unit], max_distance=10):
    units = units.copy()
    to_defend_units = agent.opponent_units.copy()
    for enemy_unit in sorted(
        to_defend_units, key=lambda u: u.point.closest_own_factory_distance
    ):
        point = enemy_unit.point
        if enemy_unit.is_targeted:
            continue

        if (
            enemy_unit.is_heavy
            and enemy_unit.point.closest_own_factory.n_all_lichen_tiles == 0
        ):
            continue

        if enemy_unit.direct_path_to_factory:
            continue

        if point.closest_own_factory_distance > max_distance:
            break  # sorted so can stop

        if enemy_unit.is_light and len(point.closest_own_factory.lichen_points) == 0:
            continue

        if (
            point.closest_factory.is_enemy
            and point.closest_own_factory_distance > 5
            and not enemy_unit.digs_lichen
        ):
            continue

        if point.closest_enemy_factory_distance <= 1:
            continue

        if (
            point.closest_enemy_factory_distance <= 2
            and enemy_unit.is_digging
            and not enemy_unit.digs_lichen
        ):
            continue

        if enemy_unit.dies:
            continue

        # check if unit is already being chased
        next_units_at_position = agent.game_board.unit_grid_in_time[1][enemy_unit.point]
        if len(next_units_at_position) > 0 and any(
            [u.is_own for u in next_units_at_position]
        ):
            continue

        # check if enemy moves away from us
        if not enemy_unit.point.own_lichen:
            prev_distance = point.closest_own_factory_distance
            next_unit = False
            for i, p in enumerate(enemy_unit.path):
                distance = p.closest_own_factory_distance
                if distance < prev_distance:
                    break
                prev_distance = distance
                if distance > max_distance:
                    next_unit = True
                    break
            if next_unit:
                continue

        lprint(f"DEFEND FACTORY against {enemy_unit}")
        workers = sorted(
            [
                u
                for u in units
                if u.is_heavy == enemy_unit.is_heavy
                and u.factory == enemy_unit.point.closest_own_factory
                and (u.cargo.metal == 0 and u.cargo.water == 0)
                and (u.is_heavy or u.can_be_replanned())
                and (
                    not (u.is_hub and u.digs_ice)
                    or u.factory.available_water()
                    > 100 + 10 * u.factory.n_connected_tiles
                )
                and (
                    u.factory.available_water() > 40
                    or len(u.factory.nearby_heavies()) > 1
                )
            ],
            key=lambda u: u.point.distance_to(point),
        )

        heavy_workers = sorted(
            [u for u in workers if u.is_heavy],
            key=lambda u: u.available_power(u.point.closest_own_factory_distance <= 1),
        )
        best_heavy = heavy_workers[-1] if len(heavy_workers) > 0 else None

        heavy_backup = None
        for unit in workers:
            distance_1 = unit.point.distance_to(point)

            if distance_1 > len(enemy_unit.path) - 1:
                target_point = enemy_unit.path[-1]
            else:
                target_point = enemy_unit.path[distance_1]

            distance_2 = unit.point.distance_to(target_point)
            if distance_2 > max_distance:
                break

            lprint(f"TRY DEFEND FACTORY {unit} against {enemy_unit}")
            if (
                unit.point.distance_to(enemy_unit.point) == 1
                and unit.go_home_power_left()
                > 2 * unit.move_cost_rubble(enemy_unit.point.rubble) + unit.comm_cost
            ):
                # unit next to enemy
                lprint(f">>>>>>>>TRY ATTACK {unit} against {enemy_unit}")
                if Attack(
                    unit=unit, targets=[enemy_unit.point], target_bots=[enemy_unit]
                ).execute():
                    units.remove(unit)
                    break
                lprint(f">>>>>>>>TRY ATTACK {unit} against {enemy_unit} FAILED")

            if unit.total_cargo > 50:
                if heavy_backup is None:
                    heavy_backup = unit
            else:
                if target_if_possible(unit, enemy_unit, max_distance=max_distance):
                    units.remove(unit)
                    break

            if enemy_unit.is_heavy and unit.total_cargo <= 50:
                # lprint(
                #     f"HEAVY UNIT DEFENSE {unit} against {enemy_unit}, best_heavy {best_heavy}"
                # )
                power_req = unit.required_from_to(
                    unit.point, target_point
                ) + unit.required_from_to(
                    target_point, target_point.closest_own_factory.center
                )

                # need to have a good power surplus and be able to return home
                start_charge = False
                if unit.power - power_req < enemy_unit.power:
                    if unit.point.closest_own_factory_distance <= 1:
                        if (
                            unit.power + unit.factory.available_power() - power_req
                            < enemy_unit.power
                        ):
                            continue
                        else:
                            start_charge = (
                                unit.power < 0.95 * unit.battery_capacity
                                and unit.factory.available_power() > 10
                            )
                    else:
                        # if unit != best_heavy:
                        continue

                distance = distance_1 if distance_2 < distance_1 else distance_2
                target_point = enemy_unit.next_point
                lprint(f"{unit} TRY ATTACK {enemy_unit}")
                if unit.point == target_point and unit <= enemy_unit:
                    continue

                if Attack(
                    unit=unit,
                    targets=[target_point],
                    target_bots=[enemy_unit],
                    start_charge=start_charge,
                ).execute():
                    lprint(
                        f">>>>>> FACTORY DEFENSIVE ACTION {unit}-->{enemy_unit}: {target_point}, ETA: {distance}"
                    )
                    units.remove(unit)
                    break

        if not enemy_unit.is_targeted:
            if best_heavy and best_heavy.total_cargo > 50:
                if Dump(unit=best_heavy).execute():
                    units.remove(best_heavy)
            else:
                if heavy_backup is not None:
                    if Dump(unit=heavy_backup).execute():
                        units.remove(heavy_backup)


def ambush_tile_or_adjacent(unit, tile, source):
    lprint(f"{unit} AMBUSH FROM source:{source} to factory--tile:{tile}")

    adjacent_targets = [ap for ap in tile.adjacent_points() if ap.factory is None]

    if unit.point in adjacent_targets:
        own_units_at_position = [
            u
            for u in unit.game_board.unit_grid_in_time[1][unit.point]
            if u != unit and u.is_own
        ]

        if (
            unit.game_board.attack_risk_in_time[1][unit.point]
            > unit.combat_strength(False, unit.power)
        ) or (len(own_units_at_position) > 0):
            adjacent_targets = [p for p in adjacent_targets if p != unit.point]

    # see if there are targets which blockade more than one ore/ice
    priority_targets = {
        ap: len(
            [
                app
                for app in ap.adjacent_points() + [ap]
                if app.ice
                or app.ore
                and not (app.unit and app.unit.is_own)
                and (not ap.closest_factory.is_own or (ap.unit and ap.unit.is_enemy))
            ]
        )
        for ap in adjacent_targets
    }
    priorities = {
        value: [key for key in priority_targets if priority_targets[key] == value]
        for value in set(priority_targets.values())
    }

    enemy_digger = [u for u in tile.dug_by if u.is_enemy]
    if enemy_digger:
        enemy_digger = enemy_digger[0]
    min_power = (
        max(unit.init_power, enemy_digger.power) if enemy_digger else unit.init_power
    )

    start_charge = True if unit.power < min_power else None

    for priority in sorted(priorities, reverse=True):
        ambush = Ambush(unit, targets=priorities[priority], start_charge=start_charge)
        if ambush.execute():
            return True


def handle_factory_attack(agent, verbose=False):
    lprint("FACTORY ATTACK == handle_factory_attack")
    for f in agent.opponent_factories:
        if verbose:
            lprint(f"FACTORY ATTACK == {f} handle_factory_attack")
        if len(f.ice_points) == 0 or f.ice_points[0].distance_to(f) > 2:
            if verbose:
                lprint(f"FACTORY ATTACK == {f} no ice")
            continue

        if not f.ice_points:
            continue

        min_distance = min(f.ice_points, key=lambda ip: ip.distance_to(f)).distance_to(
            f
        )
        ice_targets = [
            p
            for p in f.ice_points
            if p.distance_to(f) == min_distance
            # and p.unit
            # and p.unit.is_enemy
            # and p.unit.is_heavy
        ]
        # if not ice_targets:
        #     if verbose:
        #         lprint(f"FACTORY ATTACK == {f} no ice targets")
        #     ice_targets = f.ice_points

        # tiles = ice_targets

        low_on_water = f.available_water() < 50
        heavy_mines_ice = [
            ip
            for ip in ice_targets
            if ip.closest_factory.is_enemy
            and len(
                [
                    u
                    for u in ip.dug_by
                    if u.is_heavy and u.is_enemy and u.path.index(ip) < 4
                ]
            )
            > 0
        ]
        if verbose:
            lprint(
                f"HEAVY MINES ICE tile {ice_targets}: {heavy_mines_ice}, low_on_water: {low_on_water}"
            )

        adjacent_p = [
            ap for ap in ice_targets[0].adjacent_points() if ap.factory is None
        ] + [ice_targets[0]]
        units_inbound = flatten([p.visited_by for p in adjacent_p])
        light_units_inbound = [u for u in units_inbound if u.is_own and u.is_light]
        # heavy_units_inbound = [u for u in units_inbound if u.is_own and u.is_heavy]
        has_light_inbound = len(light_units_inbound) > 0
        # has_heavy_inbound = len(heavy_units_inbound) > 0
        enemy_n_heavies = len(
            [u for u in f.units if u.is_heavy and u.point.distance_to(f) < 12]
        )
        close_heavies = [u for u in agent.heavies if u.point.distance_to(f) <= 12]
        if verbose:
            lprint(
                f"{f}: has_light_inbound: {has_light_inbound}, enemy_n_heaviess: {enemy_n_heavies}, closest_heavies: {len(close_heavies)}",
                (enemy_n_heavies <= 1 or not heavy_mines_ice),
                enemy_n_heavies > 3,
                f.available_water()
                < (275 if f.center.closest_own_factory.n_heavies > 1 else 150),
            )

        if low_on_water or (
            (enemy_n_heavies <= 1 or not heavy_mines_ice)
            and (
                (len(close_heavies) > 3 and f.available_water() < 250)
                or f.available_water()
                < (275 if f.center.closest_own_factory.n_heavies > 1 else 150)
            )
        ):
            if verbose:
                lprint(f"{f} LOW ON WATER OR ENEMY HAS FEW HEAVIES")
            for f2 in agent.factories:
                # lprint(
                #     f"FACTORY ATTACK == {f2}--> {f} handle_factory_attack",
                #     f2.power_hub_push,
                #     f.available_water(),
                #     f2.n_power_hubs < len(f2.get_power_hub_positions()),
                #     f.available_water() > 150,
                # )
                if (
                    f2.power_hub_push
                    and not f2.full_power_hub()
                    and (
                        (
                            f2.n_power_hubs < f2.n_power_hub_positions
                            or f.available_water() > 150
                        )
                    )
                ):
                    # lprint("FACTORY ATTACK == power_hub_push")
                    if (
                        f.available_water() > 50
                        or f.center.closest_own_factory != f2
                        or any(
                            f3
                            for f3 in agent.opponent_factories
                            if f3 != f
                            and f3.distance_to(f2) < 12
                            and f3.available_water() > 75
                        )
                    ):
                        if verbose:
                            lprint(
                                f"{f} has power hub push and target not really out of water, skipping"
                            )
                        continue

                if f2.distance_to(f) > 12:
                    continue
                n_heavies = len(
                    [
                        u
                        for u in f2.heavies
                        if u.last_point.distance_to(ice_targets[0]) < 15
                    ]
                )
                n_lights = f2.n_lights
                if n_heavies == 0:
                    continue

                n_ice_hubs = len(f2.ice_hubs)

                if n_heavies <= 1 and not low_on_water:
                    if verbose:
                        lprint(
                            f"{f} not low on water and not many heavies on ownfactory {n_heavies}, skipping"
                        )
                    continue
                if n_heavies == 1 and f2.available_water() < 75:
                    if verbose:
                        lprint(
                            f"{f} not enough water on own factory {f2.available_water()}, skipping"
                        )
                    continue

                # start with dug by tile
                for tile in sorted(
                    ice_targets, key=lambda p: len(p.dug_by), reverse=True
                ):
                    attack_candidates = sorted(
                        [u for u in f2.heavies if u.can_be_replanned()],
                        key=lambda u: u.point.distance_to(tile),
                    )
                    highest_power_bot = max(
                        f2.heavies, key=lambda u: u.available_power()
                    )
                    attack_candidates.append(highest_power_bot)

                    for unit in attack_candidates:
                        if verbose:
                            lprint("handle_factory_attack", unit, tile)

                        power_cost = power_required(unit, unit.point, tile)
                        # detect expensive attack
                        if (
                            tile != unit.point
                            and power_cost
                            / unit.point.distance_to(tile)
                            / unit.base_move_cost
                            > 2
                            and f.available_water() > 50
                        ):
                            # do not do an expensive attack if we can still expand on this factory
                            lprint("Expensive attack, skipping?")
                            if n_ice_hubs * 2 > n_heavies:
                                continue
                            if f2.n_power_hubs < f2.n_power_hub_positions:
                                continue

                        if not unit.can_be_replanned():
                            if unit.is_attacking and not unit.try_replan:
                                continue

                            if unit.last_point in adjacent_p:
                                continue

                        if (
                            unit.available_power() - 2 * power_cost
                            < 0.4 * unit.battery_capacity
                        ) and unit.point.distance_to(f) > 6:
                            lprint(
                                f"{unit}: Not enough power to attack {f}, skipping PWR:{unit.available_power() - 2 * power_cost} < {0.4 * unit.battery_capacity}"
                            )
                            continue

                        if heavy_mines_ice or unit.can_be_replanned():
                            if unit.total_cargo > 0:
                                is_dumping = (
                                    unit.has_queue
                                    and unit.last_point.factory
                                    and any(
                                        [
                                            a[0] == 1 and a[1] == 0
                                            for a in unit.action_queue
                                        ]
                                    )
                                )
                                if is_dumping:
                                    continue
                                else:
                                    lprint("Dumping as preparation for attack")
                                    if Dump(unit).execute():
                                        continue

                            covered = unit.is_target_covered(tile)
                            lprint(
                                ">>>>>>>>>> FACTORY ATTACK CANDIDATE",
                                f,
                                "BY",
                                unit,
                                "tile:",
                                tile,
                                "covered",
                                covered,
                                "distance",
                                unit.point.distance_to(tile),
                                "unit.can_be_replanned()",
                                unit.can_be_replanned(),
                            )
                            if tile.unit and tile.unit.is_enemy and not tile.unit.dies:
                                start_charge = (
                                    True if unit.power < unit.init_power else None
                                )
                                lprint("TRY ATTACK")
                                if Attack(
                                    unit,
                                    targets=[tile],
                                    target_bots=[tile.unit],
                                    start_charge=start_charge,
                                ).execute():
                                    continue

                                if covered or ambush_tile_or_adjacent(
                                    unit, tile, "handle_factory_attack 2"
                                ):
                                    continue

                            if (
                                unit.point.distance_to(tile) <= 3
                                and unit.can_be_replanned()
                            ):
                                if covered or ambush_tile_or_adjacent(
                                    unit, tile, "handle_factory_attack 3"
                                ):
                                    continue

                    if (
                        not heavy_mines_ice
                        and (n_heavies >= 1 or n_lights > 5)
                        and agent.n_opp_lights > 3
                    ):
                        # blockade tile with light
                        available_lights = [
                            u
                            for u in f2.lights
                            if u.can_be_replanned(close_to_enemy_factory=True)
                            and not u.is_attacking
                            and not u.is_defending
                            # either no inbounds yet, or u already inbound
                            and (not has_light_inbound or u in light_units_inbound)
                        ]
                        # if tile.unit and [u for u in tile.dug_by if u.is_enemy and u.is_light]:
                        if verbose:
                            lprint(f"{f}: available_lights: {available_lights}")
                        is_targeted = False
                        for unit in sorted(
                            available_lights, key=lambda u: u.point.distance_to(tile)
                        ):
                            if (
                                unit.near_enemy_factory()
                                and unit.point.closest_enemy_factory != f
                                and unit.point.distance_to(f) > 9
                            ):
                                continue

                            if tile.unit and tile.unit.is_enemy and tile.unit.is_light:
                                lprint(f"{unit} factory attack target {tile.unit}")
                                if Attack(
                                    unit, targets=[tile], target_bots=[tile.unit]
                                ).execute():
                                    is_targeted = True
                                    break

                            if ambush_tile_or_adjacent(
                                unit, tile, "handle_factory_attack"
                            ):
                                is_targeted = True
                                break
                        # if is_targeted:
                        #     continue

    lprint("END FACTORY ATTACK == handle_factory_attack")


def handle_rss_steal(agent, units, max_distance=12):
    risk_grid = agent.game_board.attack_risk_in_time
    for point, (t, enemy_unit, power) in agent.rss_steal.items():
        factory = point.closest_own_factory
        enemy_strength = enemy_unit.combat_strength(has_moved=True, power_left=power)

        min_t = t
        point_risk_grid = risk_grid[point]
        max_t = t + enemy_unit.digs[point]
        targeted = False
        action = UniOre if point.ore else UniIce

        can_target_tile = True
        if enemy_unit.is_light:
            # don't even bother trying this if a heavy is next to it
            max_attack_risk = max(
                [point_risk_grid[tt] for tt in range(min_t, min_t + 3)]
            )

            if max_attack_risk >= BASE_COMBAT_HEAVY:
                can_target_tile = False

        if can_target_tile:
            # only dig if rubble cleared
            if point.get_rubble_at_time(t + 1) == 0:
                for unit in sorted(units, key=lambda u: u.point.distance_to(point)):
                    if enemy_unit.unit_type != unit.unit_type:
                        continue

                    distance = unit.point.distance_to(point)
                    if distance > max_distance:
                        break  # sorted by distance, so we can stop here

                    if (
                        distance + 2 >= min_t
                    ):  # don't come too soon (include charge and pressure)

                        if action_if_possible(
                            agent,
                            action,
                            unit,
                            point,
                            min_t,
                            max_t,
                            enemy_strength,
                            enemy_unit,
                        ):
                            lprint("SCARE MINE ENEMY", unit, point, t, enemy_unit)
                            targeted = True
                            break
            if targeted:
                continue

            for unit in sorted(factory.units, key=lambda u: u.point.distance_to(point)):
                if (
                    unit.is_heavy == enemy_unit.is_heavy
                    and unit.can_be_replanned()
                    and not unit.has_cargo
                ):
                    if enemy_unit.is_heavy and enemy_unit.point.distance_to(
                        point
                    ) < unit.point.distance_to(point):
                        continue

                    if action_if_possible(
                        agent,
                        Defend,
                        unit,
                        point,
                        min_t,
                        max_t,
                        enemy_strength,
                        enemy_unit,
                    ):
                        break


def handle_units_close_to_enemy(agent, units: List[Unit]):

    steps_left = agent.game_board.steps_left

    close_units = []
    for unit in units:
        if not unit.can_be_replanned(close_to_enemy_factory=True):
            continue

        if steps_left > 10 and (unit.must_retreat or unit.has_cargo):
            continue

        if (
            unit.near_enemy_factory()
            or unit.ends_recharge(RECHARGE.CLOSE_FACTORY)
            or (
                unit.was_ambushing
                and unit.can_be_replanned(close_to_enemy_factory=True)
                and (
                    unit.point.closest_own_factory_distance > 6
                    or unit.factory.available_water() > 50
                    or (
                        unit.point.closest_enemy_factory.available_water()
                        < unit.factory.available_water()
                    )
                )
            )
        ):
            close_units.append(unit)

    close_units = sorted(close_units, key=lambda u: u.power, reverse=True)

    for unit in close_units:
        # when being near an enemy factory there are a few options to do
        # 1. attack lichen
        # 2. pressure enemy
        # 3. ambush enemy
        # 4. camp lichen
        #   a. check the amount of lichen tiles, if high, go for the lichen
        #   b. if low and water is low, or is heavy and high power do an ambush
        #   c. heavies should not move too much and just recharge if neither of the above

        enemy_factory = unit.point.closest_enemy_factory

        nearby_enemy_factories = sorted(
            [f for f in agent.opponent_factories if f.distance_to(unit.point) <= 6],
            key=lambda f: f.distance_to(unit.point),
        )
        for enemy_factory in nearby_enemy_factories:
            if not unit.can_be_replanned(close_to_enemy_factory=True):
                continue
            available_water = enemy_factory.available_water()
            water_lowish = available_water < 200

            # my_factory =
            # if unit.factoryavailable_water

            priority_ambush = (
                unit.is_heavy and unit.power > unit.init_power and water_lowish
            )
            if priority_ambush:
                closest_ice = (
                    enemy_factory.ice_points[0] if enemy_factory.ice_points else None
                )

                if closest_ice:
                    d = closest_ice.distance_to(enemy_factory)
                    targets = [
                        p
                        for p in enemy_factory.ice_points
                        if p.distance_to(enemy_factory) <= d
                    ]
                    if (
                        unit.point == unit.last_point
                        and unit.point in targets
                        and not unit.could_die
                        and not unit.dies
                    ):
                        unit.remove_from_work_queue()
                        continue
                    tile_targeted = False
                    for tile in targets:
                        if ambush_tile_or_adjacent(
                            unit, tile, "handle_units_close_to_enemy"
                        ):
                            tile_targeted = True
                            break
                    if tile_targeted:
                        continue

            lprint("NEAR ENEMY FACTORY", unit)

            # beware, this does not plan it yet!
            ambush_action = Ambush(unit=unit)
            ambush_action.targets = ambush_action.get_filtered_ambush_targets(
                enemy_factory
            )

            if unit.is_light:
                if unit.enough_power_or_close():
                    if AmbushRSS(unit=unit).execute():
                        continue

            dig_lichen = (
                unit.is_light
                or steps_left < 100
                or not ambush_action.targets
                # > (BREAK_EVEN_ICE_TILES if available_water < 200 else 5)
            ) and enemy_factory.consider_lichen_assault()

            if dig_lichen:
                action = UniLichen(unit=unit)
                action.targets = action.get_targets(unit.point.closest_enemy_factory)
                lprint(
                    "trying unilichen from handle_units_close_to_enemy, targets:",
                    action.targets,
                )
                if action.execute():
                    continue

            if unit.is_heavy:
                heavy_mines_ice_points = [
                    ip
                    for ip in enemy_factory.ice_points
                    if ip.closest_factory.is_enemy
                    and len(
                        [
                            u
                            for u in ip.dug_by
                            if u.is_heavy
                            and u.is_enemy
                            and u.path.index(ip) <= unit.point.distance_to(ip)
                            and not u.dies
                        ]
                    )
                    > 0
                ]
                if heavy_mines_ice_points:
                    if Attack(
                        unit,
                        targets=heavy_mines_ice_points,
                        target_bots=[ip.dug_by[0] for ip in heavy_mines_ice_points],
                    ).execute():
                        continue

            if (
                ambush_action.targets == [unit.point]
                and unit.last_point == unit.point
                and not unit.could_die
                and not unit.dies
            ):
                unit.remove_from_work_queue()
                continue

            # low on power, let's not do crazy stuff
            if unit.power < unit.battery_capacity / 6:
                continue

            if unit.is_light and Harass(unit=unit).execute():
                continue

            lprint(f"{unit}: AMBUSH FROM handle_units_close_to_enemy")
            if not priority_ambush and unit.is_heavy and ambush_action.execute():
                continue

            # added ambush_action.targets because it makes no sense camping if there still were ambush targets
            if not ambush_action.targets and unit.power > unit.init_power * 0.5:
                if camp_lichen(agent, unit):
                    continue


def handle_low_prio_combat(agent, units: List[Unit]):

    dominance_targets = []
    if (
        agent.n_lights > len(agent.factories) * 10
        and agent.n_lights > agent.n_opp_lights * 3
    ):
        dominance_targets = [
            u
            for u in agent.opponent_lights
            if u.point.factory is None and u.last_point.factory is None
        ]

    if dominance_targets:
        candidates = [
            u
            for u in units
            if u.is_light
            and not u.is_attacking
            and not u.has_cargo
            and u.power > u.battery_capacity * 0.5
        ]

        for target in dominance_targets:
            candidates = sorted(
                candidates, key=lambda u: u.point.distance_to(target.point)
            )
            for unit in candidates:
                if unit.point.distance_to(target.point) > 6:
                    break  # sorted, so no need to check the rest

                if unit.power > target.power:
                    lprint(f"trying dominance attack {unit} -> {target}")
                    if target_if_possible(unit, target, max_distance=8):
                        candidates.remove(unit)
                        units.remove(unit)
                        break

    for unit in units:
        if unit.factory.power_hub_push and not unit.factory.full_power_hub():
            return

        if unit.has_cargo or unit.is_attacking:
            continue

        if unit.is_light and unit.power < unit.battery_capacity / 6:
            continue

        if attack_distance(unit):
            continue

        enough_water = unit.factory.cargo.water > min(75, agent.game_board.steps_left)
        if not enough_water:
            continue

        lprint(f"{unit}: AMBUSH FROM handle_low_prio_combat")
        if set_ambush(agent, unit):
            continue


def handle_chase_intercept(
    agent, units: List[Unit], max_distance=MAX_ADJACENT_CHASE_INTERCEPT
):
    for enemy_unit in agent.adjacent_targets:
        point = enemy_unit.point
        for unit in sorted(
            [
                u
                for u in units
                if u.is_heavy == enemy_unit.is_heavy and not u.is_charger
            ],
            key=lambda u: u.point.distance_to(point),
        ):
            if unit.attacks_bot == enemy_unit:
                continue
            distance = unit.point.distance_to(point)

            if distance > max_distance:
                break  # break because sorted

            lprint(
                f"TRY ADJACENT_CHASE_INTERCEPT INTERCEPT {unit} against {enemy_unit}"
            )
            if target_if_possible(unit, enemy_unit, max_distance=max_distance + 1):
                break


def handle_low_power_intercept(
    agent, units: List[Unit], max_distance=MAX_LOW_POWER_DISTANCE
):
    for enemy_unit, (
        t,
        point,
        power,
        enemy_req_power,
    ) in agent.potential_targets.items():
        if enemy_unit.dies:
            continue
        if enemy_unit.is_targeted:
            continue
        distance_to_factory = point.closest_enemy_factory_distance
        if distance_to_factory <= 1 and t > 1:
            continue

        distance_to_factory_now = enemy_unit.point.closest_enemy_factory_distance
        if distance_to_factory_now < 3 and distance_to_factory > 3:
            # wait a bit
            continue

        # wait for a later opportunity
        if (point.closest_own_factory_distance + 1) * 2 < t and (
            enemy_unit.point.closest_own_factory_distance + 1
        ) * 2 < t:
            continue

        lprint(
            f"{agent.game_board.step}: !!!!!!!!!!!LOW POWER INTERCEPT SEARCH t={t}",
            enemy_unit,
            point,
            power,
            enemy_unit.is_targeted,
        )

        for unit in sorted(
            [u for u in units if u.is_heavy == enemy_unit.is_heavy],
            key=lambda u: u.point.distance_to(point),
        ):
            distance = unit.point.distance_to(point)
            if unit.is_charger and (distance > 2 or t > 2):
                continue

            if (
                distance_to_factory <= 3
                and distance >= 3 * distance_to_factory
                and unit.is_digging
            ):
                break  # break because sorted

            if distance - t > max_distance:
                break  # break because sorted

            if (
                unit.is_heavy
                and distance > 6
                and unit.factory.available_water() < 40
                and (
                    len(
                        [
                            u
                            for u in unit.factory.heavies
                            if u.point.distance_to(unit.factory) < 6
                        ]
                    )
                    == 1
                )
                and enemy_unit.point.closest_enemy_factory_distance < 3
                and enemy_unit.factory.n_heavies > 2
            ):
                continue

            lprint(
                f"TRY LOW POWER INTERCEPT {unit} against {enemy_unit} with power {power} at t={t}"
            )
            if target_if_possible(unit, enemy_unit, max_distance=max_distance):
                break


def handle_lichen_attack(agent, units, min_kills=1, max_kills=10000):
    cluster_to_kills_points = agent.choke_clusters_enemy
    steps_left = agent.game_board.steps_left
    # invert keys and values, allow multiple keys for same value
    kills_to_clusters = {}
    for cluster, (kills, points) in cluster_to_kills_points.items():
        kills_to_clusters.setdefault(kills, []).append(cluster)

    # sort by kills, descending
    for kills, clusters in sorted(kills_to_clusters.items(), reverse=True):
        if kills < min_kills:
            break
        if kills > max_kills:
            continue
        for cluster in clusters:
            points = cluster_to_kills_points[cluster][1]
            n_points = len(points)

            # single or double choke point should get higher amount of bots targeting the area
            reserve_points = []
            if n_points <= 2:
                reserve_points = [
                    p
                    for p in points[0].adjacent_points()
                    if p.lichen and p not in points
                ]

            n_digs_light = min(math.ceil(p.lichen // 10) for p in points)

            # change logic, only send light if close or already sending a heavy
            # prefer_heavy = kills > 10
            heavy_sent = False
            max_light_distance = 7
            # 0 is the closest to enemy factory
            n_squad = 0
            unit: Unit
            for unit in sorted(
                units,
                key=lambda u: (
                    u.is_light,
                    u.point.distance_to(points[0]),
                ),
            ):
                n_digs = n_digs_light if unit.is_light else 1

                min_distance = unit.point.min_distance_to(tuple(points))
                if steps_left < 50:
                    if min_distance + n_digs >= steps_left:
                        continue

                if min_distance + n_digs > TTMAX:
                    continue

                if (
                    unit.can_be_replanned(close_to_enemy_factory=True)
                    and not unit.must_retreat
                    and not unit.has_cargo
                    and not (unit.is_heavy and unit.factory.n_heavies == 1)
                    and (
                        unit.available_power() > 0.75 * unit.battery_capacity
                        or unit.point.distance_to(points[0]) * 2 < kills
                    )
                    and (
                        len(unit.factory.units) > 7
                        or steps_left < 100
                        or unit.near_enemy_factory()
                        # or kills > 15
                        or unit.point.distance_to(points[0]) <= max_light_distance
                        or heavy_sent
                    )
                    and unit.consider_lichen_attack()
                    and (
                        power_required(unit, points[0], unit.factory.center)
                        + power_required(unit, points[0], unit.point)
                        < unit.available_power_lichen_digs()
                    )
                ):
                    lprint(
                        f"TRY UniChokeLichen ATTACK {unit} against {points} with kills {kills}"
                    )
                    if UniChokeLichen(unit=unit, targets=points).execute():
                        points = [p for p in points if unit not in p.dug_by]
                        if unit.is_heavy:
                            heavy_sent = True
                        n_squad += 1
                        if n_squad == n_points:
                            if reserve_points:
                                new_point = reserve_points.pop(0)
                                points.append(new_point)
                                n_points += 1
                            break

    gb = agent.game_board
    for unit in units:
        if (
            unit.is_ambushing
            and unit.point.enemy_lichen
            and unit.point == unit.last_point
        ):
            enemy_factory = gb.get_factory(unit.point.lichen_strains)
            if (
                enemy_factory
                and enemy_factory.consider_lichen_assault()
                or unit.power > 0.98 * unit.battery_capacity
            ):

                if UniLichen(unit=unit, targets=[unit.point]).execute():
                    continue


def handle_lichen_defense(agent, units, other_units, max_distance=12):
    for point, (t, enemy_unit, power) in agent.lichen_points_under_attack.items():
        if enemy_unit.is_targeted:
            continue

        lprint(f"DEFEND {point} at {t} against {enemy_unit} with {power} power")
        factory = point.closest_own_factory
        enemy_strength = enemy_unit.combat_strength(has_moved=True, power_left=power)

        targeted = False
        other_candidates = set(other_units) - set(units)
        min_t = t
        max_t = t + enemy_unit.digs[point]

        for unit in sorted(units, key=lambda u: u.point.distance_to(point)):
            if unit.is_heavy == enemy_unit.is_heavy:
                if not unit.has_cargo:
                    distance = unit.point.distance_to(point)
                    if distance > max_distance:
                        break

                    if action_if_possible(
                        agent,
                        Defend,
                        unit,
                        point,
                        min_t,
                        max_t,
                        enemy_strength,
                        enemy_unit,
                    ):
                        targeted = True
                        break

        if not targeted:
            for unit in sorted(
                other_candidates, key=lambda u: u.point.distance_to(point)
            ):
                distance = unit.point.distance_to(point)

                if distance > max_distance:
                    break

                # don't unplan if not an exact hit or pressure
                if distance < min_t - 1:
                    continue

                # don't unplan if too far away to catch. sorted so break
                if distance > max_t:
                    break
                if (
                    unit.is_heavy == enemy_unit.is_heavy
                    and unit.total_cargo == 0
                    and not unit.is_charger
                    and not (
                        (unit.point.ice or unit.point.ore) and unit in unit.point.dug_by
                    )
                ):
                    if action_if_possible(
                        agent,
                        Defend,
                        unit,
                        point,
                        min_t,
                        max_t,
                        enemy_strength,
                        enemy_unit,
                    ):
                        targeted = True
                        break


def camp_lichen(agent, unit: Unit, verbose=False):
    point = unit.point
    if (
        point.adjacent_factories
        and unit.factory.unit_id not in point.adjacent_factories
        and len(unit.action_queue) > 0
    ):
        if not (unit.dies or unit.could_die or unit.must_move):
            unit.remove_from_work_queue()
            lprint(f"Already camping (unit {unit.unit_id}={unit.unit_type}) {point}")
            return True

    camp = Camp(unit)
    if camp.execute():
        lprint(f"CAMPING (unit {unit.unit_id}={unit.unit_type}) ")
        return True


def consider_attack_next(
    unit: Unit, target_point: Point, target_unit: Unit, is_adjacent: bool
):
    # don't bother attacking a unit that makes it back to the factory
    is_intercept = not is_adjacent

    if unit < target_unit:
        return False

    if not target_unit.can_move:
        return True

    if is_adjacent and target_unit.direct_path_to_factory:
        return False

    # only attack with metal/water haulers if instakill
    if unit.cargo.metal > 0 or unit.cargo.water > 0:
        return False

    if unit > target_unit:  # heavy against light
        return is_intercept and target_unit.starts_pingpong_attack()

    if target_unit.is_digging_next and (
        target_unit.point.ore or target_unit.point.ice or target_unit.point.own_lichen
    ):
        return True

    # don't bother chasing moving targets
    if is_adjacent and target_unit.is_light and target_unit.keeps_moving_or_factory:
        return False

    # # equal strength
    if (
        (not is_adjacent and unit.power < target_unit.power)
        or is_adjacent
        and (target_point.closest_factory.is_enemy)
    ):
        return unit.game_board.steps_left < 10 and target_point.enemy_lichen

    return True


def intercept(unit: Unit):
    point = unit.point
    unit_grid = unit.game_board.unit_grid_in_time[1]

    primary_candidates = []
    secondary_candidates = []
    for p in point.adjacent_points():
        if p.factory:
            continue
        if p in unit_grid:
            enemies = [
                u
                for u in unit_grid[p]
                if u.is_enemy and not u.dies
                # prevent interception code on adjacent targets
                and (not u.can_move or not u.point == p)
            ]
            if len(enemies) == 0:
                continue

            if unit in unit_grid[p]:
                keep_attacking = all(
                    consider_attack_next(unit, p, e, is_adjacent=False) for e in enemies
                )
                if keep_attacking:
                    lprint(
                        f"ALREADY ATTACKING INTERCEPT (unit {unit}={unit.unit_type}) {point}->{p} ({[f'{e.unit_id}: {e.unit_type}' for e in enemies]})"
                    )
                # else:
                #     unit.must_move = any(u >= unit for u in enemies)
                #     unit.unplan("problematic intercept")
                return keep_attacking

            own = [u for u in unit_grid[p] if u.is_own]
            if len(own) > 0:
                continue

            op_strength = max(
                [
                    u.combat_strength(
                        has_moved=u.is_moving_next, power_left=u.power_in_time[1]
                    )
                    for u in enemies
                ]
            )

            my_strength = unit.combat_strength_move(
                1, True, current_point=point, target_point=p, start_power=unit.power
            )
            if my_strength < op_strength:
                continue
            if my_strength > op_strength:
                primary_candidates.append((p, enemies))
                continue

            # both will die so needs to be worth it, if close to own factory, needs to be worth more
            if my_strength == op_strength:
                enemy_value = sum([u.value for u in enemies])
                value_factor = 1.1 if unit.near_my_factory() else 1
                if unit.value * value_factor < enemy_value:
                    secondary_candidates.append((p, enemies))
                    continue

    for p, enemies in primary_candidates:
        if consider_attack_next(unit, p, enemies[0], is_adjacent=False):
            attack = Attack(unit, targets=[p], target_bots=enemies)
            lprint(
                f"Trying to intercept {enemies[0]} at {p} with {unit}: {primary_candidates}"
            )
            if attack.execute(is_intercept=True):
                return True

    for p, enemies in secondary_candidates:
        attack = Attack(unit, targets=[p], target_bots=enemies)
        lprint(
            f"Trying to intercept {enemies[0]} at  {p} with {unit} (secondary) {secondary_candidates}"
        )
        if attack.execute(is_intercept=True):
            return True

    return False


def attack_distance(unit: Unit, max_distance=7):
    point = unit.point
    heavy_intercept = unit.factory.dangerous_enemy_near() and unit.is_heavy

    if heavy_intercept:
        candidates = unit.factory.nearby_heavy_enemies()
    else:
        reference = unit.factory if unit.factory else unit.point
        MAX_RANGE = 5 if unit.factory else 4

        candidates = reference.points_within_distance(MAX_RANGE)

        candidates = [p.unit for p in candidates if p.unit and p.unit.is_enemy]

    candidates = [
        u
        for u in candidates
        if not u.dies
        and not u.is_targeted
        and u.point.closest_enemy_factory_distance > 1
        and unit.consider_attack_distance(u)
    ]

    # check that target is still there in distance points
    if len(candidates) == 0:
        return False

    # gameboard = unit.game_board

    # targets = []
    # for candidate in candidates:
    #     # distance = point.distance_to(candidate)
    #     # units_at_t = gameboard.unit_grid_in_time[distance][candidate]
    #     # if unit in units_at_t:
    #     #     lprint(
    #     #         f"ALREADY ATTACKING (unit {unit}={unit.unit_type}) {point}->{candidate}"
    #     #     )
    #     #     return True
    #     # if candidate.unit not in units_at_t:
    #     #     lprint(f"UNIT {candidate.unit} NOT AT {candidate} AT TIME {distance}")
    #     #     continue
    #     targets.append(candidate)

    for enemy_unit in sorted(candidates, key=lambda u: point.distance_to(u.point)):
        lprint(f"TRY distant attack {unit} against {enemy_unit}")
        if target_if_possible(unit, enemy_unit, max_distance=max_distance):
            break

        # req_power = -2 * power_required(unit, point, target) - unit.comm_cost * 2
        # enough_power = unit.full_power or (target.unit.power > unit.power - req_power)

        # start_charge = point.factory is not None
        # if not enough_power:
        #     if (
        #         point.closest_own_factory_distance <= 2
        #         and unit.available_power(True) - req_power > target.unit.power
        #     ):
        #         start_charge = True
        #     else:
        #         continue

        # if Attack(
        #     unit, targets=[target], target_bots=[target.unit], start_charge=start_charge
        # ).execute():
        #     return True

    return False


def update_adj_attack_pattern(unit: Unit, target: Point):
    point = unit.point
    enemy = target.unit

    if enemy.next_point != point:
        return
    target_move_cost = unit.move_cost_rubble(target.rubble)

    combat_strength = unit.combat_strength(True, unit.power - target_move_cost)

    alternatives = [
        p
        for p in point.adjacent_points()
        if p != target
        and p.rubble < target.rubble
        and len(unit.game_board.unit_grid_in_time[1][p]) == 0
        and unit.game_board.attack_risk_in_time[1][p] <= combat_strength
    ]

    if len(alternatives) == 0:
        return

    min_rubble_point = min(alternatives, key=lambda p: p.rubble)
    alternative_move_cost = unit.move_cost_rubble(min_rubble_point.rubble)
    if alternative_move_cost + unit.comm_cost < target_move_cost:

        first_action = unit.move(point.get_direction(min_rubble_point), 1, 1)
        second_action = unit.move(min_rubble_point.get_direction(point), 1, 1)

        action_queue = [first_action, second_action]
        actions = unit.game_board.agent.actions
        unit.action_queue = action_queue
        actions[unit.unit_id] = action_queue
        unit.remove_from_work_queue()
        lprint(
            f"{unit}: MORE EFFICIENT TO UPDATE was {unit.start_action_queue} --> creating pingpong {point.xy} <-> {target.xy}: {action_queue}"
        )


def handle_adjacent_enemies(unit: Unit, verbose=False):
    point = unit.point
    agent = unit.game_board.agent

    adj_enemies_points = [
        p
        for p in point.adjacent_points()
        if p.unit and p.unit.is_enemy and not p.factory  # and not p.unit.dies
    ]

    if verbose:
        lprint(f"{unit}: ADJACENT ENEMIES {adj_enemies_points}")

    if not adj_enemies_points:
        return False

    unit_grid = unit.game_board.unit_grid_in_time[1]

    # check attack
    targets = [
        p
        for p in adj_enemies_points
        if consider_attack_next(unit, p, p.unit, is_adjacent=True)
    ]

    if verbose:
        lprint(f"{unit}: targets {targets}")

    if len(targets) > 0:
        # check if we are not already attacking another unit (unless better is instakill)
        if not any([not t.unit.can_move for t in targets]):
            if any([t.unit in unit.kills for t in targets]):
                if verbose:
                    lprint(f"{unit}: ALREADY ATTACKING OTHER", unit.kills)
                return True  # already killing

        keep_targets = []
        for target in targets:
            units_at_target = unit_grid[target]

            # don't move in if another unit is moving in
            if any([u for u in units_at_target if u.is_own and u >= target.unit]):
                continue
            keep_targets.append(target)
            # lprint(f"ALREADY ATTACKING {target} {units_at_target}")
        targets = keep_targets

        target = select_target(unit, targets, verbose=verbose)
        if verbose:
            lprint(f"{unit}: selected target {target}")
        if target is None:
            return False

        lprint(f"ATTACKING {unit}={unit.unit_type} -->{targets} selected {target.xy}")

        # check if already attacking
        # for p in targets:
        if unit in units_at_target:
            lprint(
                f"ALREADY ATTACKING ADJACENT (unit {unit}={unit.unit_type}) {point}->{target}"
            )
            if (
                target.unit.direct_path_to_factory
                and not unit.kills
                and not unit.is_defending
            ):
                # lprint(f"TARGET {target} HAS DIRECT PATH TO FACTORY")
                unit.unplan("adjacent_attacking target that has direct path to factory")
            else:
                # check if unit is moving my way, in which case it may be optimal to change the attackpattern if target has rubble
                # update_adj_attack_pattern(unit, target)
                unit.try_replan = False
                return True

        target = select_target(unit, targets, check_factory=True)
        if target is None:
            return False

        lprint(
            f"Trying to adjacent attack {target.unit} with {unit}. target.unit.dies: "
            f"{target.unit.dies} target.unit.dies_by: {target.unit.dies_by}"
        )
        if not target.unit.dies:
            attack = Attack(unit, targets=[target], target_bots=[target.unit])
            if attack.execute():
                lprint(
                    f"ATTACK ADJACENT TARGETS (unit {unit}={unit.unit_type} -->{[target]})"
                )
                # unit.game_board.agent.attack_log[unit.unit_id][unit.game_board.step] = {
                #     "target": target.unit.unit_id,
                #     "my_pos": unit.point.xy,
                #     "target_pos": target.xy,
                # }
                return True

            # if target_if_possible(unit, target.unit, max_distance=3):
            #     return True

    # unit already moving

    if unit.next_point is not None and unit not in unit_grid[point]:
        return False

    # first check if enemy can attack wrt power
    enemies = [
        p.unit for p in adj_enemies_points if not unit.point.factory and unit <= p.unit
    ]
    enemies_2 = [
        u
        for u in enemies
        if u.power
        >= u.move_cost_rubble()
        + u.is_heavy * point.rubble
        + u.comm_cost * (u not in unit_grid[point])
    ]

    if verbose:
        lprint(f"{unit}: enemies {enemies}, enemies_2 {enemies_2}")
    if enemies_2:
        # unit.must_move = True

        lowest_rubble_enemy = min(enemies_2, key=lambda u: u.point.rubble)

        if unit.is_heavy and not unit.must_retreat:
            # near enemy it is better to keep enemy busy, at home we can lure enemy to spend more power
            if (
                unit.point.closest_factory.is_enemy
                and not lowest_rubble_enemy.next_point.factory
            ):
                moving_in = lowest_rubble_enemy.point == unit.next_point
                cost = unit.move_cost_rubble(lowest_rubble_enemy.point.rubble) + (
                    unit.comm_cost if not moving_in else 0
                )

                target = lowest_rubble_enemy.point
                req = unit.required_from_to(target, unit.factory.center) + cost
                if unit.power > req:
                    if moving_in:
                        lprint(f"{unit} is already moving in, let's do so")
                        unit.remove_from_work_queue()
                        return True
                    else:
                        lprint(f"{unit} enemy is stronger, but let's move in")
                        attack = Attack(
                            unit,
                            targets=[target],
                            target_bots=[lowest_rubble_enemy],
                        )
                        if attack.execute():
                            return True

        # if not unit.dies and not unit.could_die and not unit.can_be_replanned():
        #     unit.unplan("ATTACK RISK")
        #     assert (
        #         IS_KAGGLE or False
        #     ), "{unit} SHOULD NOT REACH HERE NOT IMPLEMENTED: DEFENSE WHERE UNIT WAS NOT UNPLANNED EARLIER"
    else:
        # no risk, no need to do anything
        pass

    return False


def set_ambush(agent, unit: Unit):

    if not agent.opponent_bot_potential:
        return False

    if unit.factory.dangerous_enemy_near():
        return False

    if not (
        (unit.is_heavy and unit.factory.n_heavies > 1)
        or len(unit.factory.units) >= MAX_UNIT_PER_FACTORY * 0.9
    ):
        return False

    factory = unit.point.factory
    at_factory_with_power = (
        factory
        and (
            unit.power + factory.available_power() > 3000
        )  # only ambush with high power, even for lights
        # and factory.cargo.water > 100
    )

    # outside_with_power = not factory and unit.power > unit.battery_capacity / 4
    if at_factory_with_power:  # or outside_with_power:
        lprint(f"{unit}: AMBUSH FROM handle_low_prio_combat")
        ambush = Ambush(unit)
        if ambush.execute():
            return True
    return False


def handle_last_minute_lichen(agent, units: List[Unit]):
    steps_left = agent.game_board.steps_left
    if steps_left > 1:
        return

    for unit in units:
        if (
            unit.point.lichen
            and unit.point.lichen_strains in agent.enemy_lichen_strains
            and not unit.is_digging
            and not unit.self_destructs
        ):
            if unit.power < unit.comm_cost + unit.unit_cfg.SELF_DESTRUCT_COST:
                continue
            if unit.point.lichen >= steps_left:
                unit.unplan(f"BOMBING {unit}: {unit.point}")
                if unit.is_light:
                    action_queue = [unit.self_destruct()]
                else:
                    action_queue = [unit.dig()]
                unit.action_queue = action_queue
                unit.game_board.agent.actions[unit.unit_id] = action_queue
                unit.remove_from_work_queue()
                continue


def select_target(unit: Unit, target_points, check_factory=False, verbose=False):
    """Orders targets by priority
    * highest prio: units that cannot move
    * strength: first go for heavies if heavy
    * high cargo
    * lowest power
    """
    if verbose:
        lprint(f"SELECT TARGET from {target_points}")
    target_points = [
        t
        for t in target_points
        if (not check_factory or not t.unit.direct_path_to_factory)
        and (not t.unit.can_move or unit.unit_type == t.unit.unit_type)
    ]
    if not target_points:
        if verbose:
            lprint(f"NO TARGETS")
        return None

    targets = [t.unit for t in target_points]

    if verbose:
        t = targets[0]
        lprint((not t.dies or t.next_point not in [u.next_point for u in t.dies_by]))

        lprint(not (t.next_point == unit.point and unit.next_point != t.point))

    targets = [
        t
        for t in targets
        if (not t.dies or t.next_point not in [u.next_point for u in t.dies_by])
        # enemy unit is moving into my position, but I am moving to another position
        and not (
            t.next_point == unit.point
            and unit.next_point != t.point
            and unit.power < t.power
        )
    ]

    if not targets:
        if verbose:
            lprint(f"NO TARGETS 2")
        return None

    min_can_move = min([t.can_move for t in targets])
    candidates = [t for t in targets if t.can_move == min_can_move]

    return sorted(candidates, key=lambda t: (t.total_cargo, -t.power), reverse=True)[
        0
    ].point


def set_killnet(self, units: List[Unit]):
    for unit in units:
        if unit.factory.power_hub_push and not unit.factory.full_power_hub():
            continue

        # if unit.is_heavy:
        #     continue

        if unit.power < 0.95 * unit.battery_capacity:
            continue
        Killnet(unit).execute()


def set_shield(agent, units: List[Unit]):
    sentry_points = get_lichen_sentry_points(agent)
    for unit in units:
        factory = unit.factory

        if (
            factory.power_hub_push
            and not factory.full_power_hub()
            and not (unit.dies or unit.could_die)
        ):
            continue
        if factory.available_water() < 50:
            continue
        if unit.is_heavy:
            if factory.dangerous_enemy_near():
                continue

            if Guard(unit).execute():
                continue

            if unit.available_power() < 0.75 * unit.init_power:
                continue
            # if (
            #     max(factory.metal_in_time) < 100
            #     and agent.game_board.steps_left > 250
            # ):
            #     continue

            # sort by own factory, then by lichen, then by distance
            candidates = sorted(
                [
                    (p, lichen)
                    for p, lichen in sentry_points.items()
                    if p.distance_to(unit.point) < 20
                ],
                key=lambda p_lichen: (
                    p_lichen[0].lichen_strains == factory.id,
                    p_lichen[1]
                    if p_lichen[0].lichen_strains == factory.id
                    else p_lichen[0].distance_to(unit.point),
                ),
                reverse=True,
            )

            next_unit = False
            for p, lichen in candidates:
                lprint(f"TRYING SENTRY POINT {unit} {p}")
                if LichenShield(unit, targets=[p]).execute():
                    del sentry_points[p]
                    next_unit = True
                    break
            if next_unit:
                continue

        if unit.available_power() < 0.33 * unit.battery_capacity:
            continue

        if LichenShield(unit).execute():
            continue

        if unit.available_power() < 0.75 * unit.battery_capacity:
            continue

        lprint("TRYING SHIELD!!!!")

        Shield(unit).execute()


def determine_timed_targets(
    unit: Unit,
    enemy_unit: Unit,
    allow_charge: bool,
    max_t=3,
    max_distance=100,
    verbose=False,
):
    point = unit.point

    can_charge = False
    shortest_distance = 1000
    # t_point_shortest = None
    # t_point_fastest_charge = None

    # current_power = unit.power
    # distance_decreasing = True
    distance = 1000

    count_down_t = None

    time_targets = {}
    time_targets_no_charge = {}

    for t, p in enumerate(enemy_unit.path):
        if verbose:
            lprint(f"t {t} p {p}")
        if p.factory:
            continue
        if t == 0 or t > TTMAX:
            continue

        # cannot kill without moving if same or stronger
        if t == 1 and p == point and enemy_unit >= unit:
            continue
        distance = point.distance_to(p)

        if distance > t or distance > max_distance:
            if verbose:
                lprint(
                    f"distance {distance} > t {t} or {distance} > max_distance {max_distance}"
                )
            continue

        enemy_power = enemy_unit.power_in_time[t]

        if distance < t and allow_charge:
            can_charge = True
            # if t_point_fastest_charge is None:
            #     t_point_fastest_charge = (t, p, distance, distance < t)
        else:
            can_charge = False
            # quick heuristic, does not take rubble and charge into account
            req = power_required(unit, unit.point, p)
            if enemy_power > unit.power - req - unit.comm_cost:
                # needs a charge but cannot
                if verbose:
                    lprint(
                        f"allow_charge {allow_charge}, enemy_power {enemy_power} > unit.power {unit.power} - req {req} - unit.comm_cost {unit.comm_cost}"
                    )
                continue

        if distance < shortest_distance:
            shortest_distance = distance
            # t_point_shortest = (t, p, distance, distance < t)
            # distance_decreasing = True
            count_down_t = max_t
        else:
            # distance_decreasing = False
            count_down_t -= 1
            if count_down_t == 0:
                break

            # print("possibility to charge")
        if can_charge:
            time_targets[t] = p
        else:
            time_targets_no_charge[t] = p
            if t == 1:
                if verbose:
                    lprint(
                        "t == 1: can hit, I should not do weird patterns because enemy can already estimate the risk"
                    )
                # if I can hit it next turn, I should not do weird patterns because enemy can already estimate the risk
                break

        # lprint(t, p, int(enemy_power), distance, can_charge)

    if verbose:
        lprint(f"time_targets {time_targets}")
        lprint(f"time_targets_no_charge {time_targets_no_charge}")
    return (True, time_targets) if time_targets else (False, time_targets_no_charge)


def end_game(self, units: List[Unit], verbose=False):
    steps_left = self.game_board.steps_left
    if steps_left > 25:
        if verbose:
            lprint("END GAME: too many steps left")
        return
    steps_left = self.game_board.steps_left

    candidates = self.get_lichen_targets()

    candidates = [
        c
        for c in candidates
        if c.enemy_lichen
        and not c.dug_by
        and c.lichen >= steps_left
        and not any(u.self_destructs for u in c.visited_by)
    ]
    if not candidates:
        if verbose:
            lprint("END GAME: no candidates")
        return

    primary_candidates = [c for c in candidates if c.lichen >= 95]
    secondary_candidates = [c for c in candidates if c.lichen < 95]

    if verbose:
        lprint(f"primary_candidates {primary_candidates}")
        lprint(f"secondary_candidates {secondary_candidates}")
    sorted_units = sorted(units, key=lambda u: u.point.closest_enemy_factory_distance)
    for unit in sorted_units:
        if unit.is_heavy:
            if steps_left > 10:
                continue

        if unit.power < unit.comm_cost + unit.unit_cfg.SELF_DESTRUCT_COST:
            if verbose:
                lprint(f"END GAME: {unit} not enough power")
            continue

        if (
            unit.point.closest_enemy_factory_distance > steps_left * 2
            and steps_left > 10
        ):
            if verbose:
                lprint(
                    f"END GAME: {unit} not enough power",
                    unit.point.closest_enemy_factory_distance,
                    ">",
                    2 * steps_left,
                )

            break

        if not candidates:
            return

        point = unit.point

        def handle_candidates(try_candidates):
            if verbose:
                lprint(f"handle_candidates {try_candidates}")
            my_candidates = sorted(try_candidates, key=lambda c: c.distance_to(point))
            candidates_left = my_candidates.copy()

            for c in my_candidates:
                if c.dug_by:
                    continue

                distance = point.distance_to(c)

                # do not want kamikaze too early
                if not (unit.dies or unit.could_die) and distance < steps_left - 5:
                    return True  # not planned, but too early and we dont want secondary targets being picked up
                    if verbose:
                        lprint(f"END GAME: {unit} too early")
                    break

                if distance > steps_left - 1:
                    break

                if c not in candidates_left:
                    continue

                neighbours = [c2 for c2 in my_candidates if c2.distance_to(c) <= 2]

                action_type = Kamikaze  # if unit.is_light else UniLichen
                action = action_type(unit=unit, targets=neighbours)
                if action.execute():
                    return True
                else:
                    candidates_left = [
                        c2 for c2 in candidates_left if c2 not in neighbours
                    ]

        if not handle_candidates(primary_candidates):
            handle_candidates(secondary_candidates)


def monitor_surroundings(agent):
    sentries = [u for u in agent.units if u.is_shield or u.is_killnet or u.is_guard]
    for unit in sentries:
        if unit.power < 0.75 * unit.battery_capacity:
            continue

        sentry_done = False
        for p in unit.last_point.points_within_distance(RADAR_DISTANCE):
            if p.ore or p.ice:
                if p.dug_within_2 and p.dug_by:
                    digger = p.dug_by[0]
                    if digger.is_own or digger > unit:
                        continue

                    lprint(f"Sentry {unit} detected digging, trying to ambush {p}")
                    if AmbushRSS(
                        unit=unit, targets=p.points_within_distance(1)
                    ).execute():
                        sentry_done = True
                        break

            for ap in p.adjacent_points():
                if ap.dug_within_2 and ap.dug_by:
                    digger = ap.dug_by[0]
                    if digger.is_own or digger > unit:
                        continue
                    lprint(f"Sentry {unit} detected digging, trying to intercept {p}")
                    if target_if_possible(unit, digger, max_distance=4):
                        sentry_done = True
                        break

            if p.enemy_lichen:
                lprint(f"Sentry {unit} detected enemy lichen, trying to intercept {p}")
                enemy_factory = agent.game_board.get_factory(p.lichen_strains)
                if (
                    enemy_factory
                    and unit.dig_lichen()
                    and enemy_factory.consider_lichen_assault()
                ):
                    action = UniLichen(unit=unit)
                    action.targets = action.get_targets(enemy_factory)
                    if action.execute():
                        sentry_done = True
                        break

            if (
                p.unit
                and p.unit.is_enemy
                and p.unit.point.own_lichen
                and p.unit.is_heavy == unit.is_heavy
            ):
                if not unit.is_targeted or any(
                    [
                        ap
                        for ap in p.adjacent_points()
                        if ap.unit
                        and ap.unit.is_own
                        and ap.unit.is_heavy == p.unit.is_heavy
                    ]
                ):
                    lprint(
                        f"Sentry {unit} detected enemy unit, trying to intercept {p}"
                    )
                    if target_if_possible(unit, p.unit, max_distance=4):
                        sentry_done = True
                        break

        if sentry_done:
            continue
