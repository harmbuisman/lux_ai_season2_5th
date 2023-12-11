import numpy as np

from lux.actions import Attack, ChargeAtFactory, Charger, Retreat, UniLichen, IceHub
from lux.combat import target_if_possible
from lux.constants import (
    BASE_DIG_FREQUENCY_ICE,
    BREAK_EVEN_ICE_TILES,
    DIGS_TO_TRANSFER_ICE,
    BASE_DIG_FREQUENCY_ORE,
    IS_KAGGLE,
    RECHARGE,
    MUST_RETREAT_UP_TO_STEPS_LEFT,
    UNPLAN_ORE_OUT_OF_WATER_TIME,
    BASE_DIG_FREQUENCY_ORE_PUSH,
)
from lux.router import power_required, get_optimal_path
from lux.unit import Unit
from lux.utils import lprint


def continue_or_unplan_mark_for_replan(unit: Unit, msg: str = ""):
    bots_harassed = unit.current_move_prevents_dig_from()
    if bots_harassed:
        lprint(
            f"{unit}: {msg} but not cancelling at it prevents from digging: {unit.current_move_prevents_dig_from()} "
        )
        previous_target = unit.attacks_bot
        if previous_target:
            if unit in previous_target.targeted_by:
                previous_target.targeted_by.remove(unit)
            if not previous_target.targeted_by:
                previous_target.is_targeted = False

        unit.attacks_bot = bots_harassed[0]
        unit.is_attacking = True
        unit.skip_adjacent_attack = True

        for u in bots_harassed:
            u.is_targeted = True
            if unit not in u.targeted_by:
                u.targeted_by.append(unit)

        unit.remove_from_work_queue()
        return True

    point = unit.point

    if unit.game_board.attack_risk_in_time[1][point] < unit.combat_strength(
        has_moved=False,
        power_left=unit.power + unit.charge_rate_now,
    ):
        unit.unplan(msg)
        return True
    else:
        lprint(f"{unit}: mark for replan {msg}")
        if (
            unit.go_home_power_left()
            - unit.move_power_cost(0, unit.point, unit.next_point)
            - unit.move_power_cost(0, unit.next_point, unit.point)
            < 0
        ):
            lprint(f"{unit}: MUST RETREAT!")
            unit.could_die = True
            unit.must_retreat = True

        unit.add_to_work_queue()
        if unit.attacks_bot:
            unit.attacks_bot.is_targeted = False
        unit.try_replan = True


def unplan_units(self):
    gb = self.game_board
    steps_left = gb.steps_left
    unit: Unit
    for unit in self.units:
        factory = unit.factory
        point = unit.point
        if unit.has_been_reset:
            continue

        if (
            unit.is_hub
            # and unit.is_heavy
            and unit.under_attack_dies_at_t
            and (unit.power < 100 or unit.under_attack_dies_at_t <= 3)
        ):
            unit.unplan("hub under attack and dies")
            continue

        if unit.digs_own_lichen:
            unit.unplan("digging own lichen")
            continue

        if unit.illegal_self_destruct:
            unit.unplan("illegal self destruct")
            continue

        if unit.illegal_lichen_dig:
            unit.unplan("digging rubble next to enemy lichen")
            continue

        if unit.illegal_move:
            unit.unplan("illegal move (enemy factory or off map)")
            continue

        if unit.illegal_pickup:
            unit.unplan("illegal pickup SHOULD NOT HAPPEN TOO MUCH!!!")
            continue

        if not is_valid(self, unit):
            unit.unplan("invalid unit")
            continue

        if unit.lower_power_hub:
            if (
                (
                    np.max(unit.power_in_time[:4])
                    < unit.base_move_cost + unit.comm_cost + unit.dig_cost
                )
            ) and len(  # only unplan if enemy heavy in the area
                [
                    u
                    for u in self.opponent_heavies
                    if u.point.distance_to(unit.point) <= 12
                ]
            ) > 0:
                unit.unplan("hub with lower power")
                continue

        # if unit.needs_final_attack_action:
        #     assert (
        #         False
        #     ), f"{unit} >> NEEDS TO IMPLEMENT UNPLAN FOR ATTACKING BOT, {unit.attacks_bot}"
        #     unit.unplan(f"needs final attack action for attacking {unit.attacks_bot}")
        #     continue

        unplan_when_at_risk = True
        if (unit.dies or unit.could_die) and len(unit.action_queue) > 0:
            if unit.dies_by_own and not unit.could_die:
                if not (
                    all([unit < u for u in unit.dies_by]) and unit.can_move
                ) and all([u.is_attacking for u in unit.dies_by]):
                    for u in unit.dies_by:
                        u.unplan(f"attacking unit kills own unit {unit}")
                        # continue --> could still need to be unplanned for another reason
                else:
                    unit.unplan("dies by own unit")
                    continue
            else:
                if unit.could_die and unit.kills_next:
                    if any([not u.can_move for u in unit.kills_next]):
                        # killing a unit, accept the risk
                        unit.remove_from_work_queue()
                        continue
                if unit.could_die and unit.digs_lichen and unit.next_point.enemy_lichen:
                    if (
                        len(
                            [
                                p
                                for p in unit.next_point.adjacent_points()
                                if p.unit and p.unit.is_enemy and p.unit.is_light
                            ]
                        )
                        > 0
                    ):
                        # light can kill me
                        unplan_when_at_risk = steps_left > 10
                    else:
                        # only heavy attack risk.
                        unplan_when_at_risk = not self.unit_dominance

                if unit.dies or (unit.could_die and unplan_when_at_risk):
                    if (
                        unit.is_light
                        and steps_left < 10
                        and unit.point.lichen > steps_left
                        and unit.point.enemy_lichen
                        and not unit.dies_by_own
                    ):
                        lprint(
                            f"{unit}: dieing or could die, but on enemy lichen so don't care"
                        )
                        unit.remove_from_work_queue()
                        continue
                    unit.unplan("will die otherwise")

                if unit.dies and not unit.dies_by_own:
                    has_exit_position = False
                    for p in point.adjacent_points():
                        if gb.attack_risk_in_time[1][p] > unit.combat_strength(
                            has_moved=False,
                            power_left=unit.power
                            + unit.charge_rate_now
                            - unit.comm_cost
                            - unit.move_cost_rubble(p.rubble),
                        ):
                            continue
                        units_at_p = gb.unit_grid_in_time[1][p]
                        if not units_at_p:
                            has_exit_position = True
                            break
                    unit.needs_escape = not has_exit_position
                    if unit.needs_escape:
                        if (
                            unit.point.distance_to(unit.factory) < 8
                            and ChargeAtFactory(unit=unit).execute()
                        ):
                            continue
                        if Retreat(unit).execute():
                            continue

                if unit.dies or unplan_when_at_risk:
                    if (
                        steps_left > MUST_RETREAT_UP_TO_STEPS_LEFT
                        and unit.go_home_power_left() <= 2 * unit.base_move_cost
                    ):
                        unit.must_retreat = True
                    continue

        if unit.has_queue and unit.action_queue[0][-2] >= 8000:
            if not unit.dies and not (unit.could_die and unplan_when_at_risk):
                if (
                    unit.point.enemy_lichen
                    # don't dig disconnected lichen unless it is a unichoke hit
                    and (
                        not self.game_board.agent.connected_lichen_map[unit.point.xy]
                        == -1
                        or unit.action_queue[0][-2] > 8500
                    )
                    and UniLichen(unit=unit, targets=[unit.point]).execute()
                ):
                    lprint(f"UNILICHEN HIDDEN DIG {unit} {unit.action_queue}")
                    continue
                else:
                    unit.unplan("unilichen failed")
                    continue

        if (
            (unit.is_killnet or unit.is_shield)
            and unit.point == unit.last_point
            and (
                unit.power >= unit.init_power * 0.5
                or (unit.point.enemy_lichen and unit.point.lichen in [8, 9])
            )
        ):
            surroundings = [
                p for p in unit.point.points_within_distance(1) if p.enemy_lichen
            ]
            if surroundings:
                enemy_factory = gb.get_factory(surroundings[0].lichen_strains)
                if enemy_factory and enemy_factory.consider_lichen_assault():
                    lprint(f"{unit} Shield next to enemy lichen ")
                    if UniLichen(unit=unit, targets=surroundings).execute():
                        continue
                    else:
                        unit.unplan(
                            f"{unit} Shield next to enemy lichen, but unilichen failed"
                        )
                        continue

        if unit.digs_nothing:
            unit.unplan("digging nothing")
            continue

        if unit.digs_disconnected_lichen():
            unit.unplan("digging disconnected lichen")
            continue

        if (
            unit.is_hub
            and len(set(unit.path)) == 1
            and unit.power < unit.dig_cost + unit.comm_cost + unit.base_move_cost
            and any(
                [
                    p
                    for p in point.points_within_distance(3)
                    if p.unit and p.unit.is_enemy and p.unit.is_heavy
                ]
            )
        ):
            unit.unplan("hub low on power and enemy heavy nearby")
            continue

        if (
            unit.ends_recharge(RECHARGE.AMBUSH)
            and unit.last_point.closest_enemy_factory_distance > 3
        ):
            unit.unplan("No factory near to ambush")
            if unit.is_heavy:
                # check if there is a heavy nearby that is released from the factory
                for p in unit.last_point.points_within_distance(6):
                    if not p.unit or p.unit.is_own or p.unit.is_light or p.unit.dies:
                        continue
                    enemy_unit = p.unit
                    enemy_unit_point = enemy_unit.point

                    distance = point.distance_to(point.closest_enemy_factory)

                    charged = enemy_unit.charge_up_to(distance)

                    power_needed = (
                        power_required(
                            enemy_unit,
                            enemy_unit_point,
                            factory.closest_tile_no_chargers(enemy_unit_point),
                        )
                        + enemy_unit.comm_cost
                    )
                    available = enemy_unit.power + charged

                    if power_needed < available:
                        if Attack(
                            unit=unit,
                            targets=[enemy_unit_point],
                            target_bots=[enemy_unit],
                        ).execute():
                            break
            continue

        if (
            unit.is_hub
            and point.ore
            and factory.cargo.metal > 250
            and unit.cargo.ore == 0
        ):
            unit.unplan("ore hub, but swimming in it")
            continue

        if unit.is_hub and point == unit.last_point and not point.adjacent_factories:
            lprint(
                f"RESETTING {unit.unit_type} HUB {unit.unit_id} at {point}",
            )
            unit.unplan("hub")
            continue

        if (
            unit.is_power_hub
            and factory.cargo.water < 50
            and len(factory.heavy_ice_hubs) == 0
            and len(factory.ice_hubs) > 0
        ):
            if IceHub(unit=unit).execute():
                continue
            # unit.unplan("factory low on water and no ice hubs")
            # continue

        if (  # ore hub, but low on water
            unit.is_hub
            and point.ore
            and factory.out_of_water_time() < UNPLAN_ORE_OUT_OF_WATER_TIME
            and unit.cargo.ore == 0
            and len(factory.ice_hubs) > 0
            and len(factory.heavy_ice_hubs) == 0
        ):
            lprint(
                f"RESETTING {unit.unit_type} HUB {unit.unit_id} at {point} ore hub, but low on water",
            )
            unit.unplan("hub")
            continue

        if (  # ore hub, but lake cleared
            unit.is_hub
            and point.ore
            and unit.cargo.ore == 0
            and not factory.heavy_ice_hubs
            and len(factory.ice_hubs) > 0
            and factory.get_lichen_lake_size() >= BREAK_EVEN_ICE_TILES
            and (factory.cargo.water < 50 or factory.cargo.metal > 150)
        ):
            lprint(
                f"RESETTING {unit.unit_type} HUB {unit.unit_id} at {point} ore hub, but lake cleared",
            )
            unit.unplan("hub")
            continue
        # if unit.is_charger:
        #     action_queue = unit.action_queue.copy()
        #     next_pickup_action = [a for a in action_queue[:1] if a[0] == 2][0]
        #     if next_pickup_action:
        #     amount = pickup_action[-3]

        if unit.is_retreating or (
            unit.charges_at_factory
            and unit.point.closest_own_factory_distance > 12
            and len(unit.factory.units) > 10
        ):
            unit_combat_strength = unit.combat_strength(
                has_moved=False, power_left=unit.power
            )

            must_retreat = (
                steps_left > MUST_RETREAT_UP_TO_STEPS_LEFT
                and unit.go_home_power_left() <= 2 * unit.base_move_cost
            )

            units_nearby = [
                p
                for p in point.points_within_distance(4 if must_retreat else 2)
                if p.unit
                and p.unit.is_enemy
                and (
                    p.unit.is_heavy == unit.is_heavy
                    or p.distance_to(point) == 1
                    or p.unit.next_point.distance_to(point) == 1
                )  # no need to retreat if no unit of the same type
                and p.unit.combat_strength(has_moved=True) > unit_combat_strength
            ]
            if not units_nearby:
                msg = "retreating" if unit.is_retreating else "distant charging at home"
                unit.unplan(f"{msg}, but no danger")
                continue

        if (
            unit.is_hub
            and unit.dig_frequency == BASE_DIG_FREQUENCY_ORE_PUSH
            and unit.point.ore
            and unit.point == unit.last_point
            and not unit.cargo.ore
            and unit.factory.heavies_for_full_power_hub()
        ):
            action_queue = unit.action_queue.copy()
            action_queue = [a for a in action_queue if a[0] != 0]
            digaction = [a for a in action_queue if a[0] == 3][0]
            digaction[-1] = BASE_DIG_FREQUENCY_ORE
            digaction[-2] = BASE_DIG_FREQUENCY_ORE
            transfer_action = [a for a in action_queue if a[0] == 1][0]
            transfer_action[-3] = (
                BASE_DIG_FREQUENCY_ORE * unit.unit_cfg.DIG_RESOURCE_GAIN
            )
            unit.action_queue = action_queue
            unit.remove_from_work_queue()
            self.actions[unit.unit_id] = action_queue

            lprint(f"REPLAN ORE HUB {unit} TO LOWER FREQUENCY: {action_queue}")
            charging_units = [
                p.unit for p in point.adjacent_points() if p.unit and p.unit.is_charger
            ]
            if not charging_units:
                adj_points = point.adjacent_points()
                charging_units = [
                    u for u in self.units if u.is_charger and u.last_point in adj_points
                ]

            for charger in charging_units:
                # start_charge = unit.power < unit.init_power * 0.75
                Charger(
                    unit=charger,
                    targets=[charger.last_point],
                    # start_charge=start_charge,
                ).execute()

            continue

        if (
            unit.is_hub
            and unit.dig_frequency == BASE_DIG_FREQUENCY_ICE
            and (
                factory.n_connected_tiles >= BREAK_EVEN_ICE_TILES
                or (unit.is_power_hub_push_ice and not unit.factory.power_hub_push)
            )
            # and not unit.cargo.ice
            # and not unit.cargo.ore
        ):

            action_queue = unit.action_queue.copy()
            action_queue = [a for a in action_queue if a[0] != 0]
            digaction = [a for a in action_queue if a[0] == 3][0]
            digaction[-1] = DIGS_TO_TRANSFER_ICE
            digaction[-2] = DIGS_TO_TRANSFER_ICE
            transfer_action = [a for a in action_queue if a[0] == 1][0]
            transfer_action[-3] = DIGS_TO_TRANSFER_ICE * unit.unit_cfg.DIG_RESOURCE_GAIN
            unit.action_queue = action_queue
            unit.remove_from_work_queue()
            self.actions[unit.unit_id] = action_queue

            lprint(f"REPLAN HUB {unit} ON LOW FREQUENCY: {action_queue}")
            charging_units = [
                p.unit for p in point.adjacent_points() if p.unit and p.unit.is_charger
            ]
            if not charging_units:
                adj_points = point.adjacent_points()
                charging_units = [
                    u for u in self.units if u.is_charger and u.last_point in adj_points
                ]

            for charger in charging_units:
                # start_charge = unit.power < unit.init_power * 0.75
                Charger(
                    unit=charger,
                    targets=[charger.last_point],
                    # start_charge=start_charge,
                ).execute()

        if unit.is_charger and point == unit.last_point:

            transfers = [
                a
                for a in unit.action_queue
                if a[0] == 1 and a[-2] > 0 and a[2] == 4 and a[1] != 0
            ]
            unplanned = False

            for a in transfers:
                direction = a[1]
                target_point = point.apply(direction)
                target_unit = target_point.unit
                has_target_unit = (target_unit and target_unit.is_hub) or (
                    any(u.last_point == target_point and u.is_hub for u in self.units)
                )

                if not has_target_unit:
                    unit.unplan("no unit at hub point")
                    unplanned = True
                    break

                enemy_unit = target_unit and target_unit.is_enemy
                if enemy_unit:
                    unit.unplan("enemy unit at hub point")
                    unplanned = True
                    break
            if unplanned:
                continue

                # enemy =
                # if not (

                # ) or (
                #     target_unit
                #     and target_unit.team_id != unit.team_id
                #     or not target_unit.is_hub
                # ):
                #     unit.unplan("charger hub not there anymore")
                #     break

        if unit.attacks_bot:
            # lprint("NEEDS TO IMPLEMENT UNPLAN FOR ATTACKING BOT")
            enemy_unit = unit.attacks_bot

            # moving into its position is fine
            if unit.next_point != enemy_unit.point:
                if enemy_unit.point.distance_to(point) == 1 and not unit.kills_next:
                    continue_or_unplan_mark_for_replan(
                        unit,
                        "attacking bot next to it, but not killing or moving into it next turn",
                    )
                    continue
                if unit.attacks_bot not in unit.kills:
                    # check if target bot is in an intercept position at the time of attack
                    if any(
                        [
                            p == enemy_unit.next_point
                            for p in unit.next_point.adjacent_points()
                        ]
                    ):
                        # we are chasing this bot, call for help for direct attack
                        unit.remove_from_work_queue()
                    else:
                        # check if distance of next step is closer to enemy
                        if unit.next_point.distance_to(
                            enemy_unit.next_point
                        ) >= point.distance_to(enemy_unit.next_point):
                            continue_or_unplan_mark_for_replan(
                                unit,
                                f"attacking bot {unit.attacks_bot}, but not killing it and not closing in"
                                f", next point: {unit.next_point}, target next point: {unit.attacks_bot.next_point}",
                            )
                            # try to change course
                            lprint(
                                f"TRYING TO CHANGE COURSE FOR {unit} to target bot {enemy_unit}"
                            )
                            # if point.distance_to(enemy_unit.point) <= 3:
                            target_if_possible(unit, enemy_unit)
                            continue

                    # if unit.next_point != unit.attacks_bot.point:
                    # unit.unplan("attacking bot, but not at target")
                    # continue

        if unit.target_gone:
            continue_or_unplan_mark_for_replan(unit, "target no longer exists")
            continue

        if unit.is_attacking:
            # don't unplan until almost there, bot could come back
            if len(unit.action_queue) <= 3 and unit.ends_pingpong():
                # only consider kills of same type, or if they are sure kills

                kills = unit.kills_next

                if (
                    len(
                        [
                            u
                            for u in kills
                            if u.is_heavy == unit.is_heavy
                            or (unit.next_point == u.next_point and not u.can_move)
                        ]
                    )
                    == 0
                ):
                    unplan = True
                    if unit.next_point and unit.next_point.unit:
                        enemy_unit = unit.next_point.unit
                        if (
                            enemy_unit.has_cargo
                            or enemy_unit.unit_type == unit.unit_type
                        ):
                            unplan = False
                            unit.remove_from_work_queue()

                    if unplan:
                        continue_or_unplan_mark_for_replan(
                            unit, "attacking, but target gone"
                        )
                        continue

            # check if we still have the option to retreat
            next_point = unit.next_point
            if (
                unit.kills
                and next_point.unit
                and next_point.unit.is_enemy
                and not next_point.unit.can_move
            ):
                pass
            else:
                # if not unit.attacks_bot:
                #     # assume unit is attacking bot in next point
                #     attacks_bot = unit.attacks_bot
                #     if next_point.unit and next_point.unit.is_enemy:
                #         attacks_bot = next_point.unit
                #     lprint(
                #         f"{unit} next_point {next_point} attacks_bot: {attacks_bot} {next_point.unit}"
                #     )
                #     if not unit.attacks_bot:
                #         unit.unplan("attacking, but target gone")
                #         continue
                #     my_lichen_near = any(p.own_lichen for p in point.adjacent_points())
                #     factory_near = unit.near_my_factory()
                #     unit.attacks_bot = attacks_bot
                #     attacks_bot.is_targeted = True

                #     if attacks_bot and not my_lichen_near and not factory_near:
                #         if attacks_bot.direct_path_to_factory:
                #             unit.unplan(
                #                 "attacking, but target is retreating with direct path to factory"
                #             )
                #             continue

                distance = next_point.distance_to(
                    factory.closest_tile_no_chargers(next_point)
                )
                if steps_left > distance:
                    charged = unit.charge_up_to(distance)

                    power_needed = (
                        power_required(
                            unit,
                            next_point,
                            factory.closest_tile_no_chargers(next_point),
                        )
                        + unit.comm_cost
                        + unit.move_cost_rubble(next_point.rubble)
                    )
                    available = unit.power_in_time[1] + charged

                    if available < power_needed:
                        unit.unplan("attacking, but no power to retreat")
                        if unit.dies or unit.could_die:
                            Retreat(unit=unit).execute()
                            continue

        if unit.is_digging_next:
            power_left = unit.go_home_power_left() - unit.dig_cost
            path_home = get_optimal_path(gb, unit.point, unit.factory.center)
            next_point_return = path_home[1]
            if next_point_return.unit:
                next_unit = next_point_return.unit
                if next_unit.is_digging_next and next_unit.digs[next_point_return] > 1:
                    min_rubble = min(
                        [
                            ap.rubble
                            for ap in unit.point.adjacent_points()
                            if ap.closest_own_factory_distance
                            <= point.closest_own_factory_distance
                        ]
                    )
                    additional_cost = unit.move_cost_rubble(min_rubble)
                    if power_left - additional_cost <= 0:
                        unit.unplan("route home is blocked, no power left to get home")
                        continue

        if unit.is_corner_charger:
            max_hub_power = 2950
            hubs = [
                p.unit
                for p in unit.point.adjacent_points()
                if p.unit and p.unit.is_hub and p.unit.last_point == p
            ]

            # prevent too much power in hubs
            my_hubs = [
                u for u in hubs if u in unit.charges_units or u.power < max_hub_power
            ]
            n_hubs = len(my_hubs)
            valid_hubs = [
                u
                for u in hubs
                if u in unit.charges_units
                or u.power < max_hub_power
                and not u.dies
                and not u.could_die
            ]
            n_charges = len([u for u in unit.charges_units if not u.is_attacking])

            if any(u.dies for u in my_hubs):
                unit.unplan("hub dies")
                continue

            if n_hubs != n_charges:
                unit.unplan(
                    f"number of adjacent hubs changed n_hubs {n_hubs}: {my_hubs}, "
                    f"n_charges {n_charges} {unit.charges_units}"
                    f"valid positions {Charger.get_valid_charger_positions(unit.factory)}"
                )
                if len(valid_hubs) == 0:
                    continue

                if self.game_board.steps_left > 2:
                    if Charger(unit=unit, targets=[unit.point]).execute():
                        continue
                    else:
                        assert IS_KAGGLE or False, f"Failed to replan charger {unit}"

        # if (
        #     unit.power > unit.unit_cfg.BATTERY_CAPACITY * 0.9
        #     and unit.is_repeating
        #     and not unit.is_digging
        #     and not unit.is_attacking
        # ):
        #     unit.unplan("repeating unit full power")


def is_valid(self, unit):
    point = unit.point
    if len(unit.action_queue) == 0:
        return True
    action = unit.action_queue[0]

    if action[0] == 1:  # transfer
        if action[1] == 0:  # same point
            if point.factory is None:
                return False

    # no problems
    return True
