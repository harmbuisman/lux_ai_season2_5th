from lux.actions import (
    Charger,
    Dump,
    Ice,
    IceHub,
    Ore,
    OreHub,
    Rubble,
    RubbleRaid,
    SkimRubble,
    UniIce,
    UniLichen,
    UniOre,
    UniRubble,
    UniRubbleLowPrio,
    UniRubbleTopPrio,
    Rebase,
)
from lux.constants import (
    HEAVY_ONLY_ICE,
    LIGHT_ONLY_RUBBLE,
    TTMAX,
    BREAK_EVEN_ICE_TILES,
    MAX_UNIT_PER_FACTORY,
    EARLY_GAME_MAX_LICHEN_TILES_PER_HUB,
    IS_KAGGLE,
)
from lux.unit import Unit
from lux.utils import lprint
from lux.actions import is_skimmed
import heapq
from typing import List


def factory_or_low_heavy_factory(self, unit, min_distance=10):
    if unit.is_light:
        return unit.factory

    if unit.factory.n_heavies == 1:
        return unit.factory

    low_heavy_factories = sorted(
        [
            f
            for f in self.factories
            if unit.is_heavy
            and f.n_heavies == 0
            and f.distance_to(unit.point) <= min_distance
        ],
        key=lambda f: f.distance_to(unit.point),
    )

    return low_heavy_factories[0] if low_heavy_factories else unit.factory


def check_replan_ore_cycle(self, units: List[Unit]):
    step = self.game_board.step
    candidates = sorted(units, key=lambda u: u.is_power_hub, reverse=True)
    for unit in candidates:
        if unit.is_light:
            continue

        factory = unit.factory
        if factory.power_hub_push:
            continue
        if (
            (unit.is_hub or unit.is_power_hub)
            and (unit.point.ice or unit.is_power_hub)
            and unit.total_cargo == 0
            and (
                factory.available_power() + unit.power
                > (2000 if step < 125 else 2500)
                # or (factory.n_heavies == 1 and factory.power > 1500)
            )
            and (
                factory.available_metal()
                < 100
                # < (10 if factory.available_power() <= 3500 else 100)
            )
            and (unit.is_power_hub or factory.available_water() > 75)
            and factory.cargo.ore < 100
            and len(
                [
                    u
                    for u in factory.heavies
                    if not (u.is_hub or u.is_power_hub)
                    and not u.is_charger
                    and u.point.distance_to(factory) < 8
                ]
            )
            == 0
        ):
            lprint(f"{unit}: TRY REPLANNING HUB FOR ORE CYCLE")
            charging_units = unit.charging_units()
            if Ore(unit=unit, start_charge=True, allow_low_efficiency=True).execute():
                unit.is_hub = False
                for cu in charging_units:
                    cu.unplan("charging hub")
                continue

            # heavy needs many digs to be efficient, may not succeed in 50 ticks
            unit.tried_targets = []
            if UniOre(
                unit=unit, start_charge=True, allow_low_efficiency=True
            ).execute():
                unit.is_hub = False
                for cu in charging_units:
                    cu.unplan("charging hub")
                continue


def hub_start_charge(unit):
    return (
        True
        if (unit.power < 250 and (unit.point.closest_own_factory_distance <= 1))
        or (unit.game_board.step < 15 and not unit.factory.power_hub_push)
        or (unit.factory.n_lights < 5 and unit.power < 500)
        or (unit.point.factory is None and unit.go_home_power_left() < 250)
        else False
    )


def handle_dieing_factory(self):
    def try_hubs(units):
        for unit in sorted(
            units,
            key=lambda u: (u.point.distance_to(u.factory.ice_points[0]), -u.power),
        ):
            lprint(f"{unit}: TRY REPLANNING ICE HUB FOR DIEING FACTORY")
            if IceHub(unit, start_charge=hub_start_charge(unit)).execute():
                return True

    for f in self.factories:
        out_of_water_time = f.out_of_water_time()
        if out_of_water_time < 15 and out_of_water_time < self.game_board.steps_left:
            if f.ice_hubs:
                replan_units = [u for u in f.heavies if u.can_be_replanned()]
                if try_hubs(replan_units):
                    continue
                other_units = [u for u in f.heavies if u not in replan_units]
                if try_hubs(other_units):
                    continue


def mine_high_priority(self, units: List[Unit], icehubpass=False, verbose=False):
    for unit in units:
        if unit.dies or unit.could_die:  # makes no sense if chased
            continue
        factory = factory_or_low_heavy_factory(self, unit)
        if (
            unit.is_heavy
            and factory.dangerous_enemy_near()
            and any(
                [
                    u.point.distance_to(unit.point) < 5
                    for u in factory.nearby_heavy_enemies()
                ]
            )
            # need to have some power to start up when outside
            and (unit.point.factory or unit.power > unit.init_power * 0.75)
        ):
            if factory != unit.factory and not unit.point.factory == factory:
                if Rebase(
                    unit, destinations=factory.edge_tiles(no_chargers=True)
                ).execute():
                    continue
            else:
                continue

        if not icehubpass or (
            len(factory.heavy_ice_hubs) == 0
            and (factory.n_heavies <= 1 or factory.available_water() < 50)
        ):
            if factory != unit.factory:
                targets = [
                    p
                    for p in factory.hubs
                    if p.ice and len([u for u in p.dug_by if u.is_heavy]) == 0
                ]
                lprint(
                    f"{unit}: mine_high_priority TRY REPLANNING ICE HUB FOR DIFFERENT FACTORY {factory}"
                )
                old_factory = unit.factory
                unit.factory = factory
                if IceHub(
                    unit, start_charge=hub_start_charge(unit), targets=targets
                ).execute():
                    continue
                unit.factory = old_factory
            else:
                if mine_hub(unit):
                    continue


def mine_low_priority(self, units: List[Unit], noaction=False, verbose=False):
    for unit in units:
        if not unit.point.factory and unit.power < 2 * unit.dig_cost + unit.comm_cost:
            continue

        if mining_light(unit):
            continue

        # if unit.is_light and UniRubble(unit).execute():
        #     lprint(f"Second rubble mining pass succeeded {unit}")
        #     continue

        if unit.is_light:
            continue

        if mine_hub(unit):
            continue

        if mining_heavy(unit, noaction=noaction):
            continue


def bring_back_rss(units: List[Unit]):
    for unit in units:
        if not unit.has_cargo:
            continue

        dump = Dump(unit)
        if dump.execute():
            continue
        else:
            lprint(f"Failed to dump {unit}????")


def set_chargers(agent, units, max_distance=6):
    candidates = [u for u in units if u.is_light and u.total_cargo == 0]

    for f in agent.factories:
        targets, timed_targets = Charger.get_valid_charger_positions(f)

        if not targets:
            continue

        my_candidates = sorted(
            [u for u in candidates if u.point.distance_to(f) <= max_distance],
            key=lambda u: u.point.distance_to(targets[0]),
        )

        for unit in my_candidates:
            if unit.factory != f:
                if unit.point.min_distance_to(tuple(targets)) > max_distance:
                    continue
            # distance = unit.point.distance_to(unit.factory)
            # if distance > max_distance:
            #     break
            lprint("SetChargers", unit, f, targets, timed_targets)
            if timed_targets:
                charger = Charger(unit, timed_targets=timed_targets)
            else:
                charger = Charger(unit, targets=targets)
            if charger.execute():
                break


def mine_hub(unit: Unit, verbose=False):
    if not unit.is_heavy:
        return False
    factory = unit.factory
    assert (
        IS_KAGGLE or factory.team_id == unit.team_id
    ), f"{unit} {factory} {unit.team_id}!={factory.team_id}"

    if factory:
        hubs = factory.hubs

        if verbose:
            lprint(f"mining_hub {unit} {hubs}")

        if not hubs:  # or unit.game_board.step < 50:
            if verbose:
                lprint(f"no hubs {unit} {hubs}")
            return False

        start_charge = hub_start_charge(unit)
        if any(p.ice for p in hubs) and (
            not factory.heavy_ice_hubs or unit.available_power() > 500
        ):
            if verbose:
                lprint(f"try IceHub {unit} {hubs}")

            lprint(
                f"{unit}: mine_high_priority TRY REPLANNING ICE HUB FOR DIFFERENT FACTORY {factory}"
            )

            hub = IceHub(unit, start_charge=start_charge)
            if hub.execute():
                return True

        if factory.available_metal() < 150 and not factory.heavy_ore_hubs:
            if any(p.ore for p in hubs) and (
                not factory.ice_hubs
                or factory.available_power() > 500
                or factory.game_board.step < 10
                or factory.power_hub_push
            ):
                if verbose:
                    lprint(f"try OreHub {unit} {hubs}")

                hub = OreHub(unit, start_charge=start_charge)
                if hub.execute():
                    return True

        # Heavy charger if no hubs left and light chargers on corners, but only if multiple rss
        if (
            len(factory.hubs)
            - len(factory.heavy_ice_hubs)
            - len(factory.heavy_ore_hubs)
            == 0
        ):
            light_corner_chargers = [
                u
                for u in unit.factory.chargers
                if u.is_light
                and u.last_point.is_factory_corner
                and len([p for p in u.point.adjacent_points() if p.ice or p.ore]) == 2
            ]
            if light_corner_chargers:
                targets, timed_targets = Charger.get_valid_charger_positions(
                    unit.factory, allow_heavy_overtake=True
                )
                lprint(
                    f"{unit}: TRYING HEAVY CHARGER light_corner_chargers {light_corner_chargers} {targets}"
                )
                start_charge = unit.power < unit.init_power * 0.75
                if timed_targets:
                    charger = Charger(
                        unit, timed_targets=timed_targets, start_charge=start_charge
                    )
                else:
                    charger = Charger(unit, targets=targets, start_charge=start_charge)
                if charger.execute():
                    return True

    return False


def mining_light(unit: Unit, verbose=False):
    if unit.is_heavy:
        return False
    verbose = True

    factory = unit.factory
    water = factory.cargo.water
    gb = unit.game_board
    step = gb.step  # real step
    almost_out_of_water = factory.out_of_water_time() < 50
    steps_left = gb.steps_left
    point = unit.point

    # dig_ore = (
    #     (factory.cargo.metal < 200 and gb.steps_left > 100)
    #     or unit.cargo.ore > 0
    #     or unit.point.ore
    # ) and (step > 15 or len([h for h in factory.hubs if h.ore]) == 0)

    # dig_ice = (
    #     sum(factory.ore_in_time) > 0
    #     or water < max(step, 100)
    #     or len(factory.units) == 1
    #     or len(factory.ore_points) == 0
    #     or almost_out_of_water
    #     or unit.point.ice
    #     or factory.has_lichen
    # ) and (
    #     # heavy must mine single ice point
    #     factory.n_heavies == 0
    #     or len(factory.ice_points) > 1
    #     or water > 100
    # )

    ACTIONS = {
        UniRubbleTopPrio: -2,
        UniOre: 1,
        UniIce: 2,
        UniRubble: 3,
        UniRubbleLowPrio: 5,
    }

    # increased Ice priorities
    if almost_out_of_water:
        ACTIONS[UniIce] = -10
    if unit.cargo.ice:
        ACTIONS[UniIce] = min(ACTIONS[UniIce], 0)
    if len(factory.ice_points) <= 2 and factory.n_heavies == 0:
        ACTIONS[UniIce] = min(ACTIONS[UniIce], 0)
    if unit.point.ice:
        ACTIONS[UniIce] = min(ACTIONS[UniIce], 0)
    if factory.is_ice_risk():
        ACTIONS[UniIce] = min(ACTIONS[UniIce], 0)

    # increased Ore priorities
    if unit.cargo.ore:
        ACTIONS[UniOre] = min(ACTIONS[UniOre], 0)
    if unit.point.ore:
        ACTIONS[UniOre] = min(ACTIONS[UniOre], 0)
    if step > 15 or len([h for h in factory.hubs if h.ore]) == 0:
        ACTIONS[UniOre] = min(ACTIONS[UniOre], 0)

    # decreased Ore priorities
    if factory.cargo.metal >= 150:
        ACTIONS[UniOre] = 1

    n_ore_diggers = len([u for u in factory.units if u.is_light and u.digs_ore])
    if (
        n_ore_diggers >= factory.n_lights / 2 and factory.lake_large_enough()
    ):  # don't send out more than 50% to dig ore
        free_points = [
            op.rubble
            for op in factory.ore_points[: n_ore_diggers + 1]
            if not op.dug_by and op.rubble == 0
        ]
        ACTIONS[UniOre] = 2 if free_points else 4
    if factory.cargo.metal >= 200:
        ACTIONS[UniOre] = 3
    if factory.cargo.metal >= 250:
        ACTIONS[UniOre] = 4
    if factory.cargo.metal >= 300:
        del ACTIONS[UniOre]

    # increased rubble priorities
    n_ice_hubs = len(factory.ice_hubs)
    lakesize = factory.get_lichen_lake_size()
    supported_lichen_tiles = factory.supported_lichen_tiles()
    if (
        water > 100
        and n_ice_hubs > 1
        and factory.n_lichen_tiles < BREAK_EVEN_ICE_TILES * n_ice_hubs
    ) or (water > 250 and lakesize < supported_lichen_tiles):
        ACTIONS[UniRubble] = 0 if water > 250 else 1

    if factory.has_uncleared_oreroute and factory.is_ore_risk():
        ACTIONS[UniRubble] = min(ACTIONS[UniRubble], 1)

    if factory.n_connected_tiles < factory.n_all_lichen_tiles:
        ACTIONS[UniRubble] = min(ACTIONS[UniRubble], 1 if steps_left > 100 else -1)

    n_heavy_ice_hubs = len(factory.heavy_ice_hubs)

    if (
        water > 100
        and n_heavy_ice_hubs > 0
        and factory.get_lichen_lake_size()
        < EARLY_GAME_MAX_LICHEN_TILES_PER_HUB * n_heavy_ice_hubs
    ):
        ACTIONS[UniRubble] = -1 if len(factory.units) >= MAX_UNIT_PER_FACTORY else 0

    if unit.dig_lichen():
        ACTIONS[UniLichen] = 3 if gb.agent.n_opp_lights > 3 or gb.step < 10 else 0

        if steps_left < 250 and gb.agent.opp_lichen > gb.agent.own_lichen:
            ACTIONS[UniLichen] = min(2, ACTIONS[UniLichen])

        if steps_left < 50:
            ACTIONS[UniLichen] = min(0, ACTIONS[UniLichen])

        # if any(ap.enemy_lichen for ap in point.points_within_distance(1)):
        #     ACTIONS[UniLichen] = min(-1, ACTIONS[UniLichen])

    if LIGHT_ONLY_RUBBLE:
        ACTIONS[UniRubble] = -1

    # Add skimming
    if step < 10 or (
        water < 100 and any(not is_skimmed(p.rubble) for p in factory.ice_hubs)
    ):
        ACTIONS[SkimRubble] = -3

    if (
        unit.factory.available_power() < 1000
        and unit.point.closest_own_factory_distance < 10
    ):
        if UniRubbleLowPrio in ACTIONS:
            del ACTIONS[UniRubbleLowPrio]

    if unit.factory.lake_large_enough():
        if UniRubbleLowPrio in ACTIONS:
            del ACTIONS[UniRubbleLowPrio]

    if gb.steps_left < 50:
        if UniOre in ACTIONS:
            del ACTIONS[UniOre]

    if gb.steps_left < 25:
        if UniIce in ACTIONS:
            del ACTIONS[UniIce]
        if UniRubbleLowPrio in ACTIONS:
            del ACTIONS[UniRubbleLowPrio]

    plenty_power = unit.power > 0.95 * unit.battery_capacity

    if factory.power_hub_push and (
        (not factory.full_power_hub() or factory.available_power() < 1000)
        and not (unit.dies or unit.could_die)
    ):
        if verbose:
            lprint(
                f"Powerhub full?:{factory.full_power_hub()}, PWR AV: {factory.available_power()}"
            )
        if UniOre in ACTIONS:
            del ACTIONS[UniOre]

        if UniRubbleLowPrio in ACTIONS:
            del ACTIONS[UniRubbleLowPrio]

        if UniLichen in ACTIONS:
            del ACTIONS[UniLichen]

        if not plenty_power:
            if UniRubble in ACTIONS:
                del ACTIONS[UniRubble]

        if (
            gb.step < 200
            and (
                factory.n_heavies
                < (
                    factory.n_charger_positions
                    + factory.n_power_hub_positions // 2
                    + (1 if factory.ore_only else 2)
                )
            )
            and not plenty_power
        ):
            if UniRubbleTopPrio in ACTIONS:
                del ACTIONS[UniRubbleTopPrio]

    # if gb.steps_left < 15:
    #     if UniRubble in ACTIONS:
    #         del ACTIONS[UniRubble]

    queue = []
    counter = 0  # tie breaker
    for action_type, priority in ACTIONS.items():
        heapq.heappush(queue, (priority, counter, action_type))
        counter += 1

    if verbose:
        lprint(f"{unit}: {queue}")

    while queue:
        priority, _, action_type = heapq.heappop(queue)
        if verbose:
            lprint(f"LOOP: {unit}: {action_type} {priority}")
        # for action_type in actions:
        #     if action_type == UniOre and not dig_ore:
        #         continue

        #     if action_type is None:
        #         continue

        action = action_type(unit=unit)
        if action == UniOre and factory.is_ore_risk():
            action.allow_low_efficiency = True
        if action == UniIce and factory.is_ice_risk():
            action.allow_low_efficiency = True

        if action.execute():
            return True
    return False


def mining_heavy(unit: Unit, noaction=False, verbose=False):
    if unit.is_light:
        return False

    factory = unit.factory
    agent = unit.game_board.agent
    step = unit.game_board.step
    steps_left = unit.game_board.steps_left
    point = unit.point

    # heavies should not mine after 50 steps unless there is a need
    almost_out_of_water = factory.out_of_water_time() < 50
    if (
        step > 50
        and unit.is_heavy
        and len(factory.units) > 5
        and not almost_out_of_water
        and len(factory.ice_points) > 1
        and unit.power + factory.available_power() < unit.battery_capacity * 0.95
        and agent.enemy_lichen() <= 0
    ):
        if verbose:
            lprint(f"{unit}: heavy_mining: HEAVY early exit")
        return False

    ice_primary = factory.ice_points and (
        factory.cargo.water < 100 or unit.cargo.ice or len(factory.ice_points) == 1
    )

    dig_ore = (
        (steps_left > 100 or unit.cargo.ore > 0)
        and not almost_out_of_water
        and factory.cargo.metal < 200
        and len(factory.heavy_ore_hubs) == 0
    )
    dig_ice = (
        sum(factory.ore_in_time) > 0
        or factory.cargo.water < max(step, 100)
        or len(factory.units) == 1
        or len(factory.ore_points) == 0
        or almost_out_of_water
        or len(set(factory.ice_hubs) - set(factory.heavy_ice_hubs)) > 1
    )
    if verbose:
        lprint(
            f"heavy_mining: HEAVY {unit.id} ice_primary={ice_primary} dig_ore={dig_ore} dig_ice={dig_ice}"
        )

    # mine
    primary = Ice if ice_primary else Ore
    secondary = Ore if primary == Ice else Ice if dig_ice else Rubble
    tertiary = Rubble if secondary != Rubble else Ice

    dig_lichen = unit.dig_lichen()

    allow_low_efficiency = (
        noaction
        and point.factory
        and unit.available_power() >= 3000
        and factory.available_power() >= 1000
    )
    if verbose:
        lprint("allow_low_efficiency", unit, allow_low_efficiency)

    actions = [primary, secondary, tertiary]
    if dig_lichen:
        actions = [primary, secondary, UniLichen, tertiary]

    if HEAVY_ONLY_ICE:
        actions = [Ice]

    for action_type in actions:
        if unit.is_heavy and action_type == Rubble:
            continue

        if action_type == Ore and not dig_ore:
            continue

        if action_type is None:
            continue

        # if verbose:
        # lprint(f"heavy_mining: HEAVY {unit.id} action_type={action_type}")

        if allow_low_efficiency or (
            action_type == Ore
            and factory.is_ore_risk()
            and max(factory.metal_in_time) < 100
            and unit.available_power() > 2000
        ):
            uniop = (
                UniOre if action_type == Ore else UniIce if action_type == Ice else None
            )

            if uniop:
                # heavy needs many digs to be efficient, may not succeed in 50 ticks
                unit.tried_targets = []
                if uniop(unit=unit, allow_low_efficiency=True).execute():
                    return True
        else:
            action = action_type(unit=unit, allow_low_efficiency=allow_low_efficiency)

            if action.execute():
                return True

    return False


def rubble_raid(unit):
    min_power = unit.unit_cfg.BATTERY_CAPACITY * 0.5

    if (
        unit.power > min_power
        and len([p for p in unit.point.points_within_distance(12) if p.enemy_unit]) == 0
    ):
        rubble = RubbleRaid(unit=unit)
        if rubble.execute():
            return True
