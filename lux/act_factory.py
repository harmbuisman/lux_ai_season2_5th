from lux.constants import (
    BUILD_HEAVY,
    BUILD_LIGHT,
    LIGHT_TO_HEAVY_RATIO,
    MAX_UNIT_PER_FACTORY,
    MIN_LIGHT_PER_FACTORY,
)
from lux.utils import lprint


def factory_act(self):  # self: Agent
    # FACTORY ACTIONS

    build_robots(self)
    water(self)
    # force_water(self)


def force_water(self):
    actions = self.actions
    for factory in self.factories:
        water_cost = factory.water_cost()
        if water_cost > 0:
            lprint(
                f"{factory.unit_id}: av: {factory.available_water()}, watering lichen with {water_cost} water #growth tiles: {factory.n_growth_tiles}"
            )
            actions[factory.unit_id] = factory.water()


def water(self):
    actions = self.actions
    game_board = self.game_board
    # if game_board.steps_left > WATER_PUSH_TIME:
    #     return

    steps_left = game_board.steps_left
    # for now don't water first half of the game
    for factory in self.factories:
        # building a bot has priority
        if factory.unit_id in actions:
            continue

        if factory.water_cost() == 0:
            continue

        if (
            factory.min_lichen_value == 100
            and factory.n_growth_tiles == factory.n_connected_tiles
            and steps_left > 1
        ):
            return False

        if factory.water_for_power():
            lprint(
                f"{factory.unit_id}: av: {factory.available_water()}, watering "
                f"lichen with {factory.water_cost()} of water #growth tiles: {factory.n_growth_tiles}"
            )
            actions[factory.unit_id] = factory.water()
            continue

        # quick and dirty for now
        water_cost = factory.water_cost()

        if water_cost * (steps_left + 1) < factory.cargo.water:
            lprint(
                f"{factory.unit_id}: av: {factory.available_water()}, watering lichen with {water_cost} water #growth tiles: {factory.n_growth_tiles}"
            )
            actions[factory.unit_id] = factory.water()
            continue

        if steps_left < 25:
            if factory.cargo.water - water_cost > steps_left:
                lprint(
                    f"{factory.unit_id}: av: {factory.available_water()}, watering lichen with {water_cost} water #growth tiles: {factory.n_growth_tiles}"
                )
                actions[factory.unit_id] = factory.water()
                continue

        # if factory.has_enough_water_to("end"):
        #     water_cost = factory.water_cost()
        #     if (
        #         water_cost < factory.cargo.water
        #         and (factory.cargo.water - water_cost) > steps_left
        #     ):
        #         lprint(
        #             f"{factory.unit_id}: av: {factory.available_water()}, watering lichen with {water_cost} water #growth tiles: {factory.n_growth_tiles}",
        #
        #         )
        #         actions[factory.unit_id] = factory.water()


def build_robots(self, verbose=False):  # self: Agent
    if self.game_board.step >= 975:
        return
    actions = self.actions

    # do not build robots during cold snap due to  double costs
    game_state = self.game_state

    n_units = len(self.units)
    n_factories = len(self.factories)
    heavy_cfg = self.env_cfg.ROBOTS["HEAVY"]
    light_cfg = self.env_cfg.ROBOTS["LIGHT"]

    low_metal_factories = [
        f for f in self.factories if max(f.metal_in_time) < 100 and f.is_ore_risk()
    ]

    for factory in self.factories:
        if not factory.can_spawn(self.game_board):
            continue

        available_power = factory.available_power()

        if (
            BUILD_HEAVY
            and available_power
            >= heavy_cfg.POWER_COST
            * (1 if self.game_board.step < 250 or factory.power_hub_push else 2)
            and factory.available_metal() >= heavy_cfg.METAL_COST
            and (
                game_state.real_env_steps > 10
                or factory.available_metal() < 120
                or factory.power_hub_push
                and factory.n_lights > 0
            )
            and (
                self.opponent_heavy_bot_potential > 0
                or (2 * n_units / n_factories > MAX_UNIT_PER_FACTORY)
            )
        ):
            if factory.power_hub_push and factory.full_power_hub():
                if low_metal_factories and factory.available_metal() < 150:
                    continue
                lprint(
                    f"{factory.unit_id}: power_hub_push, build HEAVY, n_units: {n_units}, n_factories: {n_factories}"
                )
            actions[factory.unit_id] = factory.build_heavy()
            n_units += 1
            lprint(f"{factory.unit_id}: build HEAVY")
        elif BUILD_LIGHT and should_build_light(self, factory, verbose=verbose):
            actions[factory.unit_id] = factory.build_light()
            n_units += 1
            lprint(f"{factory.unit_id}: build light")


def should_build_light(self, factory, verbose=False):
    available_power = factory.available_power()
    light_cfg = self.env_cfg.ROBOTS["LIGHT"]

    if factory.cargo.metal < light_cfg.METAL_COST:
        if verbose:
            lprint(
                f"{factory.unit_id}: not enough metal to build light, cargo.metal: {factory.cargo.metal}",
            )
        return False

    # only build a light if we have enough power to fully charge it
    if available_power < light_cfg.BATTERY_CAPACITY:  # light_cfg.POWER_COST:
        if verbose:
            lprint(
                f"{factory.unit_id}: not enough power to build light, available_power: {available_power}",
            )
        return False

    n_power_hub_positions = factory.n_power_hub_positions
    more_power_hub_potential = n_power_hub_positions > len(
        [u.is_power_hub for u in factory.heavies]
    )

    n_lights_needed = (len(factory.ore_hubs) > 0) + (len(factory.ice_hubs) > 0)
    n_free_inner_hub_positions = len(factory.get_power_hub_positions(factory_only=True))
    n_active_hubs = len([u for u in factory.heavies if u.is_power_hub])

    if factory.power_hub_push and not factory.full_power_hub():
        if verbose:
            lprint(
                f"{factory.unit_id}: power_hub_push requires more heavies?",
                factory.n_lights,
                factory.n_heavies,
                n_lights_needed,
            )
        if factory.n_lights < n_lights_needed:
            pass
        elif (
            factory.n_lights == factory.n_heavies
            and factory.power > 1000
            and n_free_inner_hub_positions <= n_active_hubs
        ):
            pass
        else:  # factory.n_lights >= factory.n_heavies and factory.power < 800:
            if verbose:
                lprint(
                    f"{factory.unit_id}: power_hub_push requires more heavies or more power first",
                )
            return False

    # # push heavies!
    # if (
    #     factory.ore_only
    #     and (self.n_lights / len(self.factories) > 4 and more_power_hub_potential)
    #     or (self.n_lights / len(self.factories) > 10)
    #     # and (
    #     #     len(
    #     #         [
    #     #             p
    #     #             for p in factory.ore_points
    #     #             if p.distance_to(factory) <= 5 and p.closest_factory == factory
    #     #         ]
    #     #     )
    #     #     < 4
    #     # )
    # ):
    #     return False

    if (
        factory.heavy_ore_hubs
        and factory.n_lights > 4
        and factory.n_heavies <= 1
        and factory.n_all_lichen_tiles == 0
    ):
        if verbose:
            lprint(
                f"{factory.unit_id}: not building light because ore_hubs and n_lights > 4 and n_heavies <= 0",
            )
        return False

    # ore_close = (
    #     min([factory.distance_to(p) for p in factory.ore_points]) < 4
    #     if factory.ore_points
    #     else False
    # )

    if (
        # (ore_close or max(factory.metal_in_time) >= 100)
        # and
        factory.n_heavies > 0
        and (factory.n_lights_exclude / factory.n_heavies) > LIGHT_TO_HEAVY_RATIO
    ):
        units_in_range = [
            p.unit for p in factory.center.points_within_distance(12) if p.unit
        ]
        my_units_in_range = [u for u in units_in_range if u.is_own and u.is_light]
        enemy_units_in_range = [u for u in units_in_range if u.is_enemy and u.is_light]

        if len(my_units_in_range) * 1.5 > len(enemy_units_in_range):
            if verbose:
                lprint(
                    f"{factory.unit_id}: too many lights compared to heavies, n_lights_exclude"
                    f": {factory.n_lights_exclude}, n_heavies: {factory.n_heavies}",
                )
            return False

    if factory.is_crowded(verbose=verbose):
        if verbose:
            lprint(f"{factory.unit_id}: factory is crowded")
        return False

    if len(factory.units) >= MAX_UNIT_PER_FACTORY:
        if verbose:
            lprint(
                f"{factory.unit_id}: too many units at factory ({len(factory.units)}))>={MAX_UNIT_PER_FACTORY}"
            )
        return False

    # if len(self.units) >= MAX_UNIT_PER_FACTORY * len(self.factories):
    #     if verbose:
    #         lprint(f"{factory.unit_id}: too many units")
    #     return False

    if factory.n_lights_exclude >= 1.5 * max(
        len(factory.ore_points + factory.ice_points), 10
    ):
        if verbose:
            lprint(
                f"{factory.unit_id}: too many lights compared to number of resource tiles",
            )
        return False

    n_factories = len(self.factories)
    n_lights = len(
        [
            u
            for u in self.units
            if u.is_light
            and not u.is_shield
            and u.point.closest_own_factory_distance < 15
        ]
    )
    n_opp_lights = len([u for u in self.opponent_units if u.is_light])

    if n_lights / n_factories > MIN_LIGHT_PER_FACTORY:
        if (
            n_lights > n_factories * MAX_UNIT_PER_FACTORY
            and len(self.units) > len(self.opponent_units) * 3
        ) or (
            (len([u for u in self.units if u.is_light]) > len(self.opponent_units) * 4)
            and n_opp_lights > 3
        ):
            if verbose:
                lprint(f"{factory.unit_id}: too many units")
            return False

    n_heavies = len([u for u in self.units if u.is_heavy])
    n_opp_heavies = len([u for u in self.opponent_units if u.is_heavy])

    min_light_no_build_threshold = 10 * len(self.game_board.factories)
    more_light_than_total = n_lights > max(
        len(self.opponent_units), min_light_no_build_threshold
    )
    more_light_than_total_stricter = n_lights > max(
        len(self.opponent_units) + 2 * n_factories, min_light_no_build_threshold
    )
    enemy_more_heavies = (
        n_heavies < n_opp_heavies and n_lights > min_light_no_build_threshold
    )

    # lprint(
    #     factory,
    #     more_light_than_total,
    #     more_light_than_total_stricter,
    #     enemy_more_heavies,
    #     n_lights,
    #     len(self.opponent_units),
    #     len(self.opponent_units) + 2 * n_factories,
    #     max(factory.metal_in_time),
    #     factory.available_metal(),
    # )
    if self.game_board.step > 25:
        if factory.n_lights_exclude < MIN_LIGHT_PER_FACTORY:
            return True
        if (
            (
                (more_light_than_total or enemy_more_heavies)
                and factory.available_metal() >= 100
            )
            or (
                (more_light_than_total_stricter or enemy_more_heavies)
                and max(factory.metal_in_time) >= 100
            )
            and factory.lake_large_enough()
        ):
            # prefer heavy bot construction
            if verbose:
                lprint(f"{factory.unit_id}: wait for heavy bot")

            return False
    return True
