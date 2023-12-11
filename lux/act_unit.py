from lux.actions import Attack
from lux.combat import (
    ambush_rss,
    handle_adj_attack_opportunities,
    handle_cargo_transport_intercepts,
    handle_chase_intercept,
    handle_factory_attack,
    handle_factory_defense,
    handle_last_minute_lichen,
    handle_lichen_attack,
    handle_lichen_defense,
    handle_low_power_intercept,
    handle_low_prio_combat,
    handle_rss_steal,
    handle_units_close_to_enemy,
    set_killnet,
    set_shield,
    monitor_surroundings,
    end_game,
)
from lux.idle import handle_idle, rebase_closest_enemy, set_power_hubs
from lux.mining import (
    bring_back_rss,
    check_replan_ore_cycle,
    mine_high_priority,
    mine_low_priority,
    set_chargers,
    handle_dieing_factory,
)
from lux.rebalance import handle_low_water, handle_no_heavy_factories, rebalance
from lux.unit import Unit
from lux.unplan import unplan_units
from lux.utils import lprint


def order_units(self):
    self.units_to_consider = sorted(
        self.units_to_consider,
        key=lambda u: (
            u.needs_escape,
            u.is_heavy,
            u.dies,
            u.could_die,
            u.must_move,
            u.try_replan,
            # begin with the bots with few options
            len([ap for ap in u.point.adjacent_points() if not ap.unit]),
            u.point.factory is not None,
        ),
        reverse=True,
    )
    self.units_to_be_replanned = sorted(
        self.units_to_be_replanned,
        key=lambda u: (
            u.is_heavy,
            u.must_retreat,
            u.dies,
            u.could_die,
            u.must_move,
            u.point.factory is not None,
        ),
        reverse=True,
    )


def unit_act(self):  # self: Agent
    init_workers(self)
    # return
    unplan_units(self)

    handle_dieing_factory(self)
    order_units(self)

    handle_last_minute_lichen(self, self.units.copy())

    # return
    set_chargers(self, self.units_to_be_replanned.copy())

    handle_adj_attack_opportunities(self.units)
    handle_no_heavy_factories(self, self.units_to_be_replanned.copy())

    # handle_chase_intercept(self, self.units_to_consider.copy())
    handle_cargo_transport_intercepts(self, self.units_to_consider.copy())
    handle_low_power_intercept(self, self.units_to_consider.copy())
    handle_low_water(self, self.units_to_be_replanned.copy())
    handle_factory_defense(self, self.units_to_consider.copy())

    if self.game_board.steps_left <= 25:
        if self.game_board.steps_left > 10:
            end_game_units = [u for u in self.units_to_consider if not u.is_charger]
        else:
            end_game_units = self.units
        end_game_units = [
            u
            for u in end_game_units
            if not u.is_digging and not u.self_destructs and not u.kills_next
        ]
        end_game(self, end_game_units)
    handle_factory_attack(self)

    bring_back_rss(self.units_to_be_replanned.copy())

    handle_lichen_defense(
        self, self.units_to_be_replanned.copy(), self.units_to_consider.copy()
    )

    handle_rss_steal(self, self.units_to_be_replanned.copy())
    monitor_surroundings(self)

    close_to_factory_replan = [
        u
        for u in self.units_to_consider
        if u.can_be_replanned(close_to_enemy_factory=True)
    ]
    ambush_rss(self, close_to_factory_replan.copy())

    rebalance(self, self.units_to_be_replanned.copy())
    mine_high_priority(self, self.units_to_be_replanned.copy(), icehubpass=True)

    # uses can_be_replanned close to enemy. need to check how this works
    handle_lichen_attack(self, self.units_to_consider.copy(), min_kills=15)
    handle_units_close_to_enemy(self, self.units_to_consider.copy())

    mine_high_priority(self, self.units_to_be_replanned.copy(), icehubpass=False)
    order_units(self)
    mine_low_priority(self, self.units_to_be_replanned.copy())

    check_replan_ore_cycle(self, self.units_to_consider.copy())

    handle_low_prio_combat(self, self.units_to_be_replanned.copy())

    # uses can_be_replanned close to enemy
    handle_lichen_attack(
        self, self.units_to_consider.copy(), min_kills=10, max_kills=15
    )

    set_power_hubs(self)

    close_to_factory_replan = [
        u
        for u in self.units_to_consider
        if u.can_be_replanned(close_to_enemy_factory=True)
    ]
    ambush_rss(self, close_to_factory_replan.copy(), no_actions=True)

    set_shield(self, self.units_to_be_replanned.copy())
    set_killnet(self, self.units_to_be_replanned.copy())

    handle_idle(self, self.units_to_be_replanned.copy())
    rebase_closest_enemy(self, self.units_to_be_replanned.copy())

    handle_to_die(self)

    for unit in self.units:
        if len(unit.action_queue) == 0:
            lprint(
                f"FIRST PASS {unit.game_board.step}: NO ACTION FOR {unit.unit_id}={unit.unit_type} at {unit.point} with power {unit.power}"
            )
            if [
                u for u in self.game_board.unit_grid_in_time[1][unit.point] if u != unit
            ]:
                unit.must_move = True

    # retry all units that were unplanned in handle_to_die
    bring_back_rss(self.units_to_be_replanned.copy())

    mine_low_priority(self, self.units_to_be_replanned.copy(), noaction=True)

    grid = self.game_board.unit_grid_in_time[1]
    for unit in self.units_to_be_replanned:
        if len([u for u in grid[unit.point] if u != unit]) == 0 and not unit.could_die:
            continue
            # lprint(
            #     "UNIT DOES NOT HAVE TO MOVE",
            #     unit,
            #     unit.next_point,
            #     unit.path,
            #     unit.action_queue,
            #     [u for u in grid[unit.point] if u != unit],
            #     unit.unit_id in self.actions,
            # )
        if unit.dies or unit.must_move or unit.could_die:
            unit.tried_targets = []
            unit.accept_attack_risk = True

    at_risk_units = [u for u in self.units_to_be_replanned if u.accept_attack_risk]
    lprint(f"AT RISK UNITS: {at_risk_units}")
    mine_low_priority(self, at_risk_units.copy())

    at_risk_units = [u for u in self.units_to_be_replanned if u.accept_attack_risk]
    handle_idle(self, at_risk_units.copy(), at_risk=True)

    for unit in self.units:
        if len(unit.action_queue) == 0:
            lprint(
                f"END: {unit.game_board.step}: NO ACTION FOR {unit.unit_id}={unit.unit_type} at {unit.point} with power {unit.power}"
            )


def init_workers(self):
    self.units_to_be_replanned = []
    self.units_to_consider = []

    for u in self.units:
        if u.power == 0:
            continue
        consider = not (u.is_attacking or u.is_defending)
        if consider:
            self.units_to_consider.append(u)
        if u.can_be_replanned():
            self.units_to_be_replanned.append(u)


def handle_to_die(self):
    gb = self.game_board
    next_turn_grid = gb.unit_grid_in_time[1]
    attack_risk_grid = gb.attack_risk_in_time[1]
    unit: Unit
    for unit in self.units:
        if unit.dies:
            next_point = unit.next_point if unit.next_point else unit.point
            units_at_next_point = next_turn_grid[next_point]
            # assert (
            #     IS_KAGGLE or unit.next_point is None or unit in units_at_next_point
            # ), f"{unit} not at next point in the grid????"
            if len(units_at_next_point) == 0:
                lprint(
                    f"{unit} UNIT NO LONGER DIES???, unplanned: {unit.next_point is None}"
                )

            own_units = [u for u in units_at_next_point if u.is_own and u != unit]
            enemy_units = [u for u in units_at_next_point if u.is_enemy]
            attack_risk = attack_risk_grid[next_point]
            if own_units:
                for other in set(own_units):
                    if len(next_turn_grid[other.point]) == 0 and (
                        attack_risk
                        > unit.combat_strength(has_moved=False, power_left=unit.power)
                    ):
                        other.unplan("Kills own unit and position is free")
                    else:
                        lprint(
                            f"UNIT {other.unit_id} WILL KILL {unit.unit_id} at {other.next_point}"
                        )
                        other.add_to_work_queue()
                continue

            if enemy_units:
                # attack the unit that will kill us
                enemy_unit = enemy_units[0]
                if Attack(
                    unit=unit, targets=[enemy_unit.point], target_bots=[enemy_unit]
                ).execute():
                    continue
                else:
                    lprint(
                        f"$$$$$$$$$ UNIT {unit.unit_id} WILL DIE at {unit.next_point} by enemy units {enemy_units}"
                    )
            # lprint(
            #     f"handle_to_die {unit}, attack risk: {attack_risk}, next_point{next_point}, {unit.path} {unit.dies} {unit.could_die}"
            # )
            if attack_risk > unit.combat_strength(
                has_moved=False, power_left=unit.power
            ):
                step_in_enemies = [
                    p.unit
                    for p in next_point.adjacent_points()
                    if p.unit
                    and p.unit.is_enemy
                    and (p.unit <= unit or p.unit.is_moving_next)
                ]
                if step_in_enemies:
                    enemy_unit = sorted(
                        step_in_enemies,
                        key=lambda u: (
                            u > unit,
                            u.is_moving_next,
                            -u.total_cargo,
                            u.power,
                        ),
                    )[0]
                    if Attack(
                        unit=unit, targets=[enemy_unit.point], target_bots=[enemy_unit]
                    ).execute():
                        continue
                    else:
                        lprint(
                            f"$$$$$$$$$$ UNIT {unit.unit_id} COULD DIE at {unit.next_point} by attack risk {attack_risk}"
                        )

            for ap in unit.point.adjacent_points():
                units_at_adj_point = next_turn_grid[ap]
                own_units = [u for u in units_at_adj_point if u.is_own and u != unit]
                enemy_units = [u for u in units_at_adj_point if u.is_enemy]
                if enemy_units:
                    continue

                if len(own_units) == 1:
                    other = own_units[0]
                    if other.is_heavy or other.power < 4 * other.base_move_cost:
                        continue
                    if (
                        attack_risk_grid[other.point] == 0
                        and len(next_turn_grid[other.point]) == 0
                    ):
                        other.unplan("Need to save own unit and position is free")
                    else:
                        if any(
                            [
                                attack_risk_grid[ap2] == 0
                                and len(next_turn_grid[ap2]) == 0
                                for ap2 in ap.adjacent_points()
                                if ap2 != ap
                            ]
                        ):
                            other.unplan(
                                "Need to save own unit and position next to it is free"
                            )
