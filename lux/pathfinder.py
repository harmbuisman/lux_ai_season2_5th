# need to build maps for positions of factories, ice, and units

# a[0] = action type
# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)

# a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)

# a[3] = X, amount of resources transferred or picked up if action is transfer or pickup.
# If action is recharge, it is how much energy to store before executing the next action in queue

# a[4] = 0,1 - repeat false or repeat true. If true, action is sent to end of queue once consumed

# mining/attacking:
# a[0] 0, 3, 5
# a[1] 0,1,2,3,4
# a[3] X
# a[4] repeat or not

import sys
import time
from collections import defaultdict
from typing import List
import numpy as np
from lux.board import GameBoard, Point
from lux.candidate import RESOURCE_MAP, Candidate
from lux.constants import DIRECTION_MAP, IS_KAGGLE, TMAX, TTMAX, VALIDATE, ACTION, STATE
from lux.unit import Unit
from lux.utils import lprint
from lux.router import get_optimal_directions_dict


class Pathfinder:
    def __init__(
        self,
        agent,
        unit: Unit,
        goal,
        destinations: List[Point],
        targets: List[Point] = None,
        goal_amount: int = None,
        max_it: int = TMAX,
        target_bots: List[Unit] = None,
        max_steps_off_target: int = 0,
        max_waits: int = 2,
        verbose: bool = False,
    ):
        ts = time.time()
        self.agent = agent
        self.game_state = agent.game_state
        gb: GameBoard = agent.game_board
        self.gb = gb
        self.steps_left = gb.steps_left
        self.player = agent.player
        self.goal = goal
        self.target_bots = [] if target_bots is None else target_bots
        self.scores_per_rubble = {}
        self.max_steps_off_target = max_steps_off_target

        self.max_waits = 0 if unit.accept_attack_risk else max_waits

        self.charge_rate = unit.unit_cfg.CHARGE * gb.charge_rate
        self.move_cost = unit.unit_cfg.MOVE_COST
        self.charge_rates = {}
        self.move_costs = {}
        self.targets = tuple() if targets is None else tuple(targets)
        self.p2p_power = agent.p2p_power_cost[unit.unit_type]

        # caching vars
        self.min_move_power_to_destinations = {}
        self.charge_in_time = {}
        self.min_distance_to_destinations = {}
        self.min_distance_to_targets = {}

        if destinations is not None:
            self.destination_ids = np.array([d.id for d in destinations])

        self.comm_cost = unit.unit_cfg.ACTION_QUEUE_POWER_COST

        self.goal_amount = (
            goal_amount if goal_amount is not None else unit.unit_cfg.CARGO_SPACE
        )
        self.goal_amount = int(min(self.goal_amount, unit.unit_cfg.CARGO_SPACE))
        assert (
            IS_KAGGLE or self.goal_amount > 0
        ), f"{unit} {goal} goal_amount={self.goal_amount} should be positive"

        self.start = self.gb.get_point(unit.pos)
        self.unit = unit
        self.unit_grid_in_time = gb.unit_grid_in_time
        self.attack_risk_in_time = gb.attack_risk_in_time

        self.unit_grid = self.unit_grid_in_time[0]
        self.max_it = min(max_it, gb.steps_left)
        self.risk_grid = self.attack_risk_in_time[0]
        self.battery_capacity = unit.unit_cfg.BATTERY_CAPACITY
        self.destinations = tuple(destinations) if destinations else tuple()

        self.t = 0
        self.any_on_goal_action_time = False
        self.dig_cost = unit.unit_cfg.DIG_COST
        root = Candidate(
            direction=None,
            pathfinder=self,
            start=self.start,
            verbose=not IS_KAGGLE or verbose,
        )
        self.root = root

        msg = f"PF for {unit.unit_id} t={gb.step}: at {root.point} goal={goal.name()} start_charge={goal.start_charge} power={unit.power} max_it={self.max_it} targets:{targets} dest:{destinations}"
        lprint(msg)

        if not root.is_valid:
            lprint(f"PF for {unit.unit_id} t={gb.step}: invalid root")
            self.completed = []
            return

        self.init_paths()

        # don't do optimal search for nearby targets, can lead to suboptimal paths as it could choose the furthest target
        skip_optimal = False
        if (
            self.targets
            and not goal.needs_to_return
            and min([target.distance_to(unit.point) for target in targets]) < 4
        ):
            skip_optimal = True

        if skip_optimal:
            self.completed = []
        else:
            self.is_optimal = True
            self.completed = self.run_search(optimal=True, verbose=verbose)
        if not self.completed:
            if not skip_optimal:
                te = time.time()
                duration = te - ts
                lprint(f"Pathfinder optimal took {duration:.1f} sec")

            self.is_optimal = False
            self.completed = self.run_search(optimal=False, verbose=verbose)

        if not self.completed and goal.plan_even_if_not_complete:
            self.completed = goal.get_best_uncompleted_candidates(self.candidates)

        te = time.time()
        duration = te - ts
        lprint(f"Pathfinder total took {duration:.1f} sec")
        # assert IS_KAGGLE or duration < 0.3

        if self.completed:
            pass
            # lprint("# completed", len(completed))
            # for c in self.completed:
            #     lprint(c)
        else:
            lprint(
                f"{unit.unit_id}: NO PATH FOUND!!!, t={gb.step} from {unit.point.xy}, iteration={self.t}",
                file=sys.stderr,
            )
            # for c in candidates:
            #     lprint(c)

    def run_search(self, optimal, verbose=False):
        lprint(f"Running search optimal={optimal}")
        goal = self.goal
        its = self.max_it + 1
        completed = []
        best_completed_score = -100000000000
        best_completed_power = -1000
        candidates = [self.root]

        self.t = 1
        global_pointdir = defaultdict(tuple)

        # in case of sentry actions where already on the spot
        if self.root.state == STATE.COMPLETE:
            return [self.root]

        if (
            self.root.state == STATE.ON_TARGET
            and self.root.point.factory is None
            and len(self.targets) > 1
        ):
            lprint(
                f"root is on target, but there are more targets, so we need to go to the next target"
            )
            root_no_target = Candidate(
                direction=None,
                pathfinder=self,
                start=self.start,
            )
            root_no_target.state = STATE.PREPARED
            candidates.append(root_no_target)

        self.candidates = []
        while its and candidates and self.t <= self.max_it:  # and not completed:
            t = self.t
            if t > self.gb.steps_left:
                break

            self.charge_in_time = {}

            goal.update_targets(self)

            self.dig_cost = self.unit.unit_cfg.DIG_COST
            self.unit_grid = self.unit_grid_in_time[self.t]
            self.risk_grid = self.attack_risk_in_time[t]

            pointdir_to_candidates = defaultdict(list)

            for candidate in candidates:
                dirs = goal.get_next_actions(candidate, optimal=optimal)
                # lprint(f"candidate {candidate} dirs={dirs}")
                if dirs is None:
                    lprint(f"dirs is None {candidate}")

                for d in set(dirs):
                    c = Candidate(
                        direction=d,
                        pathfinder=self,
                        parent=candidate,
                        verbose=verbose,
                    )

                    if goal.start_charge:
                        if t > 2 and ACTION.CHARGE not in c.commands:
                            continue

                    # if c.point.xy == (41, 22):
                    #     lprint(
                    #         f">>>>>>>>. t={t} {c.state.name}, {c.score} {c.point} {c.on_goal_action} {c.commands} {c.path}"
                    #         f"{c.is_valid}"
                    #     )

                    if not c.is_valid:
                        continue

                    # early exit search if another candidate has already reached the goal
                    if (
                        not c.on_goal_action
                        and self.any_on_goal_action_time
                        and t >= self.any_on_goal_action_time + 3
                    ):
                        continue

                    # power is score
                    # point system, same spot same time are competing candidates
                    # if d not in off_dirs:
                    #     c.steps_off_target_left -= 1

                    # can discard any candidates that did not complete goal and have less power
                    # and less accumulated score (e.g. due to digging rubble)
                    pointdir = c.point
                    pointdir_to_candidates[pointdir].append(c)

            candidates_new = []

            for pointdir, candidates in pointdir_to_candidates.items():
                best_value = (-1000000000, -100000000)
                best_candidate = None

                for c in candidates:
                    score = c.score
                    power = c.power

                    value = (score, power)

                    if value > best_value:
                        best_value = value
                        best_candidate = c

                candidates_new.append(best_candidate)
                global_pointdir[c.point] = best_value

            candidates = []
            for cn in candidates_new:
                self.any_on_goal_action_time = self.any_on_goal_action_time or (
                    t if cn.on_goal_action > 0 else False
                )

                if cn.state == STATE.COMPLETE:
                    completed.append(cn)
                    if cn.score > best_completed_score:
                        best_completed_score = cn.score
                        best_completed_power = cn.power

                else:
                    # if worse than competed so far, dont bother
                    if completed:
                        if cn.score < best_completed_score:
                            continue

                        # if no on goal actions but also less power than any candidate (so no optimal route)
                        # then drop the candidate
                        if (
                            candidate.on_goal_action == 0
                            and cn.power < best_completed_power
                        ):
                            continue

                    candidates.append(cn)

            self.t += 1
            its -= 1

            is_break = False
            if goal.stop_search_if_complete and len(completed) > 0:
                is_break = True
            else:
                candidates = goal.prune_candidates(candidates)

            self.candidates = candidates

            # lprint(
            #     f"PFIT {self.unit.unit_id} t={self.t - 1} candidates={len(candidates)} completed={len(completed)}"
            # )

            # for c in candidates:
            #     if c.point.xy != (41, 22):
            #         continue
            #     lprint(
            #         f"t={t} {c.state.name}, {c.score} {c.point} {c.on_goal_action} {c.commands} {c.path}"
            #     )

            if is_break:
                break
        return completed

    def init_paths(self):
        if self.targets:
            starts = (
                self.unit.factory.points
                if self.goal.start_charge
                else [self.unit.point]
            )
            self.to_target_directions = get_optimal_directions_dict(
                self.unit, starts=starts, ends=self.targets
            )
        else:
            self.to_target_directions = defaultdict(list)

        if self.destinations and self.goal.needs_to_return:
            starts = [self.unit.point] if not self.targets else self.targets
            self.to_destination_directions = get_optimal_directions_dict(
                self.unit, starts=starts, ends=self.destinations
            )
        else:
            self.to_destination_directions = defaultdict(list)

    def get_charge_power(self, start: int, dt: int):
        """Get the charge power for a given time interval (day, night)"""
        if (start, dt) in self.charge_rates:
            return self.charge_rates[(start, dt)]

        charge_power = sum(self.charge_rate[start : start + dt])  # noqa
        self.charge_rates[(start, dt)] = charge_power
        return charge_power

    def plan(self, actions: dict, debug: bool = False):
        """Plan the best found action for the unit"""
        if not self.completed:
            return actions

        ledger = self.agent.ledger
        goal = self.goal
        unit: Unit = self.unit

        if not debug and not unit.can_be_replanned():
            unit.unplan(f"planning new action: {goal}")

        min_score = min(c.score for c in self.completed)
        max_score = max(c.score for c in self.completed)

        correction = 0
        if max_score <= 0:
            correction = -min_score + 1

        completed = sorted(
            self.completed,
            key=lambda c: (-(c.score + correction) / len(c.path), -c.power),
        )

        best = completed[0]

        action_queue = []

        prev_d = None
        prev_amount = None

        wasted_power = best.power_wasted

        for i, d in enumerate(best.commands):
            t = i + 1  # -1 because first action is executed at t=1

            p = best.path[i]
            if unit not in p.visited_by:
                p.visited_by.append(unit)

            if d == ACTION.DIG:
                if d == prev_d:
                    action_queue[-1][-1] += 1
                else:
                    action_queue.append(unit.dig())
                dig_amount = unit.unit_cfg.DIG_RUBBLE_REMOVED
                if not debug:
                    p.update_rubble_in_time(t, dig_amount)
                    if unit not in p.dug_by:
                        p.dug_by.append(unit)
            elif d == ACTION.SELF_DESTRUCT:
                action_queue.append(unit.self_destruct())
                p.dug_by.append(unit)
            elif d == ACTION.PICKUP:
                amount = int(self.goal_amount)
                action_queue.append(unit.pickup(goal.resource_id, amount))
                if not debug:
                    p.factory.update_resource(t, goal.resource_id, -amount)
            elif d == ACTION.CHARGE:
                amount = int(best.pickups[t])

                if wasted_power > 0:
                    overflow = min(amount, wasted_power)
                    lprint(
                        f"{unit} ADJUSTING FOR {overflow} WASTED POWER!!!!!!!!!!!!!!!"
                    )
                    wasted_power -= overflow
                    amount -= overflow

                # if d == prev_d and amount == prev_amount:
                #     action_queue[-1][-1] += 1
                # else:
                action_queue.append(unit.pickup(4, amount))
                if not debug:
                    p.factory.transfer_power(t, amount)
                # prev_amount = amount
            elif d == ACTION.TRANSFER:
                # todo transfer can go to next cell
                # can build pipeline of lights?
                if goal.resource_id == RESOURCE_MAP["ice"]:
                    amount = best.ice_before_transfer
                    rss = RESOURCE_MAP["ice"]
                elif goal.resource_id == RESOURCE_MAP["ore"]:
                    amount = best.ore_before_transfer
                    rss = RESOURCE_MAP["ore"]
                elif goal.name() == "water":
                    amount = self.goal_amount
                    rss = RESOURCE_MAP["water"]
                elif goal.name() == "metal":
                    amount = self.goal_amount
                    rss = RESOURCE_MAP["metal"]
                else:
                    if unit.cargo.water > 0:
                        amount = unit.cargo.water
                        rss = RESOURCE_MAP["water"]
                    elif unit.cargo.metal > 0:
                        amount = unit.cargo.metal
                        rss = RESOURCE_MAP["metal"]
                    elif best.ice_before_transfer and prev_d != ACTION.TRANSFER:
                        amount = best.ice_before_transfer
                        rss = RESOURCE_MAP["ice"]
                    elif best.ore_before_transfer:
                        amount = best.ore_before_transfer
                        rss = RESOURCE_MAP["ore"]

                action_queue.append(unit.transfer(0, rss, amount))
                if not debug:
                    p.factory.update_resource(t, rss, amount)
            else:
                if d == prev_d:
                    action_queue[-1][-1] += 1
                else:
                    action_queue.append(unit.move(int(d)))

            prev_d = d

        if best.path:
            p = best.path[-1]
            if unit not in p.visited_by:
                p.visited_by.append(unit)

        assert (
            IS_KAGGLE or len(action_queue) == best.queue_len
        ), f"action queue mismathc {len(action_queue)} != {best.queue_len}, {action_queue}"

        # after an attack, just wait and recharge if it had a repeat action in the queue
        action_queue = goal.post_process_action_queue(best, action_queue)

        lprint(f"{unit.game_board.step}: ROUTE", best, action_queue)
        # lprint(
        #     [f"{i} ({i + self.gb.step}) {p}" for i, p in enumerate(best.powers)],
        # )
        # lprint(
        #     [f"{i} ({i + self.gb.step}) {p}" for i, p in enumerate(best.path)],
        # )

        # update unit grid in time
        unit_id = unit.unit_id
        point = unit.point
        max_len = self.gb.env_cfg.UNIT_ACTION_QUEUE_SIZE

        if len(action_queue) > max_len:
            lprint(
                "actionqueue too long!!!, truncating",
                goal,
                len(action_queue),
                ">",
                max_len,
                file=sys.stderr,
            )

            assert (
                IS_KAGGLE or False
            ), f"action queue too long {len(action_queue)}>{max_len}"
            action_queue = action_queue[:max_len]

        # if debug:
        #     lprint("POWER IN TIME")
        #     for i, p in enumerate(best.path):
        #         lprint(f"{i} ({i + self.gb.step}) {p}: {best.powers[i]}")
        #     # lprint([(i, p, best.powers[i]) for i, p in enumerate(best.path)])

        #     lprint("NOT UPDATIN UNIT GRID, DEBUG MODE")
        #     return

        # remove from lux.current point
        for i in range(TTMAX):
            self.unit_grid_in_time[i][point] = [
                u for u in self.unit_grid_in_time[i][point] if u != unit
            ]

        # add paths
        if VALIDATE:
            if unit_id in ledger:
                del ledger[unit_id]

        path_points = best.path.copy()
        if action_queue[-1][-2] > 0:
            path_points = path_points + [path_points[-1]] * (TTMAX - len(path_points))

        max_episode_length = self.gb.env_cfg.max_episode_length
        for i, p in enumerate(path_points):
            t = self.gb.step + i
            if t >= max_episode_length:
                break

            self.unit_grid_in_time[i][p].append(unit)

            if VALIDATE and i < len(best.powers):
                ledger[unit_id][t] = {
                    "point": p,
                    "power": best.powers[i],
                    "ice": best.ices[i],
                    "ore": best.ores[i],
                    "action": (best.commands + ["END"])[i],
                }

        n_steps = len(best.powers)
        unit.power_in_time[:n_steps] = best.powers
        unit.power_in_time[n_steps:] = (
            best.powers[-1] + self.gb.charge_rate[n_steps:] * unit.unit_cfg.CHARGE
        )

        actions[unit_id] = action_queue
        unit.action_queue = action_queue
        unit.last_point = best.point
        unit.try_replan = False
        unit.path = best.path
        unit.next_point = best.path[1] if len(best.path) > 1 else best.point

        unit.is_replanned = True

        unit.dies = False
        unit.could_die = False

        unit.remove_from_work_queue()

        for u in set(best.routed_units):
            if unit not in u.routed_by:
                u.routed_by.append(unit)
            u.unplan(f"routed by priority action from unit {unit}")
            u.dies = True

            if u.planned_action:
                try:
                    u.planned_action.execute()
                except Exception as e:
                    assert IS_KAGGLE or False, f"error executing planned action {e}"
            elif u.was_power_hub:
                lprint(f"TRY replan power hub {u}")
                u.factory.replan_power_hub(u)

        next_point_unit = unit.next_point.unit
        if next_point_unit and next_point_unit.is_own and next_point_unit != unit:
            if (
                next_point_unit.next_point is None
                or next_point_unit.next_point == unit.next_point
            ):
                next_point_unit.dies = True
                next_point_unit.dies_by_own = False
                next_point_unit.must_move = True

        if not debug:
            goal.on_plan(best)
            unit.planned_action = goal

        return actions
