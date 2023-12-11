from lux.point import Point
from lux.utils import lprint
from lux.constants import BASE_COMBAT_HEAVY

# def is_triple_visit(candidate, verbose=False):
#     if candidate.pathfinder.t < 3:
#         return False

#     if (
#         parent_point != point
#         and grandparent_point == point
#         and great_grandparent_point == parent_point
#     ):
#         if verbose:
#             lprint(f"{candidate.pathfinder.t}: triple visit {point}")
#         return True


def is_enemy_factory(candidate, verbose=False):
    point = candidate.point
    if point.factory and point.factory.is_enemy:
        if verbose:
            lprint(
                f"enemy factory != {point.factory.team_id}",
                point,
                "at time",
                candidate.pathfinder.t,
            )
        return True


def is_attack_risk(candidate, verbose=False):
    pf = candidate.pathfinder
    unit = pf.unit
    if unit.accept_attack_risk:
        return False

    point = candidate.point

    if point.enemy_lichen:
        if unit.is_light:
            if pf.agent.unit_dominance and pf.goal.ignore_attack_risk_on_unit_dominance:
                return False

        if point.lichen > pf.gb.steps_left and pf.goal.allow_death_on_lichen:
            return False

    t = pf.t

    risk = pf.risk_grid[point]

    high_risk = risk >= candidate.own_combat_strength
    if not high_risk and t > 0:
        if risk > 0 and pf.goal.disturb_is_good:
            candidate.score += 0.1

    if t > 0 and high_risk:
        # # special case for next turn
        # # opponent believes unit would be at
        # if t <= 1 and (
        #     (unit.is_light and risk >= BASE_COMBAT_HEAVY)
        #     or (
        #         unit.is_light
        #         and point.lichen > 0
        #         and point.lichen_strains not in pf.agent.own_lichen_strains
        #     )
        # ):
        #     cpoint = unit.next_point_observation
        #     if cpoint != point and point != pf.unit.point:
        #         # note bot being on the spot is handled in a different check
        #         # enemy may think I am not here
        #         return False

        if verbose:
            lprint(
                f"{t}: Moving onto ({point}) that is under attack risk of enemy {pf.risk_grid[point]}"
                f" while bot has {candidate.own_combat_strength}, direction:{candidate.commands[-1]}",
            )
        return True


def is_attack_risk_no_queue(candidate, verbose=False):
    point = candidate.point
    if (
        candidate.pathfinder.attack_risk_in_time[1][point]
        >= candidate.own_combat_strength
    ):
        if verbose:
            lprint(
                f"Tile is under attack risk of enemy {candidate.pathfinder.risk_grid[point]} while bot has {candidate.power}",
                point,
            )
        return True


def is_out_of_power(candidate, verbose=False):
    if candidate.power < 0:
        if verbose:
            lprint(
                f"{candidate.pathfinder.t}: out of power ({candidate.power}) {candidate.point}"
            )
        return True


def is_power_at_factory(candidate, power_at_factory, verbose=False):
    if power_at_factory <= 0:
        if verbose:
            lprint(f"{candidate.pathfinder.t}: no power at factory")
        return True


def invalid_on_target(candidate, goal, verbose=False):
    if goal.timed_targets is None:
        return
    point = candidate.point
    if point not in candidate.pathfinder.targets:
        return

    t = candidate.pathfinder.t

    if t == 0:
        return

    if t not in goal.timed_targets or (
        goal.timed_targets[t] != point
        and not (
            isinstance(goal.timed_targets[t], list) and point in goal.timed_targets[t]
        )
    ):
        if verbose:
            lprint(f"{t}: arrived too soon on {point}, {goal.timed_targets}")
        return True

    if goal.timed_targets_need_move and candidate.parent.point == point:
        if verbose:
            lprint(f"{t}: arrived without moving on {point}")
        return True
