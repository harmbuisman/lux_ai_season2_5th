from collections import defaultdict
from typing import List

from lux.point import Point
import numpy as np


def _get_path(predecessors, i, j):
    path = [j]
    k = j
    while predecessors[i, k] != -9999:
        path.append(predecessors[i, k])
        k = predecessors[i, k]
    return path[::-1]


def get_optimal_path(gb, start: Point, end: Point, unit=None, type="HEAVY"):
    unit_type = unit.unit_type if unit else type
    predecessors = gb.agent.predecessors[unit_type]
    path = _get_path(predecessors, start.id, end.id)
    return gb.get_points_by_idxs(path)


def get_rss_optimal_path(gb, start: Point, end: Point):
    predecessors = gb.agent.rss_predecessors
    path = _get_path(predecessors, start.id, end.id)
    return gb.get_points_by_idxs(path)


def collapse_commands_to_queue(commands, repeat=0):
    collapsed_commands = []
    prev_command = None
    action = [None]
    for direction in commands:
        if direction == prev_command:
            action[-1] += 1
        else:
            if action[0] is not None:
                collapsed_commands.append(action)
            action = np.array([0, int(direction), 0, 0, repeat, 1])
        prev_command = direction

    if action[0] is not None:
        collapsed_commands.append(action)
    return collapsed_commands


def get_rss_path_queue(agent, start: Point, end: Point, repeat):
    key = (start, end)
    if key not in agent.rss_path_commands:
        path = get_rss_optimal_path(agent.game_board, start, end)
        commands = commands_from_path(path)

        agent.rss_path_commands[key] = collapse_commands_to_queue(commands, repeat)
    return agent.rss_path_commands[key].copy()


def commands_from_path(path: List[Point]):
    commands = []
    for i in range(len(path) - 1):
        commands.append(path[i].get_direction(path[i + 1]))
    return commands


def power_required(unit, start: Point, end: Point):
    p2p_power_cost = unit.game_board.agent.p2p_power_cost[unit.unit_type]
    return p2p_power_cost[start.id, end.id]


def get_optimal_directions_dict(unit, starts=None, ends=None):
    assert ends is not None
    if starts is None:
        starts = [unit.point]

    paths = []
    gb = unit.game_board
    p2p = defaultdict(list)
    directions = defaultdict(list)
    for start in starts:
        for end in ends:
            paths.append(get_optimal_path(gb, start, end, unit))

    for path in paths:
        prev_p = path[0]
        for p in path[1:]:
            if p not in p2p[prev_p]:
                p2p[prev_p].append(p)
                directions[prev_p].append(prev_p.get_direction(p))
            prev_p = p

    return directions
