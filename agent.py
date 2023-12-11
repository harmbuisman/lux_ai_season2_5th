import sys
import pickle
import time
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.sparse.csgraph import dijkstra
from skimage.segmentation import expand_labels, find_boundaries
import math
from lux.act_factory import factory_act
from lux.act_unit import unit_act
from lux.board import GameBoard
from lux.cargo import UnitCargo
from lux.point import Point
from lux.constants import (
    IS_KAGGLE,
    LICHEN_SHIELD_COVERAGE,
    MICRO_PENALTY_POWER_COST,
    POWER_ENEMY_FACTORY_MAGIC_VALUE,
    VALIDATE,
)
from lux.sentry import (
    get_shield_targets,
    get_lichen_shield_targets,
)
from lux.early_game import best_positions, get_factory_action, has_inbalanced_touch_both
from lux.factory import Factory, set_lichen_lake_ids
from lux.globals import _ATTACK_LOG, _COMBAT_OBSERVATIONS, _LEDGER, _N_OBSERVATIONS
from lux.kit import Board, EnvConfig, GameState, obs_to_game_state

# from lux.learning import get_empty_series, observation_from_feature_dict
from lux.lichen import (
    get_lichen_targets,
    set_lichen_choke_points,
)
from lux.team import FactionTypes, Team
from lux.unit import Unit
from lux.utils import (
    flatten,
    lprint,
    my_turn_to_place_factory,
    create_abort_file,
    abort_if_file_gone,
)
from lux.agent_methods import get_solar_targets
from lux.rubble import get_rubble_target_grid
import traceback

nested_default_dict = lambda: defaultdict(lambda: defaultdict(int))
cache_dir = Path("cache")


def agent_obs_to_game_state(player, step, env_cfg: EnvConfig, obs):
    units = dict()
    for agent in obs["units"]:
        units[agent] = dict()
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            cargo = UnitCargo(**unit_data["cargo"])
            unit = Unit(
                **unit_data,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg,
            )
            unit.cargo = cargo
            units[agent][unit_id] = unit

    factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
    factories = dict()

    #### CHANGED BETWEEN
    current_factories = sorted(list(player.all_factories.keys()))
    new_factories = sorted(flatten(list(v.keys()) for k, v in obs["factories"].items()))
    needs_update = current_factories != new_factories
    player.reset_factories = needs_update
    ### END

    for agent in obs["factories"]:
        factories[agent] = dict()
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            cargo = UnitCargo(**f_data["cargo"])
            ### CHANGED BETWEEN
            if needs_update:
                factory = Factory(**f_data, env_cfg=env_cfg)
            else:
                factory = player.all_factories[unit_id]
                factory.power = f_data["power"]
            ### END
            factory.cargo = cargo
            factories[agent][unit_id] = factory
            factory_occupancy_map[factory.pos_slice] = factory.team_id

    teams = dict()
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        faction = FactionTypes[team_data["faction"]]
        teams[agent] = Team(**team_data, agent=agent)

    return GameState(
        env_cfg=env_cfg,
        env_steps=step,
        board=Board(
            rubble=obs["board"]["rubble"],
            ice=obs["board"]["ice"],
            ore=obs["board"]["ore"],
            lichen=obs["board"]["lichen"],
            lichen_strains=obs["board"]["lichen_strains"].copy(),
            factory_occupancy_map=factory_occupancy_map,
            factories_per_team=obs["board"]["factories_per_team"],
            valid_spawns_mask=obs["board"]["valid_spawns_mask"],
        ),
        units=units,
        factories=factories,
        teams=teams,
    )


def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        # np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        self.factories = []
        self.opponent_factories = []
        self.all_factories = {}
        self.reset_factories = True

        self.monitor_units = []
        self.last_power_grid_update = "HEAVY"
        self.total_rubble = 0
        self._lichen_targets = None
        self.lichen_strains = None

        self.rss_path_commands = {}
        self.n_factories_per_team = None

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        ts = time.time()
        if step == 0:
            global _LEDGER
            # global _COMBAT_OBSERVATIONS
            # global _N_OBSERVATIONS
            # global _ATTACK_LOG

            # # reset the ledger and combat observations on a new game
            # _COMBAT_OBSERVATIONS = defaultdict(
            #     lambda: np.full((COMBAT_BUFFER_SIZE, 400), np.nan, dtype=np.float32)
            # )
            # _N_OBSERVATIONS = [0]
            # _ATTACK_LOG = defaultdict(nested_default_dict)
            # self.attack_log = _ATTACK_LOG[self.player]

            if not IS_KAGGLE:
                create_abort_file()

            _LEDGER = defaultdict(nested_default_dict)
            self.ledger = _LEDGER[self.player]

            self.game_state = obs_to_game_state(step, self.env_cfg, obs)
            game_state = self.game_state
            gb = GameBoard(game_state, agent=self)
            self.gb = gb

            score = best_positions(gb)
            best_score = np.max(score)

            best_pos = np.unravel_index(score.argmax(), score.shape, order="C")

            # 9x9 grid around best position set to 0 in a 48x48 numpy grid

            score[
                max(0, best_pos[0] - 4) : min(48, best_pos[0] + 5),
                max(0, best_pos[1] - 4) : min(48, best_pos[1] + 5),
            ] = 0
            second_best_score = np.max(score)
            second_best_pos = np.unravel_index(score.argmax(), score.shape, order="C")
            n_factories = obs["board"]["factories_per_team"]

            factor = best_score / second_best_score

            bid = max(20, int(min(factor // 1.5, n_factories - 1, 2)) * 10)
            if has_inbalanced_touch_both(gb, n_factories_per_team=n_factories):
                bid += 10
                bid = max(20, bid)

            self.n_factories_per_team = n_factories

            # bid = 1 if BID_ONE else min(1 + np.random.poisson(factor), MAX_BID)
            lprint(
                f"{best_pos}:{best_score}, {second_best_pos}: {second_best_score}, factor:{factor}, bid:{bid}"
            )

            te = time.time()
            duration = te - ts
            lprint(f"Step took {duration:.1f} sec", file=sys.stderr)

            return dict(faction="AlphaStrike", bid=bid)
        else:

            if step == 2:
                self.initialize_rss_power_grid()

            self.game_state = obs_to_game_state(step, self.env_cfg, obs)
            game_state = self.game_state
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )
            if not my_turn_to_place:
                return dict()

            # factory placement period
            gb = GameBoard(game_state, agent=self)

            if step == 1:
                # trigger numba cache
                get_distance_matrix(game_state.board.rubble)

            te = time.time()
            duration = te - ts
            lprint(f"Step took {duration:.1f} sec", file=sys.stderr)

            return get_factory_action(gb, self.player)

    def act(self, step: int, obs, remainingOverageTime: int = 60, silenced=False):
        global _LEDGER
        if _LEDGER is None:
            _LEDGER = defaultdict(nested_default_dict)
        # global _COMBAT_OBSERVATIONS
        # global _N_OBSERVATIONS
        # global _ATTACK_LOG
        # if _COMBAT_OBSERVATIONS is None:
        #     _COMBAT_OBSERVATIONS = defaultdict(
        #         lambda: np.full((COMBAT_BUFFER_SIZE, 400), np.nan, dtype=np.float32)
        #     )
        #     _N_OBSERVATIONS = [0]
        #     _ATTACK_LOG = defaultdict(nested_default_dict)

        self._lichen_targets = None
        self.ledger = _LEDGER[self.player]
        # self.n_observations = _N_OBSERVATIONS
        # self.combat_observations = _COMBAT_OBSERVATIONS[self.player]
        # self.attack_log = _ATTACK_LOG[self.player]

        ts = time.time()

        self.actions = dict()
        self._rubble_target_grid = None
        self._killnet_targets = None
        self._shield_targets = None
        self._path_shield_targets = None
        self._lichen_shield_targets = {"LIGHT": None, "HEAVY": None}
        self._get_lake_rubble_targets = None
        self._solar_targets = None
        self.lichen_points_under_attack = {}
        self.rss_steal = {}
        self.potential_targets = {}

        self.game_state = agent_obs_to_game_state(self, step, self.env_cfg, obs)
        game_state = self.game_state

        self.factories = list(game_state.factories[self.player].values())
        self.opponent_factories = list(game_state.factories[self.opp_player].values())
        factories = self.factories
        self.all_factories = {f.unit_id: f for f in self.opponent_factories + factories}
        self.strain_to_factory = {
            f.strain_id: f for f in self.opponent_factories + factories
        }

        units = list(game_state.units[self.player].values())

        day_str = "day" if game_state.is_day() else "night"

        lprint(
            f"####### STEP {game_state.real_env_steps} ({game_state.env_steps}) {day_str} {remainingOverageTime}:",
            f"Water:{[f.cargo.water for f in factories]}",
            f"Metal:{[f.cargo.metal for f in factories]}",
            f"Power:{[f.power for f in factories]}",
            # f"U/f:{[(len([u for u in f.units if u.is_heavy]), len([u for u in f.units if not u.is_heavy])) for f in factories]}",
            f"Units:{len(units)}",
            # f"Power: {[u.power for u in units]}",
            # f"Ice: {[u.cargo.ice for u in units]}",
            # f"Pos: {[(u.unit_id, u.point.xy) for u in units]}",
            # f"ID U: {[u.unit_id for u in units]}",
            file=sys.stderr,
        )

        check_time = game_state.real_env_steps > 1 and hasattr(self, "p2p_power_cost")
        init_power_grid = not hasattr(self, "p2p_power_cost")
        if init_power_grid or game_state.real_env_steps <= 1:
            self.initialize_power_grids(
                init_power_grid and game_state.real_env_steps > 0
            )
        self.game_board = GameBoard(game_state, self)

        self.adjacent_targets = []
        for u in units:
            for p in u.point.adjacent_points():
                if p.factory or p.unit is None or p.unit.is_heavy != u.is_heavy:
                    continue
                if p.unit.is_enemy:
                    self.adjacent_targets.append(p.unit)

        set_lichen_lake_ids(self.game_board)
        set_lichen_choke_points(self)

        self.opponent_adjacent_factory_points = set(
            flatten([f.adjacent_points for f in self.opponent_factories])
        )
        self.units = units
        self.heavies = [u for u in units if u.is_heavy]
        self.lights = [u for u in units if u.is_light]
        self.n_heavies = len(self.heavies)
        self.n_lights = len(self.lights)

        self.opponent_units = list(game_state.units[self.opp_player].values())
        self.pipeline_targets = [
            u for u in self.opponent_units if u.is_light and u.is_pipeline
        ]
        self.enemy_transports = [
            u for u in self.opponent_units if u.cargo.water or u.cargo.metal
        ]

        self.opponent_heavies = [u for u in self.opponent_units if u.is_heavy]
        self.opponent_lights = [u for u in self.opponent_units if u.is_light]
        self.n_opp_lights = len(self.opponent_lights)
        self.n_opp_heavies = len(self.opponent_heavies)
        light_bot_cost = self.env_cfg.ROBOTS["LIGHT"].METAL_COST
        self.opponent_bot_potential = len(game_state.units[self.opp_player]) + sum(
            [f.cargo.metal // light_bot_cost for f in self.opponent_factories]
        )

        heavy_bot_cost = self.env_cfg.ROBOTS["HEAVY"].METAL_COST
        self.opponent_heavy_bot_potential = len(
            [u for u in self.opponent_units if u.is_heavy]
        ) + sum([f.cargo.metal // heavy_bot_cost for f in self.opponent_factories])

        real_step = game_state.real_env_steps

        self.unit_dominance = False
        # (
        #     self.n_opp_lights * 2 < self.n_lights
        #     and self.n_opp_heavies < 2 * self.n_heavies
        # )

        if VALIDATE and real_step < 950:
            for unit in units:
                unit_id = unit.unit_id
                if unit_id not in self.ledger:
                    continue
                unit_ledger = self.ledger[unit_id]

                if real_step not in unit_ledger:
                    continue
                ledger_item = unit_ledger[real_step]

                if unit.power != ledger_item["power"]:
                    if unit.unit_type == "HEAVY" and unit.power > ledger_item["power"]:
                        lprint(
                            f"WARNING: {unit_id}: Power t={real_step} ({step}) {unit.point.xy}: {unit.power} != {ledger_item['power']} - potentially due to removal of rubble",
                            file=sys.stderr,
                        )
                    else:
                        assert (
                            unit.power == ledger_item["power"]
                        ), f"{unit_id}: Power t={real_step} ({step}) {unit.point.xy}: {unit.power} != {ledger_item['power']}"
                assert (
                    unit.cargo.ice == ledger_item["ice"]
                ), f"{unit_id}: Ice t={real_step} ({step}) {unit.point.xy}: {unit.cargo.ice} != {ledger_item['ice']}"
                assert (
                    unit.cargo.ore == ledger_item["ore"]
                ), f"{unit_id}: Ore t={real_step} ({step}) {unit.point.xy}: {unit.cargo.ore} != {ledger_item['ore']}"
                assert (
                    unit.point == ledger_item["point"]
                ), f"{unit_id}: Pos t={real_step} ({step}) {unit.point.xy}: {unit.point.xy} != {ledger_item['point'].xy}"

        self.get_rubble_target_grid()

        lichen = self.game_board.board.lichen.copy()
        strains = self.game_board.board.lichen_strains.copy()
        self.own_lichen = np.sum(np.isin(strains, self.own_lichen_strains) * lichen)
        self.opp_lichen = np.sum(np.isin(strains, self.enemy_lichen_strains) * lichen)

        try:
            unit_act(self)
            factory_act(self)
        except Exception as e:
            tb = traceback.format_exc()

            if IS_KAGGLE:
                print(str(tb)[::-1], file=sys.stderr)

            raise

        # self.add_combat_observations()
        te = time.time()

        duration = te - ts

        total_rubble = np.sum(game_state.board.rubble)

        add_time = 0.0
        if (
            duration < 1.2 and real_step % 25 <= 1
        ):  # and total_rubble != self.total_rubble:
            lprint(f"Step until power update took {duration:.1f} sec")
            self.initialize_power_grids()
            add_time = 1.0

        if not IS_KAGGLE:
            abort_if_file_gone()
            for u in self.units:
                prev_queue = to_json(u.start_action_queue)
                new_queue = to_json(u.action_queue)
                if not (u.unit_id not in self.actions or prev_queue != new_queue):
                    msg = f"######### {u} has same action queue {new_queue} as last step: {prev_queue}. Removing from agent actions!!!!"
                    lprint(msg)

                    del self.actions[u.unit_id]

                # assert u.unit_id not in self.actions or prev_queue != new_queue, msg

        remove_keys = []
        for key in self.actions:
            if key in self.all_factories:
                continue
            unit = self.game_board.unit_dict[key]
            if not unit.is_own:
                assert (
                    IS_KAGGLE or False
                ), f"Opponent unit {unit} has action {self.actions[key]}"
                remove_keys.append(key)
            if np.array_equal(unit.start_action_queue, unit.action_queue):
                assert (
                    IS_KAGGLE or False
                ), f"{unit} has same action {self.actions[key]} was {unit.start_action_queue}"
                remove_keys.append(key)

        for key in remove_keys:
            del self.actions[key]

        self.total_rubble = total_rubble
        te = time.time()
        duration = te - ts
        lprint(f"Step took {duration:.1f} sec", file=sys.stderr)
        # assert (
        #     IS_KAGGLE or not check_time or silenced or duration < 3.0 + add_time
        # ), f"TIMEOUT - Step took {duration:.1f} sec"
        # lprint(f"ACTIONS: {self.actions}", file=sys.stderr)
        return to_json(self.actions)

    def add_combat_observations(self):
        for (
            idx,
            unit_id,
            point,
            expected_point,
            is_digging_next,
            is_heavy,
            rubble_in_neighbourhood,
        ) in self.monitor_units:
            if unit_id in self.game_board.unit_dict:
                u = self.game_board.unit_dict[unit_id]
                move_dir = point.get_direction(u.point)
                self.combat_observations[idx, -1] = move_dir
            else:
                # unit died, need to find out how
                # need to check which point has higher rubble
                if expected_point.rubble > rubble_in_neighbourhood[expected_point]:
                    move_dir = point.get_direction(expected_point)
                    self.combat_observations[idx, -1] = move_dir
                else:
                    found = False
                    for p in [point] + point.adjacent_points():
                        if p.rubble > rubble_in_neighbourhood[p]:
                            move_dir = point.get_direction(p)
                            self.combat_observations[idx, -1] = move_dir
                            found = True
                            break

                    if not found and is_digging_next:
                        cfg = self.env_cfg.ROBOTS
                        rubble_dug = (
                            cfg["HEAVY"] if is_heavy else cfg["LIGHT"]
                        ).DIG_RUBBLE_REMOVED
                        if point.rubble > rubble_in_neighbourhood[point] - rubble_dug:
                            move_dir = 0
                            found = True

                    if not found and point == expected_point:
                        cfg = self.env_cfg.ROBOTS
                        rubble_dug = (
                            cfg["HEAVY"] if is_heavy else cfg["LIGHT"]
                        ).DIG_RUBBLE_REMOVED
                        rubble_before = rubble_in_neighbourhood[point]

                        # unit did not have a queue, but started digging
                        # e.g. dig = 2, drop = 1
                        if rubble_before - rubble_dug < point.rubble < rubble_before:
                            move_dir = point.get_direction(expected_point)
                            found = True

                        # unit died, but rubble was already at 100
                        if not found and point.rubble == 100:
                            move_dir = point.get_direction(expected_point)
                            found = True

                        # unit died in its own factory
                        if not found and expected_point.factory:
                            move_dir = point.get_direction(expected_point)
                            found = True

                    if not found:
                        assert (
                            IS_KAGGLE
                        ), f"Unit {unit_id} died, but could not determine move direction from {point} to {expected_point}?"
                        self.combat_observations[idx, -1] = -1

        self.monitor_units = []
        for u in self.units:
            for u2 in self.opponent_units:
                distance = u.point.distance_to(u2.point)
                if distance <= 2:
                    observation = u2.point.build_features()
                    # observation = observation_from_feature_dict(features)
                    idx = self.n_observations[0]
                    if idx > len(self.combat_observations) - 1:
                        lprint("WARNING: combat_observations buffer full")
                        return
                    self.combat_observations[idx] = observation
                    rubble_in_neighbourhood = {
                        p: p.rubble for p in [u2.point] + u2.point.adjacent_points()
                    }
                    self.monitor_units.append(
                        (
                            idx,
                            u2.unit_id,
                            u2.point,
                            u2.next_point,
                            u2.is_digging_next,
                            u2.is_heavy,
                            rubble_in_neighbourhood,
                        )
                    )
                    self.n_observations[0] += 1

    def initialize_rss_power_grid(self):
        gs = self.game_state
        env = gs.env_cfg
        board = gs.board
        N = env.map_size

        if not hasattr(self, "rss_p2p_power_cost"):
            lprint("Initializing RSS power grid")
            self.rss_p2p_power_cost = None
            self.rss_predecessors = None

            # add mega cost for rss so that they are avoided
            # add cost for rubble so that when choosing, prefer low rubble

            matrix = (
                np.ones_like(board.rubble)
                + (0.0001 * board.rubble)
                + (board.ice + board.ore) * 1000
            )

            # add small cost for enemy factories
            enemy_map = np.zeros_like(board.factory_occupancy_map)
            enemy_map[board.factory_occupancy_map == 1 - self.team_id] = 1

            total_map = enemy_map.copy()
            for _ in range(12):
                enemy_map = expand_labels(enemy_map, 1)
                total_map += enemy_map

            matrix = matrix + 0.01 * total_map

            # Find the shortest path from the start to the end
            dmat = get_distance_matrix(matrix, N)
            rss_p2p_power_cost, rss_predecessors = dijkstra(
                dmat, return_predecessors=True
            )
            # cast to int such that we can do == comparisons
            self.rss_p2p_power_cost = rss_p2p_power_cost.astype(int)
            self.rss_predecessors = rss_predecessors

            if not IS_KAGGLE:
                # save to disk
                with open(cache_dir / "rss_p2p_power_cost.pkl", "wb") as f:
                    pickle.dump(rss_p2p_power_cost, f)
                with open(cache_dir / "rss_predecessors.pkl", "wb") as f:
                    pickle.dump(rss_predecessors, f)

    def initialize_power_grids(self, both=False):
        gs = self.game_state
        env = gs.env_cfg
        board = gs.board
        N = env.map_size

        enemy_id = 1 - self.team_id
        enemy_factories = board.factory_occupancy_map == enemy_id

        if not hasattr(self, "p2p_power_cost"):
            self.p2p_power_cost = {}
            self.predecessors = {}
        # deal with slow performance
        last_heavy = self.last_power_grid_update == "HEAVY"
        self.last_power_grid_update = "LIGHT" if last_heavy else "HEAVY"

        unit_types = (
            ["LIGHT", "HEAVY"] if both else ["LIGHT"] if last_heavy else ["HEAVY"]
        )

        if both:
            create_abort_file()
            path = cache_dir / "p2p_power_cost_HEAVY.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.rss_p2p_power_cost = pickle.load(f)
                with open(cache_dir / "rss_predecessors.pkl", "rb") as f:
                    self.rss_predecessors = pickle.load(f)
                for unit_type in unit_types:
                    with open(cache_dir / f"p2p_power_cost_{unit_type}.pkl", "rb") as f:
                        self.p2p_power_cost[unit_type] = pickle.load(f)
                    with open(cache_dir / f"predecessors_{unit_type}.pkl", "rb") as f:
                        self.predecessors[unit_type] = pickle.load(f)
                return

            self.initialize_rss_power_grid()

        for unit_type in unit_types:
            lprint(f"Initializing power grids for {unit_type}")
            unit_cfg = gs.env_cfg.ROBOTS[unit_type]

            move_cost = unit_cfg.MOVE_COST
            rubble_factor = unit_cfg.RUBBLE_MOVEMENT_COST
            matrix = np.floor(
                board.rubble[:N, :N] * rubble_factor
                + move_cost
                + enemy_factories * POWER_ENEMY_FACTORY_MAGIC_VALUE  # magic value
            )

            # on same cost, prefer the route that skips ice, ore and factories
            matrix = (
                matrix
                + (board.ice + board.ore + (board.factory_occupancy_map >= 0))
                * MICRO_PENALTY_POWER_COST
            )

            # add small cost for enemy factories
            enemy_map = np.zeros_like(board.factory_occupancy_map)
            enemy_map[board.factory_occupancy_map == 1 - self.team_id] = 1

            total_map = enemy_map.copy()
            for _ in range(12):
                enemy_map = expand_labels(enemy_map, 1)
                total_map += enemy_map

            matrix = matrix + MICRO_PENALTY_POWER_COST * total_map

            # Find the shortest path from the start to the end
            dmat = get_distance_matrix(matrix, N)
            p2p_power_cost, predecessors = dijkstra(dmat, return_predecessors=True)

            # cast to int such that we can do == comparisons
            self.p2p_power_cost[unit_type] = p2p_power_cost.astype(int)
            self.predecessors[unit_type] = predecessors
            if not IS_KAGGLE:
                # save to disk
                with open(cache_dir / f"p2p_power_cost_{unit_type}.pkl", "wb") as f:
                    pickle.dump(p2p_power_cost, f)
                with open(cache_dir / f"predecessors_{unit_type}.pkl", "wb") as f:
                    pickle.dump(predecessors, f)

    def enemy_lichen(self):
        enemy_factories = self.opponent_factories
        if not enemy_factories:
            return 0
        return sum(f.lichen for f in enemy_factories)

    def get_lichen_targets(self, verbose=False):
        return get_lichen_targets(self, verbose=verbose)

    def min_factory_distance(self):
        def euclidean_distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        min_dist = 10000
        for f in self.factories:
            for ef in self.opponent_factories:
                distance = euclidean_distance(
                    f.center.x, f.center.y, ef.center.x, ef.center.y
                )
                if distance < min_dist:
                    min_dist = distance

        return min_dist

    def get_shield_targets(self, verbose=False):
        control_range = max(3, (self.min_factory_distance()) // 2)
        return get_shield_targets(self, control_range=control_range, verbose=verbose)

    def get_lichen_shield_targets(self, unit=None, verbose=False):
        return get_lichen_shield_targets(self, unit=unit, verbose=verbose)

    def get_killnet_targets(self, verbose=False):  # agent
        if self._killnet_targets is None:
            gb = self.game_board
            enemy_map = gb.board.factory_occupancy_map == (1 - gb.agent.team_id)

            enemy_map = expand_labels(enemy_map, 3)

            enemy_map = find_boundaries(enemy_map, mode="inner")

            existing_kill_net = np.zeros_like(enemy_map)
            for unit in self.units:
                if unit.is_killnet:
                    existing_kill_net[unit.last_point.xy] = 1

            existing_kill_net = expand_labels(existing_kill_net, 2)
            enemy_map = enemy_map * (existing_kill_net == 0)

            if verbose:
                plt.imshow(existing_kill_net.T)
                plt.title("existing killnet")
                plt.show()

                plt.imshow(((5 * enemy_map + gb.board.rubble / 100)).T)
                plt.title("killnet candidates")
                plt.show()

            targets = gb.grid_to_points(enemy_map)
            self._killnet_targets = set(targets)
        return self._killnet_targets

    def update_killnet_action(self, point: Point):
        covered_points = set(point.points_within_distance(2))
        self._killnet_targets = self._killnet_targets - covered_points

    def update_shield_action(self, point: Point):
        covered_points = set(point.points_within_distance(2))
        self._shield_targets = self._shield_targets - covered_points

    def update_solar_hit(self, point: Point):
        if point in self._solar_targets:
            del self._solar_targets[point]

    def update_lichen_shield_action(self, point: Point, unit: Unit):
        if self._lichen_shield_targets[unit.unit_type] is None:
            return
        covered_points = set(point.points_within_distance(LICHEN_SHIELD_COVERAGE))
        self._lichen_shield_targets[unit.unit_type] = (
            self._lichen_shield_targets[unit.unit_type] - covered_points
        )

    def get_rubble_target_grid(self, verbose=False):
        return get_rubble_target_grid(self, verbose=verbose)

    def get_solar_targets(self, verbose=False):
        return get_solar_targets(self, verbose=verbose)


@jit(nopython=True, cache=True)
def get_distance_matrix(matrix, N=48):
    # Create a 2D array to store the p2p_power_cost between the cells
    p2p_power_cost = np.zeros((N * N, N * N))

    # # Fill the p2p_power_cost array with the weights of the cells
    for i in range(N):
        for j in range(N):
            for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if x >= 0 and x < N and y >= 0 and y < N:
                    p2p_power_cost[i * N + j, x * N + y] = matrix[x, y]
    return p2p_power_cost
