from skimage.segmentation import expand_labels, find_boundaries
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from lux.constants import LICHEN_SHIELD_COVERAGE, BREAK_EVEN_ICE_TILES
from lux.router import get_optimal_path, get_rss_optimal_path
from lux.utils import lprint


def get_path_shield_targets(self, verbose=False):
    if self._path_shield_targets is None or verbose:
        gb = self.game_board

        def get_points_from(f1, f2, direct=False):
            fn = get_rss_optimal_path if direct else get_optimal_path
            path = fn(gb, f1.center, f2.center)

            # don't consider paths that go through other factories
            if any(
                [
                    p.closest_factory not in [f1, f2]
                    and p.closest_factory.distance_to(p) <= 2
                    for p in path
                ]
            ):
                return []

            center = path[len(path) // 2]
            distance = center.distance_to(f1)
            targets = [
                p
                for p in center.points_within_distance(1)
                if 2 < p.distance_to(f1) <= distance
            ]

            return targets

        all_targets = []
        for f1 in self.factories:
            for f2 in self.opponent_factories:
                if f1.distance_to(f2) > 15:
                    continue

                targets = get_points_from(f1, f2, direct=False) + get_points_from(
                    f1, f2, direct=True
                )
                all_targets += targets

        point_ids = [p.id for p in set(all_targets)]
        self._path_shield_targets = point_ids
    return self._path_shield_targets


def get_shield_targets(self, control_range=10, verbose=False):
    """Return a list of targets for shields"""
    if verbose:
        self._shield_targets = None
    if self._shield_targets is None:
        gb = self.game_board

        control = gb.board.factory_occupancy_map + 1
        control = expand_labels(control, distance=control_range)
        if verbose:
            plt.imshow(control.T + gb.board.factory_occupancy_map.T)
            plt.title(f"Control range={control_range}")
            plt.show()

        # check if there is overlap
        boundaries = find_boundaries(control, background=0, mode="outer") * control == (
            2 - self.team_id
        )

        if boundaries.max() == 0:
            return set()
        mode = "thick"  # "inner" if control_range <= 10 else "outer"
        boundaries = (
            find_boundaries(control, background=0, mode=mode)
            * expand_labels(control == (self.team_id + 1), 1)
            * expand_labels(control == (2 - self.team_id), 1)
        )

        if verbose:
            plt.imshow(
                boundaries.T
                + gb.board.factory_occupancy_map.T
                + gb.board.rubble.T / 300
            )
            plt.title(f"Boundaries range={control_range}")
            plt.show()

        path_shield_targets = get_path_shield_targets(self, verbose=verbose)
        boundaries.ravel()[path_shield_targets] = 2

        if verbose:
            plt.imshow(
                boundaries.T
                + gb.board.factory_occupancy_map.T
                + gb.board.rubble.T / 300
            )
            plt.title(f"Boundaries range={control_range} + pathshield")
            plt.show()

        existing_shield = np.zeros_like(control)
        for unit in self.units:
            if unit.is_shield:
                existing_shield[unit.last_point.xy] = 1

        existing_shield = expand_labels(existing_shield, 2)
        boundaries = boundaries * (existing_shield == 0)

        if verbose:
            plt.imshow(
                boundaries.T
                + gb.board.factory_occupancy_map.T
                + gb.board.rubble.T / 300
            )
            plt.title(f"Boundaries minus covered range={control_range}")
            plt.show()

        targets = gb.grid_to_points(boundaries)
        if not targets and control_range < 20:
            targets = get_shield_targets(self, control_range + 2, verbose=verbose)
        self._shield_targets = set(targets)

    return self._shield_targets


def get_lichen_shield_targets(self, unit, max_enemy_distance=15, verbose=False):
    if verbose:
        self._lichen_shield_targets[unit.unit_type] = None
    if self._lichen_shield_targets[unit.unit_type] is None:
        gb = self.game_board
        own_lichen_mask = np.isin(
            self.connected_incl_single_lichen_map, self.own_lichen_strains
        )

        own_lichen_mask_exp = expand_labels(own_lichen_mask, 1)
        selem = morphology.disk(2)

        eroded_labeled_image = morphology.erosion(own_lichen_mask_exp, selem)
        eroded_labeled_image = morphology.erosion(
            eroded_labeled_image, morphology.disk(1)
        )

        protection_mask = expand_labels(eroded_labeled_image, 1)

        boundaries = find_boundaries(protection_mask, background=0, mode="inner")

        boundaries = 1.0 * boundaries * (gb.board.factory_occupancy_map == -1)

        if verbose:
            plt.imshow((1.0 * protection_mask + boundaries + own_lichen_mask).T)
            plt.title("protection_mask+boundaries+own_lichen_mask")
            plt.show()
        enemy_control = gb.board.factory_occupancy_map == 1 - self.team_id

        enemy_control_exp = enemy_control
        distance_matrix = 1.0 * enemy_control.copy()
        for i in range(max_enemy_distance):
            enemy_control_exp = 1.0 * expand_labels(enemy_control_exp, 1)
            distance_matrix += enemy_control_exp

        if verbose:
            plt.imshow((distance_matrix * 1.0 * boundaries).T)
            plt.title("distance_matrix*boundaries")
            plt.show()

        target_score_map = distance_matrix * 1.0 * boundaries

        existing_shield = np.zeros_like(target_score_map)
        for u in self.units:
            if u.is_heavy != unit.is_heavy:
                continue

            if u.is_shield or u.is_guard or u.is_hub:
                existing_shield[u.last_point.xy] = 1

        existing_shield = expand_labels(existing_shield, LICHEN_SHIELD_COVERAGE)
        target_score_map = (target_score_map * (existing_shield == 0)) > 0.5

        targets = gb.grid_to_points(target_score_map)

        self._lichen_shield_targets[unit.unit_type] = set(targets)

    return self._lichen_shield_targets[unit.unit_type]


def get_lichen_sentry_points(agent, verbose=False):
    sentry_points = {}
    gb = agent.game_board
    for f_own in agent.factories:
        for f_opp in agent.opponent_factories:
            if f_own.n_connected_tiles > BREAK_EVEN_ICE_TILES:
                sentry_point = None
                total_lichen = 0
                for p in get_optimal_path(gb, f_own.center, f_opp.center, type="HEAVY"):
                    if (
                        p.lichen > 0
                        and p.own_lichen
                        and p.closest_own_factory_distance > 1
                        and p.lichen_strains == f_own.id
                    ):
                        #                     print(f_own, f_opp, p, p.own_lichen, p.lichen, p.lichen_strains==f_own.id)
                        sentry_point = p
                        total_lichen += p.lichen
                if sentry_point and sentry_point not in sentry_points:
                    points_to_check = [
                        ap for ap in sentry_point.surrounding_points()
                    ] + [sentry_point]
                    if verbose:
                        lprint(sentry_point, points_to_check)
                    covered = any(
                        [
                            u.last_point == pp
                            for pp in points_to_check
                            for u in pp.visited_by
                            if u.is_heavy
                        ]
                    )
                    if not covered:
                        sentry_points[sentry_point] = total_lichen

    if verbose:
        for p, lichen in sentry_points.items():
            lprint(p, p.own_lichen, p.lichen, p.lichen_strains, lichen)

    return sentry_points
