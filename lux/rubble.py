import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries, expand_labels

from lux.utils import set_inverse_zero
from lux.constants import PRIORITY, TTMAX, MIN_NORMAL_RUBBLE_PRIORITY

# from lux.learning import get_empty_series, observation_from_feature_dict
from lux.lichen import (
    get_lichen_connector_points,
)
from lux.router import get_rss_optimal_path, get_optimal_path
from lux.utils import (
    flatten,
    lprint,
)


def get_rubble_target_grid(self, verbose=False):
    if verbose:
        self._rubble_target_grid = None
    if self._rubble_target_grid is None:
        board = self.game_state.board
        gb = self.game_board
        candidate_board = np.zeros_like(board.rubble)

        rb = (
            1.0
            * (board.rubble == 0)
            * (board.ice == 0)
            * (board.ore == 0)
            * (board.factory_occupancy_map < 0)
        )
        clusters = label(rb, connectivity=1)

        enemy_lichen = np.isin(board.lichen_strains, self.enemy_lichen_strains)
        enemy_factories = np.isin(board.factory_occupancy_map, [1 - self.team_id])
        enemy_lichen_growth = expand_labels(enemy_lichen + enemy_factories, 1)

        strains_map = board.lichen_strains.copy()
        for f in gb.factories:
            if f.is_own and not f.clear_rubble_for_lichen():
                continue
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    strains_map[f.pos[0] + dx, f.pos[1] + dy] = f.id

        if verbose:
            lprint(f"Clusters: {np.unique(clusters)}")
            plt.imshow(
                (
                    clusters
                    + (np.max(clusters) + 5) * (board.factory_occupancy_map >= 0)
                ).T
            )
            plt.title(f"Clusters: {np.unique(clusters)}")
            plt.show()

        clusters = clusters * 1.0 * (enemy_lichen_growth == 0)

        for c in range(int(np.max(clusters)) + 1):
            mapping = clusters == c
            size = np.sum(mapping)
            if size <= 4:
                clusters[mapping] = 0

        if verbose:
            plt.imshow((clusters + (c + 5) * (board.factory_occupancy_map >= 0)).T)
            plt.title(f"clusters ex enemy lichen growth: {np.unique(clusters)}")
            plt.show()

        for p in get_lake_rubble_targets(self):
            p.rubble_target_value = 0
            candidate_board[p.xy] = max(candidate_board[p.xy], PRIORITY.LAKE_CLEARING)

        # visited points (TODO)
        # lichen expansion (TODO)
        for f in gb.agent.factories:
            has_ice_hubs = len(f.ice_hubs) > 0
            enough_space = f.lake_large_enough()
            avg_rubble = sum([p.rubble for p in f.adjacent_points]) / 12

            # routes to rss
            for factory_point in [f.center]:  # f.edge_tiles():
                rss_points = []
                if not f.ore_hubs and f.cargo.metal < 200 and f.ore_points:
                    closest_ore = f.ore_points[0]
                    co_distance = closest_ore.distance_to(f)
                    rss_points += [
                        p
                        for p in f.ore_points[:5]
                        if p.distance_to(f) <= min(co_distance, 5)
                    ]

                if f.ice_points:
                    closest_ice = f.ice_points[0]
                    ci_distance = closest_ice.distance_to(f)
                    rss_points += [
                        p
                        for p in f.ice_points[:5]
                        if p.distance_to(f) <= min(ci_distance, 5)
                    ]

                for i, rp in enumerate(
                    rss_points
                ):  # max on 20 to prevent too many paths
                    # don't dig paths when I will likely never mine there from this factory
                    distance = rp.distance_to(f)
                    if distance == 1:
                        continue
                    if rp.distance_to(f) - 1 <= rp.closest_factory_distance:
                        pwr_cost = self.p2p_power_cost["LIGHT"][factory_point.id][rp.id]

                        tough_route = distance > 6 and pwr_cost / distance > 2.5
                        paths = []

                        paths.append(get_rss_optimal_path(gb, factory_point, rp))

                        priority = (
                            PRIORITY.RSS_CONNECTOR_TOUGH
                            if tough_route
                            else PRIORITY.RSS_CONNECTOR
                        )

                        # if i == 0 and rp.ore:
                        #     # priority path should be fastest to clear, not shortest
                        #     paths.append(
                        #         get_optimal_path(gb, factory_point, rp, type="LIGHT")
                        #     )

                        for i_path, candidates in enumerate(paths):
                            # if supporting a factory, we can assume that one will provide us with metal
                            if not f.supports_factory:
                                if i == 0 and rp.ore:
                                    if rp.distance_to(f) < 4:
                                        priority = PRIORITY.CLOSEST_ORE_CLOSE
                                    else:
                                        priority = PRIORITY.CLOSEST_ORE

                            for p in candidates:
                                if p.rubble == 0 or p == rp:
                                    continue

                                candidate_board[p.xy] = max(
                                    candidate_board[p.xy], priority
                                )
                                # long distant routes are taken by lights, not heavies
                                if (
                                    p.distance_to(f) <= (4 + has_ice_hubs)
                                    or not enough_space
                                    or priority == PRIORITY.CLOSEST_ORE
                                ):
                                    p.rubble_target_value = 0
                                else:
                                    p.rubble_target_value = min(
                                        p.rubble_target_value, 19
                                    )

                                if (
                                    priority == PRIORITY.CLOSEST_ORE
                                    and p.get_rubble_at_time(TTMAX) > 0
                                ):
                                    f.has_uncleared_oreroute = True

            distance = 1 if enough_space or not has_ice_hubs else 2
            if f.full_power_hub():
                distance = 2

            if f.clear_rubble_for_lichen() or f.full_power_hub():
                for p in f.points_within_distance(distance):
                    if not p.factory:
                        priority = (
                            (
                                PRIORITY.SURROUNDING_ICE_TOUGH
                                if p.rubble >= 59
                                else PRIORITY.SURROUNDING_ICE
                                if (p.rubble >= 19) or enough_space
                                else PRIORITY.KICK_START_SURROUNDING_ICE
                            )
                            if has_ice_hubs
                            else PRIORITY.SURROUNDING_NON_ICE
                        )

                        # lprint("PRIORITY CLOSE FACOTRY", p, priority, priority.value)
                        candidate_board[p.xy] = max(candidate_board[p.xy], priority)
                        p.rubble_target_value = 0

            if self.n_heavies > self.n_opp_heavies * 2:
                for p in f.get_assault_rubble_points():
                    candidate_board[p.xy] = max(candidate_board[p.xy], PRIORITY.ASSAULT)
                    p.rubble_target_value = min(p.rubble_target_value, 0)

            for p in f.get_transport_rubble_points():
                candidate_board[p.xy] = max(
                    candidate_board[p.xy],
                    PRIORITY.TRANSPORT_SUPPORT if f.ore_only else PRIORITY.TRANSPORT,
                )
                p.rubble_target_value = min(p.rubble_target_value, 19)

            # closest cluster
            if f.clear_rubble_for_lichen():
                max_distance = 5 if enough_space else 9  # for cluster connector
                if has_ice_hubs:
                    matrix = clusters.copy()
                    set_inverse_zero(matrix, f.center, max_distance=max_distance)

                    # add the factory and recalculate clusters as factory can connect them
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            matrix[f.pos[0] + dx, f.pos[1] + dy] = 1
                    matrix = 1.0 * (matrix > 0)
                    matrix = label(matrix, connectivity=1)

                    if verbose:
                        plt.imshow(matrix.T)
                        plt.title(f"{f} inverse zero with factory filling")
                        plt.show()

                    # remove the cluster of the factory itself
                    matrix = np.where(matrix == matrix[f.pos[0], f.pos[1]], 0, matrix)

                    if verbose:
                        plt.imshow(matrix.T)
                        plt.title(f"{f} inverse zero with connected removed")
                        plt.show()

                    cluster_to_distance = {}
                    for c in np.unique(matrix):
                        if c > 0:
                            cluster_size = np.sum(matrix == c)

                            cps = sorted(
                                gb.get_points_by_idxs(gb.grid_to_ids(matrix == c)),
                                key=f.distance_to,
                            )

                            closest_cluster_point = cps[0]
                            distance_to_cluster = f.distance_to(closest_cluster_point)

                            if distance_to_cluster <= max_distance:
                                if verbose:
                                    lprint(f"{f}: cluster {c} size: {cluster_size}")
                                    lprint(
                                        f"{f}: closest cluster point: {closest_cluster_point} ({distance_to_cluster})"
                                    )

                                cluster_to_distance[c] = distance_to_cluster

                    top_prio_given = False
                    for c in sorted(
                        cluster_to_distance.keys(), key=lambda c: cluster_to_distance[c]
                    ):
                        distance_to_cluster = cluster_to_distance[c]
                        cluster_size = np.sum(matrix == c)

                        cps = sorted(
                            gb.get_points_by_idxs(gb.grid_to_ids(matrix == c)),
                            key=f.distance_to,
                        )

                        best_cp = None
                        candidates = []
                        best_rubble = 100000
                        for cp in cps:
                            if cp.distance_to(f) == distance_to_cluster:
                                closest_tile = f.closest_tile(cp)
                                cp_candidates = get_optimal_path(
                                    gb,
                                    closest_tile,
                                    cp,
                                    type="HEAVY" if gb.step > 1 else "LIGHT",
                                )

                                if any([p.ore or p.ice for p in cp_candidates]):
                                    cp_candidates = get_rss_optimal_path(
                                        gb,
                                        closest_tile,
                                        cp,
                                    )

                                rubble = sum(p.rubble for p in cp_candidates)
                                if rubble < best_rubble:
                                    best_rubble = rubble
                                    best_cp = cp
                                    candidates = cp_candidates
                            else:
                                break

                        if any(
                            (p.factory and p.factory != f)
                            or (
                                p.closest_factory_distance == 1
                                and p.closest_factory != f
                            )
                            for p in candidates
                        ):
                            if verbose:
                                lprint(
                                    f"{f}: don't dig through or next to other factories: {candidates}",
                                )
                            continue
                        priority = (
                            PRIORITY.LAKE_CONNECTOR_LOW_PRIO
                            if distance_to_cluster > cluster_size
                            else PRIORITY.LAKE_CONNECTOR_PRIO
                            if (avg_rubble > 50 or distance_to_cluster < 3)
                            and not top_prio_given
                            else PRIORITY.LAKE_CONNECTOR
                        )

                        if priority == PRIORITY.LAKE_CONNECTOR_PRIO:
                            top_prio_given = True

                        for p in candidates:
                            if p.rubble and p != best_cp:
                                # lprint(
                                #     "PRIORITY LAKE_CONNECTOR",
                                #     p,
                                #     priority,
                                #     priority.value,
                                # )
                                candidate_board[p.xy] = max(
                                    candidate_board[p.xy], priority
                                )

                                p.rubble_target_value = 0

            # lichen growth potential
            frontier = find_boundaries(strains_map == f.id, mode="outer")
            rubble_frontier = frontier * (gb.board.rubble > 0)

            fps = gb.get_points_by_idxs(gb.grid_to_ids(rubble_frontier))

            pts = f.filter_growth_points(fps, skip_rubble=True)
            min_rubble = 0
            if pts:
                min_rubble = min(p.rubble for p in pts)
            if verbose:
                lprint(f"{f}: min rubble in growth points: {min_rubble}")

            def priority_from_rubble(rubble):
                return (
                    PRIORITY.LICHEN_0_10
                    if rubble <= 10
                    else PRIORITY.LICHEN_10_20
                    if rubble <= 20
                    else PRIORITY.LICHEN_20_30
                    if rubble <= 30
                    else PRIORITY.LICHEN_30_40
                    if rubble <= 40
                    else PRIORITY.LICHEN_40_50
                    if rubble <= 50
                    else PRIORITY.LICHEN_50_60
                    if rubble <= 60
                    else PRIORITY.LICHEN_60_80
                    if rubble <= 80
                    else PRIORITY.LICHEN_80_100
                )

            offset = 0
            if not enough_space:
                min_priority = priority_from_rubble(min_rubble)

                if min_priority < MIN_NORMAL_RUBBLE_PRIORITY:
                    offset = MIN_NORMAL_RUBBLE_PRIORITY - min_priority

            for p in pts:
                priority = priority_from_rubble(p.rubble) + offset
                if not enough_space or (p.rubble <= 40 and p.distance_to(f) <= 4):
                    if enough_space:
                        # if enough space lower the priority
                        # lprint("LOWERING PRIORITY", p, priority)
                        priority -= 2
                    if p.distance_to(f) <= 4:
                        priority += 1
                    candidate_board[p.xy] = max(candidate_board[p.xy], priority)
                    p.rubble_target_value = 0
                elif p.rubble <= 19:
                    candidate_board[p.xy] = max(
                        candidate_board[p.xy], PRIORITY.LICHEN_NOTHING_TO_DO
                    )
                    p.rubble_target_value = 0

            visit_threshold = 5 + len(gb.factories) + gb.step / 100
            for p in f.points_within_distance(12):
                if (
                    p.visit_count > visit_threshold
                    and not p.ice
                    and not p.ore
                    and p.rubble >= 20
                ):
                    candidate_board[p.xy] = max(
                        candidate_board[p.xy], PRIORITY.HIGH_VISIT
                    )
                    p.rubble_target_value = 0

            # lichen choke points
            cluster_to_kills_points = gb.agent.choke_clusters_own
            kills_to_clusters = {}
            for cluster, (kills, points) in cluster_to_kills_points.items():
                kills_to_clusters.setdefault(kills, []).append(cluster)
                aps = set(flatten([p.adjacent_points() for p in points]))
                for p in aps:

                    if p.rubble == 0:
                        continue
                    candidate_board[p.xy] = max(
                        candidate_board[p.xy],
                        kills * (100 - 20 * (p.rubble // 20)) / 99,
                    )
                    p.rubble_target_value = 0

        # lichen lakes connected by lichen at a distance
        points = get_lichen_connector_points(gb.agent, verbose=verbose)

        strains_to_connect = [f.id for f in gb.agent.factories if len(f.ice_hubs) > 0]

        valid_lichen_strains = set(
            np.unique(gb.board.lichen_strains[gb.board.lichen > 0])
        )
        strains_to_connect = set(strains_to_connect).intersection(valid_lichen_strains)

        for point in points:
            if point.lichen_strains not in strains_to_connect:
                continue
            rb = 1.0 * (board.rubble == 0) * (board.ice == 0) * (board.ore == 0)

            clusters = label(rb, connectivity=1)

            max_distance = 4
            set_inverse_zero(clusters, point, max_distance)
            #     if verbose:
            #         plt.imshow(result.T)
            #         plt.title("")
            #         plt.show()

            matrix = clusters.copy()
            my_cluster = matrix[point.xy]

            for c in np.unique(matrix):
                if c == my_cluster:
                    continue
                if c > 0:
                    cps = sorted(
                        [
                            pt
                            for pt in gb.get_points_by_idxs(gb.grid_to_ids(matrix == c))
                            if pt.factory is None and pt.lichen == 0
                        ],
                        key=point.distance_to,
                    )
                    if not cps:
                        continue

                    closest_cp = cps[0]
                    distance_to_cluster = point.distance_to(closest_cp)

                    if distance_to_cluster <= max_distance:
                        for factory_point in [point]:  # f.edge_tiles():
                            candidates = get_rss_optimal_path(
                                gb, factory_point, closest_cp
                            )
                            for p in candidates:

                                if p.rubble and p != closest_cp:
                                    candidate_board[p.xy] = max(
                                        candidate_board[p.xy],
                                        PRIORITY.LAKE_CONNECTOR_PRIO
                                        if distance_to_cluster <= 3
                                        else PRIORITY.LAKE_CONNECTOR,
                                    )

                                    p.rubble_target_value = 0

        for f in self.factories:
            if f.power_hub_push and f.n_heavies > 2:
                for p in f.get_power_hub_positions():
                    if p.factory or p.rubble == 0:
                        continue
                    candidate_board[p.xy] = max(
                        candidate_board[p.xy], PRIORITY.POWER_HUB_POSITION
                    )
                    p.rubble_target_value = 0

        include_map = (
            (board.factory_occupancy_map == -1)
            * (board.ice == 0)
            * (board.ore == 0)
            * (board.rubble > 0)
            * (enemy_lichen_growth == 0)
        )
        if verbose:
            plt.imshow((candidate_board * include_map).T)
            plt.title("rubble_target_grid")
            plt.show()

        self._rubble_target_grid = candidate_board * include_map
    return self._rubble_target_grid


def get_transport_rubble_points(self):  # factory
    if self._transport_rubble_points is None:
        points = []
        factories = self.game_board.agent.factories
        if len(factories) > 1:
            other_factories = [f for f in factories if f != self]
            other_factories = [sorted(other_factories, key=self.distance_to)[0]]
            for f2 in other_factories:

                f1_tiles = [self.center]  # self.edge_tiles()
                f2_tiles = [f2.center]  # closest.edge_tiles()

                for p1 in f1_tiles:
                    for p2 in f2_tiles:
                        candidates = get_rss_optimal_path(self.game_board, p1, p2)

                        if any(
                            [c.closest_enemy_factory_distance <= 2 for c in candidates]
                        ):
                            continue

                        candidates = [
                            c
                            for c in candidates
                            if c.rubble > 0
                            and c not in points
                            and not c.factory
                            and not c.next_to_enemy_factory
                            and c.get_rubble_at_time(TTMAX) > 0
                        ]

                        points += candidates
        self._transport_rubble_points = points
    return self._transport_rubble_points


def get_assault_rubble_points(self):  # factory
    # NOTE: this is only used if I have twice as many heavies as the enemy
    if self._assault_rubble_points is None:
        points = []

        for f2 in self.game_board.agent.opponent_factories:
            if self.n_heavies > f2.n_heavies:
                f1_tiles = [f2.center]  # self.edge_tiles()
                f2_tiles = [self.center]  # closest.edge_tiles()

                for p1 in f1_tiles:
                    for p2 in f2_tiles:
                        candidates = get_optimal_path(self.game_board, p1, p2)

                        candidates = [
                            c
                            for c in candidates
                            if c.rubble > 0
                            and c not in points
                            and not c.factory
                            and not c.next_to_enemy_factory
                            and c.get_rubble_at_time(TTMAX) > 0
                        ]

                        points += candidates
        self._assault_rubble_points = points
    return self._assault_rubble_points


def get_lake_rubble_targets(self, MAX_DIST=15, MAX_RUBBLE=19, verbose=False):
    if verbose:
        self._get_lake_rubble_targets = None
    if self._get_lake_rubble_targets is None:

        board = self.game_board.board

        rb = (
            1.0
            * (board.rubble == 0)
            * (board.ice == 0)
            * (board.ore == 0)
            * (board.factory_occupancy_map < 0)
        )
        clusters = label(rb, connectivity=1)

        if verbose:
            lprint(f"Clusters: {np.unique(clusters)}")
            plt.imshow(
                (
                    clusters
                    + (np.max(clusters) + 5) * (board.factory_occupancy_map >= 0)
                ).T
            )
            plt.title(f"Clusters: {np.unique(clusters)}")
            plt.show()

        need_lake_map = np.zeros_like(board.rubble)

        need_lake_factories = [f for f in self.factories if not f.lake_large_enough()]

        if not need_lake_factories:
            self._get_lake_rubble_targets = []

        for f in need_lake_factories:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    need_lake_map[f.pos[0] + dx, f.pos[1] + dy] = 1

        lake_factories_expanded = expand_labels(need_lake_map > 0, 1)
        keep_clusters = [
            c for c in np.unique(clusters[lake_factories_expanded]) if c != 0
        ]

        my_lakes = np.isin(clusters, keep_clusters)

        clusters = my_lakes

        enemy_lichen = np.isin(board.lichen_strains, self.enemy_lichen_strains)
        enemy_factories = np.isin(board.factory_occupancy_map, [1 - self.team_id])
        enemy_lichen_growth = expand_labels(enemy_lichen + enemy_factories, 2)

        if verbose:
            lprint(f"Clusters: {np.unique(clusters)}")
            plt.imshow(
                (
                    clusters
                    + (np.max(clusters) + 5) * (board.factory_occupancy_map >= 0)
                ).T
            )
            plt.title(f"Clusters: {np.unique(clusters)}")
            plt.show()

        closer_to_me = (
            expand_labels(board.factory_occupancy_map + 1, MAX_DIST) - 1
        ) == self.team_id

        clusters = (clusters * 1.0 * (enemy_lichen_growth == 0) * closer_to_me) > 0
        boundaries = find_boundaries(clusters, mode="outer")

        candidates = (
            1.0
            * boundaries
            * closer_to_me
            * (board.rubble > 0)
            * (board.ice == 0)
            * (board.ore == 0)
            * (board.rubble <= MAX_RUBBLE)
            * (enemy_lichen_growth == 0)
        )

        if verbose:
            plt.imshow(
                (
                    clusters
                    + (15 + 5) * (board.factory_occupancy_map >= 0)
                    + 100 * 1.0 * boundaries
                ).T
            )
            plt.title(f"clusters ex enemy lichen growth: {np.unique(clusters)}")
            plt.show()

        if verbose:
            plt.imshow((candidates + 2 * (board.factory_occupancy_map >= 0)).T)
            plt.title(f"clusters ex enemy lichen growth: {np.unique(clusters)}")
            plt.show()

        self._get_lake_rubble_targets = self.game_board.grid_to_points(candidates > 0)
    return self._get_lake_rubble_targets
