import matplotlib.pyplot as plt
from skimage.measure import label
from scipy import ndimage
import numpy as np
from lux.utils import get_n_primes, cross_filter, lprint
from skimage.segmentation import expand_labels
from lux.constants import STEPS_THRESHOLD_LICHEN, POWER_THRESHOLD_LICHEN


def set_connected_lichen(agent, verbose=False):
    board = agent.game_state.board
    strains_map = board.lichen_strains.copy()

    if verbose:
        plot_strains = strains_map.copy()
        plt.imshow(plot_strains.T)
        plt.title("strains_map start")
        plt.show()

    factories = agent.all_factories.values()
    for f in factories:
        f.n_lichen_tiles = 0

    if np.max(board.lichen) == 0:
        agent.connected_lichen_map = np.zeros_like(board.lichen)
        return

    for f in factories:
        f.n_all_lichen_tiles = np.sum(strains_map == f.id)

    for f in factories:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                strains_map[f.pos[0] + dx, f.pos[1] + dy] = f.id

    if verbose:
        plot_strains = strains_map.copy()
        plt.imshow(plot_strains.T)
        plt.title("strains_map factories added")
        plt.show()

    clusters = label(1.0 * strains_map, connectivity=1, background=-1)
    cluster_nums = np.unique(clusters)

    factory_labels = {}
    for f in factories:
        factory_labels[clusters[f.pos[0], f.pos[1]]] = f

    # drop unconnected clusters
    for i, c in enumerate(cluster_nums):
        if c == 0:
            continue
        mask = clusters == c

        if c not in factory_labels:
            clusters[mask] = 0
        else:
            factory_labels[c].n_lichen_tiles = np.sum(mask) - 9  # remove factory tiles

    clusters = -1 * clusters
    background = clusters == 0
    for c, f in factory_labels.items():
        clusters[clusters == -c] = f.id

    strains_map = clusters
    strains_map[background] = -1

    if verbose:
        plt.imshow(strains_map.T)
        plt.title("strains_map connected")
        plt.show()

    agent.connected_lichen_map = strains_map.copy()


def set_lichen_choke_points(
    agent, min_cluster_size=3, max_tiles_per_choke_segment=3, verbose=False
):
    board = agent.game_state.board
    gb = agent.game_board

    strains_map = agent.connected_lichen_map.copy()

    agent.connected_incl_single_lichen_map = agent.connected_lichen_map.copy()
    if np.max(board.lichen) == 0:
        agent.choke_points_own = []
        agent.choke_points_enemy = []
        agent.choke_clusters_enemy = {}
        agent.choke_clusters_own = {}
        return

    kernel = cross_filter(3)
    kernel[1, 1] = 0

    # remove tiles with one neighbour
    num_neighbors = ndimage.convolve(1.0 * (strains_map >= 0), kernel, mode="constant")
    single_neighbour = num_neighbors == 1
    strains_map_original = strains_map.copy()
    connected_incl_single_lichen_map = strains_map.copy()
    strains_map[single_neighbour] = -1

    # do not remove adjacent to factory tiles
    single_neighbour_ex = (
        1.0
        * single_neighbour
        * (1 - expand_labels(agent.game_state.board.factory_occupancy_map != -1))
    )
    # plt.imshow(single_neighbour_ex)
    # plt.title("single_neighbour_ex")
    # plt.show()

    # connected_incl_single_lichen_map[single_neighbour_ex > 0] = -1
    # plt.imshow(connected_incl_single_lichen_map.T)
    # plt.title("connected_incl_single_lichen_map")
    # plt.show()

    agent.connected_incl_single_lichen_map = connected_incl_single_lichen_map

    # kernels that give posivite lichen candidates
    # fmt: off
    kernels = [
        # [[1, 0, 1],
        #  [0, 0, 0],
        #  [1, 0, 1]],
        # [[0, 1, 0],
        #  [1, 0, 1],
        #  [0, 1, 0]],

        # empty corners
        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [0, 0, 0],
         [1, 0, 0]],
        # empty sides
        [[0, 1, 0],
         [0, 0, 0],
         [0, 1, 0]],
        [[0, 0, 0],
         [1, 0, 1],
         [0, 0, 0]],

        # # horse
        [[1, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],
        [[0, 0, 1],
         [0, 0, 0],
         [0, 1, 0]],
        [[0, 0, 1],
         [1, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 1]],
        [[0, 1, 0],
         [0, 0, 0],
         [1, 0, 0]],
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 1],
         [1, 0, 0]],
    ]
    # fmt: on

    circle_kernel = np.ones((3, 3))
    circle_kernel[1, 1] = 0
    first_kernel = np.zeros((3, 3))
    first_kernel[0, 0] = 1

    # detect rings and fill them in to prevent false positives
    up_one = strains_map + 1
    value = ndimage.convolve(up_one, first_kernel, mode="constant")
    circle_conv = ndimage.convolve(up_one, circle_kernel, mode="constant")
    n_lichen = ndimage.convolve(1.0 * (up_one > 0), circle_kernel, mode="constant")
    circles = (
        1.0
        * ((value * 8) == circle_conv)
        * (n_lichen == 8)
        * (strains_map == -1)
        * (circle_conv > 0)
    ) > 0

    strains_map[circles] = value[circles] - 1
    if verbose:
        plot_hits = strains_map.copy()
        plot_hits[circles] = np.max(plot_hits) + 5
        plt.imshow(plot_hits.T)
        plt.title("circles")
        plt.show()

    # fmt: off

    inverse_exclude_kernels = [
        [[0, 0, 0],
         [1, 0, 1],
         [0, 1, 0]],
        [[0, 1, 0],
         [0, 0, 1],
         [0, 1, 0]],
        [[0, 1, 0],
         [1, 0, 1],
         [0, 0, 0]],
        [[0, 1, 0],
         [1, 0, 0],
         [0, 1, 0]],
         
        # CORNERS
        [[0, 0, 1],
         [0, 0, 1],
         [1, 1, 1]],
        [[1, 0, 0],
         [1, 0, 0],
         [1, 1, 1]],
        [[1, 1, 1],
         [1, 0, 0],
         [1, 0, 0]],
        [[1, 1, 1],
         [0, 0, 1],
         [0, 0, 1]],

        # CORNERS +
        [[0, 0, 0],
         [0, 0, 1],
         [1, 1, 1]],
        [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 1]],
        [[1, 0, 0],
         [1, 0, 0],
         [1, 1, 0]],
        [[1, 1, 0],
         [1, 0, 0],
         [1, 0, 0]],
        [[0, 1, 1],
         [0, 0, 1],
         [0, 0, 1]],
        [[1, 1, 1],
         [1, 0, 0],
         [0, 0, 0]],
        [[1, 1, 1],
         [0, 0, 1],
         [0, 0, 0]],
    ]
    # fmt: on

    lichen_tiles = strains_map >= 0
    no_lichen = strains_map < 0

    include_map = board.factory_occupancy_map == -1
    # for kernel in exclude_kernels:
    #     num_hits = ndimage.convolve(1.0 * (strains_map >= 0), kernel, mode="constant")
    #     include_map[num_hits == np.sum(kernel)] = 0

    for kernel in inverse_exclude_kernels:
        num_hits = ndimage.convolve(1.0 * no_lichen, kernel, mode="constant", cval=1.0)
        include_map[num_hits == np.sum(kernel)] = 0

    hits = np.zeros_like(lichen_tiles)
    for kernel in kernels:
        hit = ndimage.convolve(1.0 * no_lichen, kernel, mode="constant", cval=1.0)
        hits = hits + 1.0 * (hit >= 2)

    candidates = 1.0 * (hits > 0) * lichen_tiles * include_map

    if verbose:
        plot_hits = plot_strains.copy()
        plot_hits[candidates > 0] = np.max(plot_hits) + 5
        plt.imshow(plot_hits.T)
        plt.title("candidates")
        plt.show()

    strains_ex_choke_points = strains_map.copy()
    strains_ex_choke_points[candidates > 0] = -1

    # calculate lichen fields without the choke points, fields that change label are cut off
    clusters = label(strains_ex_choke_points, connectivity=1, background=-1)
    cluster_nums = np.unique(clusters)
    new_nums = get_n_primes(len(cluster_nums))
    new_clusters = clusters.copy()

    # drop too small clusters and add prime ids
    for i, c in enumerate(cluster_nums):
        if c == 0:
            continue
        mask = clusters == c
        num_tiles = np.sum(mask)
        if num_tiles + np.sum(mask) <= min_cluster_size:
            new_clusters[mask] = 0
        else:
            new_clusters[mask] = new_nums[i]

    if verbose:
        plt.imshow(new_clusters.T)
        plt.title("new_clusters")
        plt.show()

    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    # add candidates to clusters and do different connection scan.
    # todo: this does keep too small clusters that are connected via 2 choke points
    # compare_map = 1.0 * (new_clusters + (max(new_nums) + 1) * candidates)
    # if verbose:
    #     plt.imshow(compare_map.T)
    #     plt.title("compare_map")

    #     plt.show()
    # num_neighbors = ndimage.convolve(1.0 * (compare_map > 0), kernel, mode="constant")
    # sum_neighbors = ndimage.convolve(compare_map, kernel, mode="constant")
    # num_neighbors[num_neighbors == 0] = 1
    # avg_neighbors = sum_neighbors / num_neighbors
    # multi_neighbors = (avg_neighbors % 1) > 0

    # connect the candidates that form a cluster
    clusters = label(candidates, connectivity=1, background=0)
    for c in np.unique(clusters):
        if c == 0:
            continue

        mask = clusters == c
        num_tiles = np.sum(mask)
        # if num_tiles <= 2:
        #     clusters[mask] = 0

    choke_points = 1.0 * candidates  # * multi_neighbors

    choke_clusters = np.unique(clusters[choke_points > 0])

    # check the impact of dropping it
    for c in choke_clusters:
        if c == 0:
            continue
        mask = clusters == c
        choke_points[mask] = 1

    clusters = label(choke_points, connectivity=1, background=0)

    kill_map_own = np.zeros_like(strains_map)
    kill_map_enemy = np.zeros_like(strains_map)

    choke_points_own = {}
    choke_points_enemy = {}
    choke_clusters_own = {}
    choke_clusters_enemy = {}

    steps_left = gb.steps_left
    for c in np.unique(clusters):
        if c == 0:
            continue

        mask = clusters == c

        sub_strains = strains_map_original.copy()
        choke_strain = sub_strains[mask][0]
        # select only the current cluster
        sub_strains[sub_strains != choke_strain] = -1
        # take out the choke cluster
        sub_strains[mask] = -1
        sub_clusters = label(sub_strains, connectivity=1, background=-1)

        unique_values, value_counts = np.unique(sub_clusters, return_counts=True)

        for f in gb.factories:
            p = f.center
            factory_cluster = sub_clusters[p.x, p.y]
            if factory_cluster > 0:
                break

        # if (
        #     f.is_enemy
        #     and f.power > POWER_THRESHOLD_LICHEN
        #     and steps_left > STEPS_THRESHOLD_LICHEN
        # ):
        #     continue

        # print the results
        kill_count = np.sum(mask)
        for value, count in zip(unique_values, value_counts):
            if value not in [0, factory_cluster]:
                kill_count += count

        def update_choke_points(
            cluster_choke_points, kill_count, kill_map, choke_clusters
        ):
            choke_clusters[c] = (kill_count, cluster_choke_points)
            kills_on_point = (
                kill_count  # closest is first, so any next point is one less kill
            )
            for p in cluster_choke_points:
                choke_points_own[p] = kill_count
                p.choke_kills = kills_on_point
                kills_on_point -= 1

        if kill_count >= min_cluster_size + 1:
            cluster_choke_points = gb.grid_to_points(mask)
            cluster_choke_points = sorted(cluster_choke_points, key=f.distance_to)[
                :max_tiles_per_choke_segment
            ]

            if f.is_own:
                update_choke_points(
                    cluster_choke_points, kill_count, kill_map_own, choke_clusters_own
                )
            else:
                update_choke_points(
                    cluster_choke_points,
                    kill_count,
                    kill_map_enemy,
                    choke_clusters_enemy,
                )
            if verbose:
                lprint(
                    f"kill_count {cluster_choke_points} f.is_own {f.is_own}:",
                    kill_count,
                )

    if verbose:
        plot_strains[kill_map_own > 0] = np.max(strains_map_original) + 2
        plot_strains[kill_map_enemy > 0] = np.max(strains_map_original) + 2

        plt.imshow((plot_strains).T)
        plt.show()

    agent.choke_points_own = choke_points_own
    agent.choke_points_enemy = choke_points_enemy
    agent.choke_clusters_own = choke_clusters_own
    agent.choke_clusters_enemy = choke_clusters_enemy


def get_lichen_targets(self, verbose=False):
    if verbose:  # for debugging
        self._lichen_targets = None
    if self._lichen_targets is None:
        board = self.game_state.board
        lichen_strains = board.lichen_strains.copy()
        lichen_ids = [i for i in np.unique(lichen_strains) if i != -1]

        if not lichen_ids:
            return []

        connected_incl_single_lichen_map = self.connected_incl_single_lichen_map.copy()
        enemy_ids = []
        for f in self.opponent_factories:
            center = f.center
            enemy_ids.append(connected_incl_single_lichen_map[center.x, center.y])

        # connected lichen, excluding single connected lichen
        connected_incl_single_lichen_map[board.factory_occupancy_map >= 0] = -1

        enemy_lichen_mask = np.isin(connected_incl_single_lichen_map, enemy_ids)
        if verbose:
            plt.imshow(enemy_lichen_mask.T)
            plt.title("enemy_lichen_mask")
            plt.show()

        # unconnected lichen is also a valid target in
        steps_left = self.game_board.steps_left
        if steps_left < 100:
            enemy_lichen_all_mask = np.isin(
                board.lichen_strains, self.enemy_lichen_strains
            )
            enemy_lichen_all = np.logical_and(
                board.lichen > steps_left, enemy_lichen_all_mask
            )

            enemy_lichen_mask = np.logical_or(enemy_lichen_mask, enemy_lichen_all)

            if verbose:
                plt.imshow(enemy_lichen_mask.T)
                plt.title("enemy_lichen_mask with disconnected lichen")
                plt.show()
        targets = self.game_board.grid_to_points(enemy_lichen_mask)

        self._lichen_targets = targets

    return self._lichen_targets


def get_lichen_connector_points(agent, verbose=False):

    gb = agent.game_board

    # find connected lichen
    connected_lichen = agent.connected_lichen_map
    if verbose:
        plt.imshow(connected_lichen.T)
        plt.title("connected_lichen")
        plt.show()

    board = gb.game_state.board
    rb = (
        1.0
        * (board.rubble == 0)
        * (board.ice == 0)
        * (board.ore == 0)
        #     * (board.factory_occupancy_map < 0)
        #     * (connected_lichen == -1)
    )
    if verbose:
        plt.imshow(rb.T)
        plt.title("rb")
        plt.show()
    clusters = label(rb, connectivity=1)
    clusters = clusters * 1.0 * (board.factory_occupancy_map < 0)
    # clusters = clusters * 1.0 *(connected_lichen == -1)
    cluster_nums = np.unique(clusters)
    new_nums = get_n_primes(len(cluster_nums))
    new_clusters = clusters.copy()

    # drop too small clusters and add prime ids
    for i, c in enumerate(cluster_nums):
        if c == 0:
            continue
        mask = clusters == c
        num_tiles = np.sum(mask)
        if num_tiles <= 4:
            new_clusters[mask] = 0
        else:
            new_clusters[mask] = new_nums[i]

    if verbose:
        plt.imshow(new_clusters.T)
        plt.title("new empty clusters")
        plt.show()

    # find points that connect multiple clusters of rubble
    kernel = np.ones((7, 7))
    num_neighbors = ndimage.convolve(1.0 * (new_clusters > 0), kernel, mode="constant")
    sum_neighbors = ndimage.convolve(1.0 * new_clusters, kernel, mode="constant")
    num_neighbors[num_neighbors == 0] = 1
    avg_neighbors = sum_neighbors / num_neighbors
    multi_neighbors = (avg_neighbors % 1) > 0

    # add check for at least one of them without rubble
    rb_free = rb * (board.factory_occupancy_map < 0) * (connected_lichen == -1)
    rb_free_clusters = 1.0 * (new_clusters > 0) * rb_free
    num_neighbors_free = ndimage.convolve(rb_free_clusters, kernel, mode="constant")
    free_mask = num_neighbors_free > 0

    # multi_neighbors = num_neighbors > 0
    if verbose:
        plt.imshow(multi_neighbors.T)
        plt.title("multi_neighbors")
        plt.show()

    candidates = 1.0 * multi_neighbors * (connected_lichen >= 0) * free_mask

    if verbose:
        plt.imshow(
            (
                (board.rubble == 0)
                + connected_lichen
                + 5 * np.max(connected_lichen) * candidates
            ).T
        )
        plt.title("candidates")
        plt.show()

    return gb.grid_to_points(candidates > 0)
