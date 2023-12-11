import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.filters import maximum_filter
from skimage.measure import label
from skimage.segmentation import expand_labels

from lux.board import GameBoard
from lux.utils import cross_filter, lprint

ICE_SIGMA = 4
ORE_SIGMA = 4
CROWDING_SIGMA = 4
RUBBLE_FACTOR = 1
RUBBLE_THRESHOLD = 25  # more than this rubble is penalized for ice
RESERVE_ICE_POINTS = 2
RESERVE_ORE_POINTS = 2
HEAVY_THRESHOLD = 50


# def heatmap(grid, title="", t=0, arrow_points=None, fmt=".0f"):
#     IS_KAGGLE = os.path.exists("/kaggle")

#     if IS_KAGGLE:
#         return  # never plot when in kaggle

#     grid = np.transpose(grid)

#     _, ax = plt.subplots(figsize=(12, 10))  # (24, 20))

#     # x and y are switched in matrices
#     # plt.imshow(
#     #     grid,
#     # )
#     sns.heatmap(
#         grid,
#         annot=True,
#         ax=ax,
#         fmt=fmt,
#     )
#     # plt.grid(True, which="both")
#     plt.tight_layout()

#     if arrow_points is not None:
#         from_p, to_p = arrow_points
#         xytext = from_p if isinstance(from_p, tuple) else from_p.xy
#         xy = to_p if isinstance(to_p, tuple) else to_p.xy
#         ax.annotate(
#             "",
#             xytext=(xytext[0] + 0.5, xytext[1] + 0.5),
#             xy=(xy[0] + 0.5, xy[1] + 0.5),
#             arrowprops=dict(facecolor="white"),
#             color="white",
#         )

#     step = 0
#     t_plot = step + 1 + t if isinstance(t, int) else [step + 1 + x for x in t]

#     plt.title(f"{step} + t{t} = {t_plot} " + title)
#     plt.show()


def rubble_cluster_count(game_state):
    rb = (
        1.0
        * (game_state.board.rubble == 0)
        * (game_state.board.ice == 0)
        * (game_state.board.ore == 0)
        * (game_state.board.factory_occupancy_map < 0)
    )
    clusters = label(rb, connectivity=1)
    counts = {i: np.sum(clusters == i) for i in range(1, np.max(clusters) + 1)}
    largest_clusters = {
        k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])
    }

    value_grid = np.zeros_like(clusters)
    for c, v in largest_clusters.items():
        value_grid += maximum_filter(v * (clusters == c), 5, mode="constant", cval=0.0)

    return value_grid


def has_inbalanced_touch_both(gb, n_factories_per_team, verbose=False):
    board = gb.board

    ice = board.ice
    ore = board.ore
    touch_grid_ore = convolve(ore * 1.0, cross_filter(5), mode="constant", cval=0.0)
    touch_grid_ice = convolve(ice * 1.0, cross_filter(5), mode="constant", cval=0.0)

    N = 48
    invalid = convolve(ice + ore, np.ones((3, 3)), mode="constant", cval=0.0)
    invalid = np.where(invalid > 0, 1, 0)
    invalid[:, 0] = 1
    invalid[:, N - 1] = 1
    invalid[0, :] = 1
    invalid[N - 1, :] = 1

    touch_grid_both = np.logical_and(touch_grid_ice > 0.5, touch_grid_ore > 0.5)

    valid_touch = np.logical_and(invalid < 0.5, touch_grid_both)

    found = 0
    for i in range(12):
        if verbose:
            print(i, np.sum(valid_touch))
            plt.imshow(valid_touch.T)
            plt.show()

        points = gb.grid_to_points(valid_touch)
        if points:
            found += 1
            fcenter = points[0]

            for p in fcenter.points_within_distance(6):
                valid_touch[p.xy] = 0
        else:
            break

    if found < n_factories_per_team * 2 and found % 2 == 1:
        return True
    return False


def best_positions(gb: GameBoard):  # , player, factory_positions):
    game_state = gb.game_state
    rubble = game_state.board.rubble.copy()
    rubble[game_state.board.factory_occupancy_map >= 0] = 100
    ore = game_state.board.ore.copy()
    ice = game_state.board.ice.copy()

    rubble_factor = (
        1 - np.maximum(0, (rubble - RUBBLE_THRESHOLD) / 101)
    ) ** RUBBLE_FACTOR

    inverse_rubble = (100 - rubble) / 101

    n_factories_per_team = gb.agent.n_factories_per_team
    must_ore = False
    if n_factories_per_team:

        n_placed = len([f for f in gb.factories if f.is_own])
        n_ore_risk = len([f for f in gb.factories if f.is_own and f.is_ore_risk()])
        last_factory = n_placed > 0 and n_placed == n_factories_per_team - 1
        must_ore = n_factories_per_team > 2 and n_ore_risk == n_placed and last_factory

        lprint(
            "gb.agent.n_factories_per_team:",
            n_factories_per_team,
            "n_factories_placed:",
            n_placed,
            "n_ore_risk:",
            n_ore_risk,
            "last_factory:",
            last_factory,
            "must_ore:",
            must_ore,
        )

    # RESERVE ICE
    free_ice = ice.copy().astype(float)
    free_ore = ore.copy().astype(float)

    # exclude rss encapsulated by other rss
    for point in gb:
        if point.ice and all(
            [p.ice + p.ore > 0 for p in point.points_within_distance(1)]
        ):
            free_ice[point.x, point.y] = 0.1
        if point.ore and all(
            [p.ice + p.ore > 0 for p in point.points_within_distance(1)]
        ):
            free_ore[point.x, point.y] = 0.1

        # exclude ice and ore that have more than 2 next to it (3=self incl.)
        if point.ice and np.sum([p.ice for p in point.points_within_distance(1)]) > 3:
            free_ice[point.x, point.y] = 0.1
        if point.ore and np.sum([p.ore for p in point.points_within_distance(1)]) > 3:
            free_ore[point.x, point.y] = 0.1

    # need to use factory positions as well. perhaps better so we do not
    # have to deal with calculating this during game time
    for factory in gb.factories:
        MAX_BLOCK_DISTANCE_OWN = 5
        MAX_BLOCK_DISTANCE_ENEMY = 2
        # block the two closest ice points
        points_to_block = sorted(
            factory.ice_points_by_distance, key=factory.distance_to
        )[:RESERVE_ICE_POINTS]
        for p in points_to_block:
            if factory.distance_to(p) > (
                MAX_BLOCK_DISTANCE_OWN if factory.is_own else MAX_BLOCK_DISTANCE_ENEMY
            ):
                continue
            free_ice[p.x, p.y] = min(free_ice[p.x, p.y], 0.1 * factory.distance_to(p))

        # block any ice points within 2 of a factory
        for p in factory.points_within_distance(2):
            free_ice[p.x, p.y] = min(free_ice[p.x, p.y], 0.1 * factory.distance_to(p))

        # block the two closest ore points
        points_to_block = sorted(
            factory.ore_points_by_distance, key=factory.distance_to
        )[:RESERVE_ORE_POINTS]
        for p in points_to_block:
            if factory.distance_to(p) > (
                MAX_BLOCK_DISTANCE_OWN if factory.is_own else MAX_BLOCK_DISTANCE_ENEMY
            ):
                continue
            free_ore[p.x, p.y] = min(free_ice[p.x, p.y], 0.1 * factory.distance_to(p))

        # # block any ore points within  1 of a factory
        for p in factory.points_within_distance(2):
            free_ore[p.x, p.y] = min(free_ice[p.x, p.y], 0.1 * factory.distance_to(p))

    # add mini constant to allow to still select a position if one of the factors = 0
    epsilon_ice = 0.5
    ice_score = free_ice.astype(float) * rubble_factor + epsilon_ice

    ice_score_s = gaussian_filter(ice_score, ICE_SIGMA, mode="constant")

    ice_score_s = ice_score_s / (ice_score_s.max() + epsilon_ice)
    ice_score_s[ice_score_s > 0.5] = 0.5
    # heatmap(ice_score_s, "ice_score_s", 0)

    ore_epsilon = 0.1
    ore_score = free_ore.astype(float) * rubble_factor + ore_epsilon

    ore_score_s = gaussian_filter(ore_score, ORE_SIGMA, mode="constant")
    ore_score_s = ore_score_s / ore_score_s.max() + ore_epsilon
    # heatmap(ore_score_s, "ore_score_s", 0)

    touch_epsilon_ice = 0.0001
    touch_grid_ice = np.sqrt(
        np.clip(
            convolve(free_ice, cross_filter(5), mode="constant", cval=0.0),
            a_min=0,
            a_max=2.1,
        )
    )
    # heatmap(touch_grid_ice, "touch_grid_ice", 0)
    # touch_grid_ice[touch_grid_ice > 1] = 1
    touch_grid_ice = (touch_grid_ice + touch_epsilon_ice) / (1 + touch_epsilon_ice)
    # plt.imshow(touch_grid_ice.T)
    # plt.show()
    # heatmap(touch_grid_ice, "touch_grid_ice", 0)

    # there must be two ice or a touching ice
    epsilon_ice_must = 0.0001
    ice_must_grid = convolve(free_ice, cross_filter(7), mode="constant", cval=0.0)
    # heatmap(ice_must_grid, "ice_must_grid", 0)
    # heatmap(touch_grid_ice, "touch_grid_ice", 0)
    ice_must_grid = np.where(
        np.logical_or(ice_must_grid >= 1.5, touch_grid_ice > 0.5), 1, epsilon_ice_must
    )

    # heatmap(ice_must_grid, "ice_must_grid", 0)
    touch_epsilon_ore = 0.00001  # 1
    touch_grid_ore = convolve(free_ore, cross_filter(5), mode="constant", cval=0.0)
    touch_grid_ore[touch_grid_ore > 1] = 1
    touch_grid_ore = (touch_grid_ore + touch_epsilon_ore) / (1 + touch_epsilon_ore)

    touch_grid = convolve(
        free_ice + free_ore, cross_filter(5), mode="constant", cval=0.0
    )
    lprint(f"free ice: {np.max(free_ice)}")
    # touch_grid[touch_grid > 1] = 1
    invalid = convolve(ice + ore, np.ones((3, 3)), mode="constant", cval=0.0)
    invalid = np.where(invalid > 0, 1, 0)

    touch_grid_ice_and_ore_epsilon = 0.000001
    touch_grid = touch_grid * (1 - invalid)
    touch_grid = (touch_grid + touch_grid_ice_and_ore_epsilon) / (
        1 + touch_grid_ice_and_ore_epsilon
    )
    # heatmap(touch_grid, "touch_grid")

    # touch_grid_both[touch_grid_both > 0.5] = 1
    # 5 x 5 convolution filter with the corners at 0

    ### DISTANCE FROM ICE
    # make sure that we are in the neighborhood of a factory with ice
    ice_expanded = ice.copy()
    ice_expanded = expand_labels(ice_expanded, 18)
    ice_expanded_epsilon = 0
    ice_expanded = (1.0 * ice_expanded + ice_expanded_epsilon) / (
        1 + ice_expanded_epsilon
    )

    ### DISTANCE FROM ORE
    # make sure that we are in the neighborhood of a factory with ore
    ore_expanded = ore.copy()
    ore_expanded = expand_labels(ore_expanded, 9)
    ore_expanded_epsilon = 0.01
    ore_expanded = (1.0 * ore_expanded + ore_expanded_epsilon) / (
        1 + ore_expanded_epsilon
    )

    ore_exp = gb.board.ore
    score_ore = gb.board.ore.copy()
    MAX_DIST = 12
    for i in range(MAX_DIST):
        ore_exp = expand_labels(ore_exp, 1)
        score_ore += ore_exp
    score_ore = (1.0 * score_ore ** (2) + ore_expanded_epsilon) / (
        1 + ore_expanded_epsilon
    )

    ### DISTANCE FROM EDGE
    # penalty if close to the edge
    N = 48
    # Create an array with the coordinates of each point in the grid
    x = np.arange(N)
    y = np.arange(N)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distance from the center of the grid
    # Calculate the distance from the edge of the grid
    # Calculate the distance from the edge of the grid
    dist_x = np.minimum(xx, N - xx - 1)
    dist_y = np.minimum(yy, N - yy - 1)

    dist_treshold = 2

    dist_from_edge = np.power(
        (dist_treshold - np.maximum(dist_treshold - np.minimum(dist_x, dist_y), 0))
        / dist_treshold,
        2,
    )
    dist_from_edge = dist_from_edge + 0.0000001

    # add removal for existing positions
    spawns = game_state.board.valid_spawns_mask

    # rubble
    rubble_epsilon = 0.05
    SIZE = 9
    FACTOR = 2
    inverse_rubble_score = convolve(
        inverse_rubble**FACTOR, cross_filter(SIZE, remove_center=True), mode="nearest"
    )
    inverse_rubble_score = inverse_rubble_score / inverse_rubble_score.max()
    inverse_rubble_score = (inverse_rubble_score + rubble_epsilon) / (
        1 + rubble_epsilon
    )

    # don't care about rubble lakes if no ice available
    touch_epsilon_both = 0.05
    if must_ore or ice_must_grid.max() < 0.5:
        touch_grid_ice = np.ones_like(touch_grid_ice)
        rubble_score = np.ones_like(inverse_rubble_score)
        # make touching ore super important
        touch_grid_both = touch_grid_ore

        # ore only should be close to own factories
        own_team = 1.0 * (gb.board.factory_occupancy_map == gb.agent.team_id)
        enemy = 1.5 * (gb.board.factory_occupancy_map == (1 - gb.agent.team_id))

        for f in gb.factories:
            if f.is_own:
                lprint(f">>>>>>>>ore risk at {f}", f.ice_hubs, f.is_ore_risk())
                score = (
                    min(len(f.ice_hubs), 2.1) * (1 + f.is_ore_risk())
                    - 2 * f.is_ice_risk()
                )

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        own_team[f.pos[0] + dx, f.pos[1] + dy] = score

        epsilon_crowding = 0.001
        crowding_score = gaussian_filter(
            own_team, CROWDING_SIGMA, mode="constant"
        ) - gaussian_filter(enemy, CROWDING_SIGMA, mode="constant")
        crowding_score = crowding_score - np.min(crowding_score)
        crowding_score = (crowding_score + epsilon_crowding) / (
            crowding_score.max() + epsilon_crowding
        )

    else:
        rubble_epsilon = 0.8
        rubble_lake = rubble_cluster_count(gb.game_state)
        rubble_score = rubble_lake / rubble_lake.max()
        rubble_score = (rubble_score + rubble_epsilon) / (1 + rubble_epsilon)
        crowding_score = np.ones_like(inverse_rubble_score)

        touch_grid_both = (
            np.logical_and(touch_grid_ice > 0.5, touch_grid_ore > 0.5)
            + touch_epsilon_both
        ) / (1 + touch_epsilon_both)

    score = (
        (ore_score_s * ice_score_s)
        * (1 - ice)
        * (1 - ore)
        * ice_must_grid
        * touch_grid_ice
        * touch_grid
        * touch_grid_both
        * dist_from_edge
        # * ice_expanded
        * ore_expanded
        * score_ore
    )

    score[score == 0] = -1000

    violations = convolve(score, np.ones((3, 3)), mode="constant", cval=0.0)
    score[violations < 0] = 0

    # here must update occupancy map
    occ_map = game_state.board.factory_occupancy_map.copy()
    # for i, j in factory_positions:
    #     point = gb.get_point((i, j))
    #     for x, y in point.surrounding_points() + [point.xy]:
    #         occ_map[x, y] = 1

    factory_grid = (occ_map != -1).astype(float)

    # epsilon_crowding = 0.3
    # if factory_grid.max() == 1:
    #     crowding_score = gaussian_filter(factory_grid, CROWDING_SIGMA, mode="constant")
    #     crowding_score = 1 - crowding_score / crowding_score.max() + epsilon_crowding
    # else:

    factory_block_grid = convolve(
        factory_grid, np.ones((3, 3)), mode="constant", cval=0.0
    )

    final_score = (
        score
        * ice_expanded
        * spawns
        * crowding_score
        * inverse_rubble_score
        * rubble_score
    )
    final_score = final_score * (factory_block_grid == 0).astype(int)

    debug = False
    if debug:
        for p in [(41, 23), (6, 15)]:
            for i, ggg in enumerate(
                [
                    f"ice_score_s: {ice_score_s[p]}",
                    f"ore_score_s: {ore_score_s[p]}",
                    f"ice_must_grid: {ice_must_grid[p]}",
                    f"touch_grid_ice: {touch_grid_ice[p]}",
                    f"touch_grid: {touch_grid[p]}",
                    # touch_grid_both,
                    f"dist_from_edge: {dist_from_edge[p]}",
                    f"ice_expanded: {ice_expanded[p]}",
                    f"ore_expanded: {ore_expanded[p]}",
                    f"score_ore: {score_ore[p]}",
                    f"score: {score[p]}",
                    f"spawns: {spawns[p]}",
                    f"crowding_score: {crowding_score[p]}",
                    f"inverse_rubble_score: {inverse_rubble_score[p]}",
                    f"rubble_score: {rubble_score[p]}",
                    f"final_score: {final_score[p]}",
                    f"violations: {violations[p]}",
                ]
            ):
                lprint(p, i, ggg)

        plt.imshow(score.T)
        plt.show()

        plt.imshow(final_score.T)
        plt.show()
    # heatmap(final_score, "score")
    # heatmap(final_score, "final_score")

    return final_score


def get_best_spawn(game_board):  # , factory_positions):
    score = best_positions(game_board)  # , factory_positions)
    pos = np.unravel_index(score.argmax(), score.shape, order="C")
    return pos


def get_factory_action(game_board, player):
    game_state = game_board.game_state
    # how much water and metal you have in your starting pool to give to new factories
    water_left = game_state.teams[player].water
    metal_left = game_state.teams[player].metal

    lprint(f"Metal:{metal_left}, Water:{water_left}")

    # how many factories you have left to place
    factories_to_place = game_state.teams[player].factories_to_place

    if factories_to_place == 0:
        return dict()

    spawn_loc = get_best_spawn(game_board)

    if factories_to_place == 1:
        metal = metal_left
        water = water_left
    else:
        metal = (metal_left // factories_to_place) // 10 * 10
        water = (water_left // factories_to_place) // 10 * 10
    return dict(spawn=list(spawn_loc), metal=metal, water=water)
