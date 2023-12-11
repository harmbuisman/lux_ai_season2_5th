import numpy as np
from scipy.ndimage import convolve
from lux.early_game import cross_filter


def get_solar_targets(self, verbose=False):
    if self._solar_targets is None:
        gb = self.game_board
        heavy_board = np.zeros_like(gb.board.lichen)
        power_board = np.zeros_like(gb.board.lichen)

        for u in self.opponent_heavies:
            power_board[u.point.xy] += u.power
            if u.point.factory is None and len(set(u.path)) == 1 and not u.dies:
                heavy_board[u.point.xy] = 1

        # covered_board = np.zeros_like(gb.board.lichen)
        # for u in self.heavies:
        #     covered_board[u.last_point.xy] = 1

        # covered_map = convolve(covered_board, cross_filter(3), mode="constant", cval=0.0)

        target_map = convolve(heavy_board, cross_filter(3), mode="constant", cval=0.0)
        power_map = convolve(power_board, cross_filter(3), mode="constant", cval=0.0)
        target_map[target_map == 1] = 0

        target_map[gb.board.factory_occupancy_map == 1 - self.team_id] = 0

        targets = gb.grid_to_points(target_map > 0)

        covered = [u.last_point for u in self.heavies]

        targets = set(targets) - set(covered)

        target_dict = {t: power_map[t.xy] for t in targets}

        self._solar_targets = target_dict
    return self._solar_targets
