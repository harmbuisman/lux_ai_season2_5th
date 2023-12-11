import os
from collections import defaultdict
from enum import IntEnum, Enum

from lux.config import EnvConfig

# SETTINGS
TMAX = 20
TTMAX = 50


# suppose non-stop mining and one travelling
# 5 power + 1 move = 6 per light

# production = 0.6, so 5.4 per light
# 50 power supports
# 50/5.4 -->9.3 * 2
# = 18 bots
LICHEN_SHIELD_COVERAGE = 3

BARE_SUBMISSION = False
HEAVY_ONLY_ICE = BARE_SUBMISSION
LIGHT_ONLY_RUBBLE = BARE_SUBMISSION
BID_ONE = BARE_SUBMISSION
ONLY_LICHEN_RUBBLE = BARE_SUBMISSION

MAX_UNIT_PER_FACTORY = 2 if BARE_SUBMISSION else 30
LIGHT_TO_HEAVY_RATIO = 8
MIN_LIGHT_PER_FACTORY = 6

BUILD_HEAVY = True
BUILD_LIGHT = True
COMBAT_BUFFER_SIZE = 50000
POWER_ENEMY_FACTORY_MAGIC_VALUE = 10000
MICRO_PENALTY_POWER_COST = 0.001
MAX_BID = 10
MIN_ICE_START = 4
MIN_LICHEN_POWER_COUNT = (
    2  # from which amount of lichen we start adding power to the factory
)
DIGS_TO_TRANSFER_ICE = 25  # 13  # 25 is a full heavy and would provide more rss, but needs more water updates now and then
DIGS_TO_TRANSFER_ORE = 25
BASE_DIG_FREQUENCY_ORE = 3
BASE_DIG_FREQUENCY_ORE_PUSH = 5
BASE_DIG_FREQUENCY_ICE = 1
WAIT_TIME_ICE = 3
EARLY_GAME_MAX_LICHEN_TILES_PER_HUB = 50
UNPLAN_ORE_OUT_OF_WATER_TIME = 40
REMOVE_FINAL_HIT_FROM_QUEUE = False  # True

MIN_DONOR_WATER_FACTORY = 75

BASE_COMBAT_LIGHT = 1000
BASE_COMBAT_HEAVY = 10000

POWER_THRESHOLD_LICHEN = 6000000  # turned off
STEPS_THRESHOLD_LICHEN = 150

BREAK_EVEN_ICE_TILES = 13
MAX_COMBINE_TARGET_DISTANCE = 5
IS_KAGGLE = not os.path.exists("dumps")
VALIDATE = False  # not IS_KAGGLE

WATER_PUSH_TIME = 200  # when to start watering

ALL_DIRECTIONS = [0, 1, 2, 3, 4]
OPPOSITE_DIRECTION = {1: 3, 2: 4, 3: 1, 4: 2}

MOVES = {
    1: (0, -1),  # "up",
    2: (1, 0),  # "right",
    3: (0, 1),  # "down",
    4: (-1, 0),  # "left",
}

DIRECTION_ACTION_MAP = defaultdict(lambda: (0, 0), MOVES)

MAX_LOW_POWER_DISTANCE = 12
MAX_ADJACENT_CHASE_INTERCEPT = 5


class RECHARGE(IntEnum):
    REPLAN = 147
    REPLAN_HEAVY = 2947
    DEFEND = 150
    AMBUSH = 3001
    AMBUSH_RSS = 1889
    ATTACK_NO_FOLLOW = 2998
    RETREAT = 1
    CLOSE_FACTORY = 2999
    KILLNET = 2997
    HEAVY_SHIELD = 2989
    LIGHT_SHIELD = 149
    GUARD = 2996


RESOURCE_MAP = {
    "ice": 0,
    "ore": 1,
    "water": 2,
    "metal": 3,
    "power": 4,
}

# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
# Agents submit actions for robots, overwrite their action queues
# Digging, self-destruct actions (removing and adding rubble)
# Robot Building
# Movement and recharge actions execute, then collisions are resolved
# Factories that watered their tiles grow lichen
# Transfer resources and power
# Pickup resources and power (in order of robot id)
# Factories refine resources
# Power gain (if started during day for robots)
EXECUTION_ORDER = {0: 1, 1: 2, 2: 3, 3: 0, 4: 0, 5: 1}

MUST_RETREAT_UP_TO_STEPS_LEFT = 25

# Efficiency
# LIGHT 14 1.0 150 30 155.0 31.0 1.194
MAX_EFFICIENT_LIGHT_DISTANCE = 15
POWERHUB_BEAM_DISTANCE = 2
POWERHUB_OREFACTORY_BEAM_DISTANCE = 1
HIDE_LICHEN_DIG_DISTANCE = 2
HIDE_ATTACK_DISTANCE = 2

# rubble priorities
class PRIORITY(IntEnum):
    CLOSEST_ORE_CLOSE = 13
    LAKE_CONNECTOR_PRIO = 12
    CLOSEST_ORE = 11
    KICK_START_SURROUNDING_ICE = 10
    LAKE_CONNECTOR = 9
    RSS_CONNECTOR = 7
    LICHEN_0_10 = 8
    SURROUNDING_ICE = 8
    LICHEN_10_20 = 7
    RSS_CONNECTOR_TOUGH = 6
    LAKE_CONNECTOR_LOW_PRIO = 6
    POWER_HUB_POSITION = 14
    HIGH_VISIT = 6
    SURROUNDING_ICE_TOUGH = 5
    TRANSPORT_SUPPORT = 6
    TRANSPORT = 6
    LAKE_CLEARING = 6
    LICHEN_20_30 = 6
    LICHEN_30_40 = 5
    SURROUNDING_NON_ICE = 5
    ASSAULT = 4
    LICHEN_40_50 = 4
    LICHEN_50_60 = 3
    LICHEN_60_80 = 2
    LICHEN_80_100 = 2
    LICHEN_NOTHING_TO_DO = 1


MAX_CHARGE_FACTORY_DISTANCE = 12

MIN_NORMAL_RUBBLE_PRIORITY = 6

RADAR_DISTANCE = 3


class STATE(IntEnum):
    START = 0
    PREPARED = 1
    ON_TARGET = 2
    TARGET_ACHIEVED = 3
    AT_DESTINATION = 4
    COMPLETE = 5


class ACTION(Enum):
    WAIT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    CHARGE = 5
    DIG = 6
    PICKUP = 7
    TRANSFER = 8
    SELF_DESTRUCT = 9


DIRECTION_MAP = {
    "center": 0,
    "up": 1,
    "right": 2,
    "down": 3,
    "left": 4,
}


# OTHER_ACTIONS = {
#     0: (0, 0),  # "center",
#     ACTION.DIG: (0, 0),  # "stay",
#     "RECHARGE": (0, 0),  # "stay",
#     "PICKUP": (0, 0),  # "stay",
#     ACTION.TRANSFER: (0, 0),  # "stay",
# }

coords = []
for x in range(-2, 3):
    for y in range(-2, 3):
        if abs(x) == 2 and abs(y) == 2:
            continue
        coords.append((x, y))
COORD_MAP = {coord: i for i, coord in enumerate(coords)}
NUM_FEATURES = 19


# POWER VALUE
env = EnvConfig()
light_cfg = env.ROBOTS["LIGHT"]  # light is more efficient
heavy_cfg = env.ROBOTS["HEAVY"]

MIN_EFFICIENCY = (
    1.05
    * (heavy_cfg.DIG_COST / heavy_cfg.DIG_RESOURCE_GAIN)
    / (light_cfg.DIG_COST / light_cfg.DIG_RESOURCE_GAIN)
)

ORE_TO_POWER = (light_cfg.DIG_COST * MIN_EFFICIENCY) / light_cfg.DIG_RESOURCE_GAIN
ICE_TO_POWER = (light_cfg.DIG_COST * MIN_EFFICIENCY) / light_cfg.DIG_RESOURCE_GAIN

ORE_TO_POWER_HV = (heavy_cfg.DIG_COST * MIN_EFFICIENCY) / heavy_cfg.DIG_RESOURCE_GAIN
ICE_TO_POWER_HV = (heavy_cfg.DIG_COST * MIN_EFFICIENCY) / heavy_cfg.DIG_RESOURCE_GAIN

METAL_TO_POWER = ORE_TO_POWER * env.ORE_METAL_RATIO
WATER_TO_POWER = ICE_TO_POWER * env.ICE_WATER_RATIO

LIGHT_TO_POWER = light_cfg.METAL_COST * METAL_TO_POWER
HEAVY_TO_POWER = heavy_cfg.METAL_COST * METAL_TO_POWER

day = env.DAY_LENGTH
cycle = env.CYCLE_LENGTH

LIGHT_POWER_PROD = day / cycle * light_cfg.CHARGE
HEAVY_POWER_PROD = day / cycle * heavy_cfg.CHARGE

LIGHT_ROI = LIGHT_TO_POWER / LIGHT_POWER_PROD
HEAVY_ROI = HEAVY_TO_POWER / HEAVY_POWER_PROD


HEAVY_DIG_EFFICIENCY = ORE_TO_POWER_HV / ORE_TO_POWER
MAX_FREE_LIGHT_MOVEMENT = 19
