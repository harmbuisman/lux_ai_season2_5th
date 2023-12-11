from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from lux.utils import lprint
from lux.constants import COORD_MAP, NUM_FEATURES

_EMPTY_SERIES = None


def get_empty_series():
    global _EMPTY_SERIES
    if _EMPTY_SERIES is None:
        _EMPTY_SERIES = _create_empty_series()
    return _EMPTY_SERIES.copy()


def _create_empty_series():
    tags = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            if abs(x) == 2 and abs(y) == 2:
                continue
            tags.append(str((x, y)))

    names = [
        "rubble",
        "ice",
        "ore",
        "heavy",
        "power",
        "cargo_value",
        "distance_factory",
        "distance_opponent_factory",
        "is_move",
        "is_dig",
        "other_action",
        "is_up",
        "is_down",
        "is_left",
        "is_right",
        "is_wait",
        "my_team",
        "is_factory",
        "factory_power",
    ]

    point_features = [" ".join(f) for f in product(tags, names)]

    obs_features = ["target"]  # "step", "unit_id", "point", ]
    return pd.Series(index=obs_features + point_features, dtype=float)


def observation_from_feature_dict(feature_dict):  # , step, unit):
    case = get_empty_series()
    for p, items in feature_dict.items():
        for k, v in items.items():
            case.loc[f"{p} {k}"] = v
    # case.loc["step"] = step
    # case.loc["unit_id"] = unit.unit_id
    # case.loc["point"] = unit.point
    return case


def train_combat_model(agent):
    # Split the data into train and test sets
    mat = agent.combat_observations
    full = mat[: agent.n_observations[0]]
    complete = full[:, -1] >= 0
    full = full[complete]

    full_augmented = get_all_rotations(full)

    X = full_augmented[:, :-1]
    y = full_augmented[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Set the hyperparameters
    # params = {
    #     'objective': 'multi:softmax',
    #     'num_class': 6,
    #     'tree_method': 'auto'
    # }

    # Train the model
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    return model


def get_all_rotations(full):
    size = len(full)

    enriched = np.tile(full, (4, 1))
    flip_x_coord = {COORD_MAP[(x, y)]: COORD_MAP[(-x, y)] for x, y in COORD_MAP.keys()}
    flip_x_dir = {0: 0, 1: 1, 2: 4, 3: 3, 4: 2}

    flip_y_coord = {COORD_MAP[(x, y)]: COORD_MAP[(x, -y)] for x, y in COORD_MAP.keys()}
    flip_y_dir = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}

    flip_xy_coord = {
        COORD_MAP[(x, y)]: COORD_MAP[(-x, -y)] for x, y in COORD_MAP.keys()
    }
    flip_xy_dir = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2}

    mappings = {
        1: {"coord": flip_x_coord, "dir": flip_x_dir},
        2: {"coord": flip_y_coord, "dir": flip_y_dir},
        3: {"coord": flip_xy_coord, "dir": flip_xy_dir},
    }

    for i in range(1, 4):
        start = size * i
        end = size * (i + 1)

        coord_map = mappings[i]["coord"]
        for old, new in coord_map.items():
            if old != new:
                enriched[start:end, new] = enriched[0:size, old]

        dir_map = mappings[i]["dir"]
        enriched[start:end, -1] = np.vectorize(dir_map.get)(enriched[:size, -1])
    return enriched
